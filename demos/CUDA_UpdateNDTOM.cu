#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/cuda/point_cloud.h>
#include <thrust/device_vector.h>
#include <pcl/point_types.h>
#include <opencv2/core/cuda.hpp>
#include <grabber_defs.h>
#include <Eigen/Dense>

#include "CUDAmap.h"

struct Mat33
{
    double elements[9]; // Row-major order

    // Helper function to add another matrix
    __device__ void add(const Mat33& other)
    {
        for(int i = 0; i < 9; ++i)
        {
            elements[i] += other.elements[i];
        }
    }
};

struct Vec3d
{
    double x, y, z;
};

// Convert Eigen::Vector3d to Vec3d
Vec3d toVec3d(const Eigen::Vector3d& eigenVec)
{
    return {eigenVec.x(), eigenVec.y(), eigenVec.z()};
}

// Parallel reduction to compute sum of points (newMeanSum)
__global__ void calculateMeanKernel(Vec3d* points, int numPoints, Vec3d* meanSum)
{
    extern __shared__ Vec3d sdata1[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared mem from global mem
    sdata1[tid] = (i < numPoints) ? points[i] : Vec3d{0, 0, 0};
    __syncthreads();

    // Do reduction in shared mem
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if(tid < s)
        {
            sdata1[tid].x += sdata1[tid + s].x;
            sdata1[tid].y += sdata1[tid + s].y;
            sdata1[tid].z += sdata1[tid + s].z;
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if(tid == 0)
    {
        atomicAdd(&meanSum->x, sdata1[0].x);
        atomicAdd(&meanSum->y, sdata1[0].y);
        atomicAdd(&meanSum->z, sdata1[0].z);
    }
}

// Parallel reduction to compute variance sum (newVarSum)
__global__ void calculateVarianceKernel(Vec3d* points, int numPoints, Vec3d* mean, Mat33* varSum)
{
    extern __shared__ Mat33 sdata2[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if(tid < 9)
    {
        sdata2[0].elements[tid] = 0;
    }
    __syncthreads();

    // Calculate variance for each point and accumulate in shared memory
    if(i < numPoints)
    {
        Vec3d point = points[i];
        Vec3d diff = {point.x - mean->x, point.y - mean->y, point.z - mean->z};

        Mat33 var;
        var.elements[0] = diff.x * diff.x; // xx
        var.elements[1] = diff.x * diff.y; // xy
        var.elements[2] = diff.x * diff.z; // xz
        var.elements[3] = diff.y * diff.x; // yx
        var.elements[4] = diff.y * diff.y; // yy
        var.elements[5] = diff.y * diff.z; // yz
        var.elements[6] = diff.z * diff.x; // zx
        var.elements[7] = diff.z * diff.y; // zy
        var.elements[8] = diff.z * diff.z; // zz

        sdata2[0].add(var);
    }
    __syncthreads();

    // Use the first thread to accumulate the result in global memory
    if(tid == 0)
    {
        for(int j = 0; j < 9; ++j)
        {
            atomicAdd(&varSum->elements[j], sdata2[0].elements[j]);
        }
    }
}

void mapping::Voxel::CUDA_UpdateNDTOM()
{
    // Declare device pointers
    Vec3d* d_points;
    Vec3d* d_mean;
    Mat33* d_varSum;

    // Allocate memory on device
    cudaMalloc(&d_points, numPoints * sizeof(Vec3d));
    cudaMalloc(&d_mean, sizeof(Vec3d));
    cudaMalloc(&d_varSum, sizeof(Mat33));

    // Copy points from host to device
    cudaMemcpy(d_points, &points, numPoints * sizeof(Vec3d), cudaMemcpyHostToDevice);

    // Initialize d_mean and d_varSum
    Vec3d zeroVec3d = {0, 0, 0};
    Mat33 zeroMat33;
    memset(zeroMat33.elements, 0, 9 * sizeof(double));
    cudaMemcpy(d_mean, &zeroVec3d, sizeof(Vec3d), cudaMemcpyHostToDevice);
    cudaMemcpy(d_varSum, &zeroMat33, sizeof(Mat33), cudaMemcpyHostToDevice);

    // Calculate block and grid sizes
    int threadsPerBlock = 256; // Example value, adjust as needed
    int blocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    // Call kernels for mean and variance
    calculateMeanKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(Vec3d)>>>(d_points, numPoints, d_mean);
    calculateVarianceKernel<<<blocks, threadsPerBlock, sizeof(Mat33)>>>(d_points, numPoints, d_mean, d_varSum);

    // Copy results back to host and finish calculations
    Vec3d h_mean;
    Mat33 h_varSum;
    cudaMemcpy(&h_mean, d_mean, sizeof(Vec3d), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_varSum, d_varSum, sizeof(Mat33), cudaMemcpyDeviceToHost);

    // Update the mean
    mean = Eigen::Vector3d(h_mean.x, h_mean.y, h_mean.z);

    // Update the variance
    var(0, 0) = h_varSum.elements[0]; // xx
    var(0, 1) = h_varSum.elements[1]; // xy
    var(0, 2) = h_varSum.elements[2]; // xz
    var(1, 0) = h_varSum.elements[3]; // yx
    var(1, 1) = h_varSum.elements[4]; // yy
    var(1, 2) = h_varSum.elements[5]; // yz
    var(2, 0) = h_varSum.elements[6]; // zx
    var(2, 1) = h_varSum.elements[7]; // zy
    var(2, 2) = h_varSum.elements[8]; // zz

    // Normalize the variance
    var /= static_cast<double>(numPoints - 1);

    // Update sample number
    sampNumber += numPoints;

    // Additional checks or updates based on the new state of the voxel
    // For example, checking if the mean is within a certain range and setting probability to zero
    if(mean.x() < 0.1 && mean.y() < 0.1 && mean.z() < 0.1)
    {
        probability = 0.0;
    }

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_mean);
    cudaFree(d_varSum);
}
