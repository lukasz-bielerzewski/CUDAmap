#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/cuda/point_cloud.h>
#include <thrust/device_vector.h>
#include <pcl/point_types.h>
#include <opencv2/core/cuda.hpp>
#include <grabber_defs.h>

#include "CUDAmap.h"

/// Constant variables memory allocation, used for performance optimization; it is automaticly managed by CUDA runtime
__constant__ double gpuCamPoseArray[16];

/// Function for preallocating memory on GPU for containers
__host__ void allocateTransformPointsMemory(SimplePoint* &gpuInputPointCloud, SimplePoint* &gpuOutputPointCloud, int &cloudSize)
{
    // Allocate memory for input point cloud on GPU
    cudaMalloc(&gpuInputPointCloud, cloudSize * sizeof(SimplePoint));

    // Allocate memory for output point cloud on GPU
    cudaMalloc(&gpuOutputPointCloud, cloudSize * sizeof(SimplePoint));
}

/// Function to deallocate memory from GPU
__host__ void deallocateTransformPointsMemory(SimplePoint* &gpuInputPointCloud, SimplePoint* &gpuOutputPointCloud)
{
    // Deallocate memory of input point cloud container on GPU
    cudaFree(gpuInputPointCloud);

    // Deallocate memory of output point cloud container on GPU
    cudaFree(gpuOutputPointCloud);
}

__host__ void sendDataToGPU(double* &camPoseArray, SimplePoint* &pointCloud, SimplePoint* &gpuInputPointCloud, int &cloudSize)
{
    // Send camera pose to constant GPU memory
    cudaMemcpyToSymbol(gpuCamPoseArray, camPoseArray, 16 * sizeof(double));

    // Send point cloud data to GPU memory
    cudaMemcpy(gpuInputPointCloud, pointCloud, cloudSize * sizeof(SimplePoint), cudaMemcpyHostToDevice);
}

// __host__ void getDataFromGPU(SimplePoint* &outputPointCloud, SimplePoint* &gpuOutputPointCloud, int &cloudSize)
// {
//     // Get point cloud data from GPU memory
//     cudaMemcpy(outputPointCloud, gpuOutputPointCloud, cloudSize * sizeof(SimplePoint), cudaMemcpyDeviceToHost);
// }

/// Kernel
__global__ void KernelTransformPoints(SimplePoint *gpuInputPointCloud, SimplePoint *gpuOutputPointCloud, int cloudSize)
{
    // Kernel index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < cloudSize)
    {
        // Get the input point
        SimplePoint &point = gpuInputPointCloud[idx];

        // Apply transformation
        double x = gpuCamPoseArray[0] * point.x + gpuCamPoseArray[1] * point.y + gpuCamPoseArray[2] * point.z + gpuCamPoseArray[3];
        double y = gpuCamPoseArray[4] * point.x + gpuCamPoseArray[5] * point.y + gpuCamPoseArray[6] * point.z + gpuCamPoseArray[7];
        double z = gpuCamPoseArray[8] * point.x + gpuCamPoseArray[9] * point.y + gpuCamPoseArray[10] * point.z + gpuCamPoseArray[11];

        // Set output point
        gpuOutputPointCloud[idx].x = x;
        gpuOutputPointCloud[idx].y = -y;
        gpuOutputPointCloud[idx].z = z;
        gpuOutputPointCloud[idx].r = point.r;
        gpuOutputPointCloud[idx].g = point.g;
        gpuOutputPointCloud[idx].b = point.b;
        gpuOutputPointCloud[idx].a = point.a;
    }
}

/// Main function
__host__ void CUDATransformPoints(double* camPoseArray, SimplePoint* &pointCloud, SimplePoint* &outputPointCloud,
                                  SimplePoint *gpuInputPointCloud, SimplePoint* &gpuOutputPointCloud, int &cloudSize)
{
    // Send current data to GPU
    sendDataToGPU(camPoseArray, pointCloud, gpuInputPointCloud, cloudSize);

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (cloudSize + blockSize - 1) / blockSize;

    // Launch kernel
    KernelTransformPoints<<<numBlocks, blockSize>>>(gpuInputPointCloud, gpuOutputPointCloud, cloudSize);

    // Wait for all kernels to finish
    cudaDeviceSynchronize();

    // Get point cloud data from GPU
    // getDataFromGPU(outputPointCloud, gpuOutputPointCloud, cloudSize);
}
