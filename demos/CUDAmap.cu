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
__constant__ double gpuPHCPModel00;
__constant__ double gpuPHCPModel02;
__constant__ double gpuPHCPModel11;
__constant__ double gpuPHCPModel12;
__constant__ int gpuOffsetX;
__constant__ int gpuOffsetY;
__constant__ int gpuImageRows;
__constant__ int gpuImageCols;
__constant__ double gpuMinDepth;
__constant__ double gpuMaxDepth;
__constant__ double gpuDepthFactor;

/// Global variable definition
__device__ int globalCounter;

/// Allocate memory on GPU and send constant parameters
__host__ void allocateMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat &depthImage, cv::Mat &colorImage,
                             grabber::CameraModel &kinectModel, double depthFactor, double minDepth, double maxDepth,
                             SimplePoint* &gpuPointCloud)
{
    // Allocate memory for depth image container on GPU
    gpuDepthImage.create(depthImage.rows, depthImage.cols, CV_16UC1);

    // Allocate memory for color image container on GPU
    gpuColorImage.create(colorImage.rows, colorImage.cols, CV_8UC3);

    // Camera parameters
    double PHCPModel00 = kinectModel.PHCPModel(0, 0);
    double PHCPModel02 = kinectModel.PHCPModel(0, 2);
    double PHCPModel11 = kinectModel.PHCPModel(1, 1);
    double PHCPModel12 = kinectModel.PHCPModel(1, 2);

    // Send PHCPModel parameters to constant GPU memory
    cudaMemcpyToSymbol(gpuPHCPModel00, &PHCPModel00, sizeof(double));
    cudaMemcpyToSymbol(gpuPHCPModel02, &PHCPModel02, sizeof(double));
    cudaMemcpyToSymbol(gpuPHCPModel11, &PHCPModel11, sizeof(double));
    cudaMemcpyToSymbol(gpuPHCPModel12, &PHCPModel12, sizeof(double));

    // Calculate offsets
    int offsetX = int(depthImage.cols - 2 * kinectModel.focalAxis[0]);
    int offsetY = int(depthImage.rows - 2 * kinectModel.focalAxis[1]);

    // Initialize offsets values on GPU
    cudaMemcpyToSymbol(gpuOffsetX, &offsetX, sizeof(int));
    cudaMemcpyToSymbol(gpuOffsetY, &offsetY, sizeof(int));
    cudaMemcpyToSymbol(gpuImageRows, &depthImage.rows, sizeof(int));
    cudaMemcpyToSymbol(gpuImageCols, &depthImage.cols, sizeof(int));
    cudaMemcpyToSymbol(gpuMinDepth, &minDepth, sizeof(double));
    cudaMemcpyToSymbol(gpuMaxDepth, &maxDepth, sizeof(double));
    cudaMemcpyToSymbol(gpuDepthFactor, &depthFactor, sizeof(double));

    // Calculate size of point cloud
    uint cloudSize = depthImage.cols * depthImage.rows;

    // Allocate memory for point cloud on GPU
    cudaMalloc(&gpuPointCloud, cloudSize * sizeof(SimplePoint));
}

/// Deallocate memory from GPU
__host__ void deallocateMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, SimplePoint* &gpuPointCloud)
{
    // Deallocate memory of depth image container on GPU
    gpuDepthImage.release();

    // Deallocate memory of color image container on GPU
    gpuColorImage.release();

    // Deallocate memory of point cloud container on GPU
    cudaFree(gpuPointCloud);
}

/// Send data of a current images to GPU
__host__ void sendImageDataToMemory(cv::cuda::GpuMat &gpuDepthImage, cv::Mat& depthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat& colorImage)
{
    // Send depth image to GPU
    gpuDepthImage.upload(cv::InputArray(depthImage));

    // Send color image to GPU
    gpuColorImage.upload(cv::InputArray(colorImage));

    // Reset counter of points
    int counterValue = 0;
    cudaMemcpyToSymbol(globalCounter, &counterValue, sizeof(int), 0, cudaMemcpyHostToDevice);
}

__host__ void getPointCloud(SimplePoint* &gpuPointCloud, SimplePoint* &pointCloud, cv::Mat& depthImage)
{
    uint cloudSize = depthImage.cols * depthImage.rows;
    cudaMemcpy(pointCloud, &gpuPointCloud, cloudSize * sizeof(SimplePoint), cudaMemcpyDeviceToHost);
}

__global__ void KernelDepthToCloud(size_t boundingBoxXFirst, size_t boundingBoxXSecond, size_t boundingBoxYFirst, size_t boundingBoxYSecond,
                                   const uchar3* colorData, const uint16_t* depthData, SimplePoint* gpuPointCloud)
{
    // Get kernel indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check boundries
    if(y > int(boundingBoxYFirst) && y < int(boundingBoxYSecond) && x > int(boundingBoxXFirst) && x < int(boundingBoxXSecond))
    {
        // check coordinates taking offsets into account
        int coordU = y + gpuOffsetY;
        int coordV = x + gpuOffsetX;

        if(coordU < 0)
        {
            coordU = 0;
        }
        if(coordU >= gpuImageRows)
        {
            coordU = gpuImageRows - 1;
        }
        if(coordV < 0)
        {
            coordV = 0;
        }
        if(coordV >= gpuImageCols)
        {
            coordV = gpuImageCols - 1;
        }

        // Calculate index
        int imageIndex = y * gpuImageCols + x;

        // Access pixel data
        uchar3 colorPixel = colorData[imageIndex];
        uint16_t depthValue = depthData[imageIndex];
        double depth = (double)depthValue * gpuDepthFactor;

        // Get real coordinates
        double Rx = depth * (coordU * gpuPHCPModel00 + gpuPHCPModel02);
        double Ry = depth * (coordV * gpuPHCPModel11 + gpuPHCPModel12);

        //printf("depth: %f, Rx: %f, Ry: %f \n", depth, Rx, Ry);

        // Check limits of depth and add data to point cloud
        if(depth > gpuMinDepth && depth < gpuMaxDepth)
        {
            // Count the number of points
            int outputIndex = atomicAdd(&globalCounter, 1);

            gpuPointCloud[outputIndex].x = (float)Rx;
            gpuPointCloud[outputIndex].y = (float)Ry;
            gpuPointCloud[outputIndex].z = (float)depth;

            printf("depth: %f, Rx: %f, Ry: %f \n", gpuPointCloud[outputIndex].z, gpuPointCloud[outputIndex].x, gpuPointCloud[outputIndex].y);

            gpuPointCloud[outputIndex].r = colorPixel.z;
            gpuPointCloud[outputIndex].g = colorPixel.y;
            gpuPointCloud[outputIndex].b = colorPixel.x;
            gpuPointCloud[outputIndex].a = 255.f;
        }
    }
}

__host__ void CUDAmap(cv::cuda::GpuMat &gpuDepthImage, cv::Mat& depthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat& colorImage,
                      SimplePoint* &gpuPointCloud, SimplePoint* &pointCloud)
{
    // Send current data to GPU
    sendImageDataToMemory(gpuDepthImage, depthImage, gpuColorImage, colorImage);

    // Kernel launch parameters
    dim3 blockSize(16, 16); // Example block size
    dim3 gridSize((depthImage.cols + blockSize.x - 1) / blockSize.x, (depthImage.rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    KernelDepthToCloud<<<gridSize, blockSize>>>(0, depthImage.cols, 0, depthImage.rows,
                                               gpuColorImage.ptr<uchar3>(), gpuDepthImage.ptr<uint16_t>(), gpuPointCloud);

    // Wait for all kernels to finish
    cudaDeviceSynchronize();

    // Get point cloud data from GPU
    getPointCloud(gpuPointCloud, pointCloud, depthImage);
}
