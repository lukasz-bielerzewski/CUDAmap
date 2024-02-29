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
// POINT CLOUD
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
__constant__ size_t gpuDepthImageStride;
__constant__ size_t gpuColorImageStride;

// MAPPING
__constant__ double poseRot00;
__constant__ double poseRot01;
__constant__ double poseRot02;
__constant__ double poseRot10;
__constant__ double poseRot11;
__constant__ double poseRot12;
__constant__ double poseRot20;
__constant__ double poseRot21;
__constant__ double poseRot22;
__constant__ double posePosX;
__constant__ double posePosY;
__constant__ double posePosZ;

//=======================================================================================================================================================================
// POINT CLOUD FUNCTIONALITY
//=======================================================================================================================================================================

/// Global variable definition
__device__ int globalCounter = 0;

/// Allocate memory on GPU and send constant parameters
__host__ void allocateMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat &depthImage, cv::Mat &colorImage,
                             grabber::CameraModel &kinectModel, double depthFactor, double minDepth, double maxDepth,
                             SimplePoint* &gpuPointCloud, SimplePoint* &gpuTransformedPointCloud)
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

    // Calculate and allocate depth image stride
    size_t depthStride = gpuDepthImage.step1();
    cudaMemcpyToSymbol(gpuDepthImageStride, &depthStride, sizeof(size_t));

    // Calculate and allocate color image stride
    size_t colorStride = gpuColorImage.step1();
    cudaMemcpyToSymbol(gpuColorImageStride, &colorStride, sizeof(size_t));

    // Calculate size of point cloud
    uint cloudSize = depthImage.cols * depthImage.rows;

    // Allocate memory for point cloud on GPU
    cudaMalloc(&gpuPointCloud, cloudSize * sizeof(SimplePoint));

    // Allocate memory for transformed point cloud on GPU
    cudaMalloc(&gpuTransformedPointCloud, cloudSize * sizeof(SimplePoint));
}

/// Deallocate memory from GPU
__host__ void deallocateMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, SimplePoint* &gpuPointCloud, SimplePoint* &gpuTransformedPointCloud)
{
    // Deallocate memory of depth image container on GPU
    gpuDepthImage.release();

    // Deallocate memory of color image container on GPU
    gpuColorImage.release();

    // Deallocate memory of point cloud container on GPU
    cudaFree(gpuPointCloud);

    // Deallocate memory of output point cloud container on GPU
    cudaFree(gpuTransformedPointCloud);
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

__host__ void getPointCloudSize(int &pointCloudSize)
{
    cudaMemcpyFromSymbol(&pointCloudSize, globalCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);
}

__global__ void KernelDepthToCloud(size_t boundingBoxXFirst, size_t boundingBoxXSecond, size_t boundingBoxYFirst, size_t boundingBoxYSecond,
                                   const uchar3* colorData, const uint16_t* depthData, SimplePoint* gpuPointCloud)
{
    // Get kernel indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check indices boundries
    if (x < gpuImageCols && y < gpuImageRows)
    {

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
            int depthImageIndex = y * gpuDepthImageStride + x;
            int colorImageIndex = y * gpuColorImageStride + x;

            // Access pixel data
            uchar3 colorPixel = colorData[colorImageIndex];
            uint16_t depthValue = depthData[depthImageIndex];
            double depth = (double)depthValue * gpuDepthFactor;

            // Get real coordinates
            double Rx = depth * (coordU * gpuPHCPModel00 + gpuPHCPModel02);
            double Ry = depth * (coordV * gpuPHCPModel11 + gpuPHCPModel12);

            // Check limits of depth and add data to point cloud
            if(depth > gpuMinDepth && depth < gpuMaxDepth)
            {
                // Count the number of points
                int outputIndex = atomicAdd(&globalCounter, 1);

                gpuPointCloud[outputIndex].x = (float)Rx;
                gpuPointCloud[outputIndex].y = (float)Ry;
                gpuPointCloud[outputIndex].z = (float)depth;

                //printf("depth: %f, Rx: %f, Ry: %f \n", gpuPointCloud[outputIndex].z, gpuPointCloud[outputIndex].x, gpuPointCloud[outputIndex].y);

                gpuPointCloud[outputIndex].r = colorPixel.z;
                gpuPointCloud[outputIndex].g = colorPixel.y;
                gpuPointCloud[outputIndex].b = colorPixel.x;
                gpuPointCloud[outputIndex].a = 255.f;
            }
        }
    }
}

//=======================================================================================================================================================================
// MAPPING FUNCTIONALITY
//=======================================================================================================================================================================

__host__ void sendCameraDataToGPU(walkers::Mat34 &camPose)
{
    // Send camera pose to constant GPU memory
    double Rot00 = camPose(0,0);
    double Rot01 = camPose(0,1);
    double Rot02 = camPose(0,2);
    double Rot10 = camPose(1,0);
    double Rot11 = camPose(1,1);
    double Rot12 = camPose(1,2);
    double Rot20 = camPose(2,0);
    double Rot21 = camPose(2,1);
    double Rot22 = camPose(2,2);
    double PosX = camPose(0,3);
    double PosY = camPose(1,3);
    double PosZ = camPose(2,3);
    cudaMemcpyToSymbol(poseRot00, &Rot00, sizeof(double));
    cudaMemcpyToSymbol(poseRot01, &Rot01, sizeof(double));
    cudaMemcpyToSymbol(poseRot02, &Rot02, sizeof(double));
    cudaMemcpyToSymbol(poseRot10, &Rot10, sizeof(double));
    cudaMemcpyToSymbol(poseRot11, &Rot11, sizeof(double));
    cudaMemcpyToSymbol(poseRot12, &Rot12, sizeof(double));
    cudaMemcpyToSymbol(poseRot20, &Rot20, sizeof(double));
    cudaMemcpyToSymbol(poseRot21, &Rot21, sizeof(double));
    cudaMemcpyToSymbol(poseRot22, &Rot22, sizeof(double));
    cudaMemcpyToSymbol(posePosX, &PosX, sizeof(double));
    cudaMemcpyToSymbol(posePosY, &PosY, sizeof(double));
    cudaMemcpyToSymbol(posePosZ, &PosZ, sizeof(double));
}

__host__ void getDataFromGPU(SimplePoint* &outputPointCloud, SimplePoint* &gpuTransformedPointCloud, int &cloudSize)
{
    // Get point cloud data from GPU memory
    cudaMemcpy(outputPointCloud, gpuTransformedPointCloud, cloudSize * sizeof(SimplePoint), cudaMemcpyDeviceToHost);
}

/// Kernel
__global__ void KernelTransformPoints(SimplePoint* gpuPointCloud, SimplePoint* gpuTransformedPointCloud, int pointCloudSize)
{
    // Kernel index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < pointCloudSize)
    {
        // Get the input point
        SimplePoint &point = gpuPointCloud[idx];

        // Apply transformation
        double x = poseRot00 * point.x + poseRot01 * point.y + poseRot02 * point.z + posePosX;
        double y = poseRot10 * point.x + poseRot11 * point.y + poseRot12 * point.z + posePosY;
        double z = poseRot20 * point.x + poseRot21 * point.y + poseRot22 * point.z + posePosZ;

        // Set output point
        gpuTransformedPointCloud[idx].x = x;
        gpuTransformedPointCloud[idx].y = -y;
        gpuTransformedPointCloud[idx].z = z;
        gpuTransformedPointCloud[idx].r = point.r;
        gpuTransformedPointCloud[idx].g = point.g;
        gpuTransformedPointCloud[idx].b = point.b;
        gpuTransformedPointCloud[idx].a = point.a;
    }
}

//=======================================================================================================================================================================
// MAIN FUNCTIONALITY
//=======================================================================================================================================================================

__host__ void CUDAmap(cv::cuda::GpuMat &gpuDepthImage, cv::Mat& depthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat& colorImage,
                      SimplePoint* &gpuPointCloud, walkers::Mat34 &camPose, SimplePoint* &gpuTransformedPointCloud,
                    TimeMeasurements &timepointtrans, TimeMeasurements &timedepth2cloud)
{
    std::chrono::steady_clock::time_point begin_depth2load = std::chrono::steady_clock::now();

    // Send current data to GPU
    sendImageDataToMemory(gpuDepthImage, depthImage, gpuColorImage, colorImage);

    // Kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((depthImage.cols + blockSize.x - 1) / blockSize.x, (depthImage.rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    KernelDepthToCloud<<<gridSize, blockSize>>>(0, depthImage.cols, 0, depthImage.rows,
                                               gpuColorImage.ptr<uchar3>(), gpuDepthImage.ptr<uint16_t>(), gpuPointCloud);

    // Wait for all kernels to finish
    cudaDeviceSynchronize();

    std::chrono::steady_clock::time_point end_depth2load = std::chrono::steady_clock::now();
    timedepth2cloud.addMeasurement(end_depth2load-begin_depth2load);

    std::chrono::steady_clock::time_point begin_pointtrans = std::chrono::steady_clock::now();

    // Get Point Cloud size
    int pointCloudSize = 0;
    getPointCloudSize(pointCloudSize);

    // Send current camera data to GPU
    sendCameraDataToGPU(camPose);

    // Kernel launch parameters
    int blockSizeT = 256;
    int numBlocksT = (pointCloudSize + blockSizeT - 1) / blockSizeT;

    // Launch kernel
    KernelTransformPoints<<<numBlocksT, blockSizeT>>>(gpuPointCloud, gpuTransformedPointCloud, pointCloudSize);

    // Wait for all kernels to finish
    cudaDeviceSynchronize();

    std::chrono::steady_clock::time_point end_pointtrans = std::chrono::steady_clock::now();
    timepointtrans.addMeasurement(end_pointtrans-begin_pointtrans);
}
