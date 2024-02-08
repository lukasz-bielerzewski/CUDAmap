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
__constant__ double gpuPHCPModel[3][3];
__constant__ int gpuOffsetX;
__constant__ int gpuOffsetY;


/// Function used outside of CUDA code for preallocating memory on GPU for further processes of transforming image matrices into point clouds
/// Additionally initialize constant memory variables on GPU
__host__ void allocateDepth2CloudMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat &depthImage, cv::Mat &colorImage,
                                        SimplePoint* &gpuPointCloud, grabber::CameraModel &kinectModel, int &cloudSize)
{
    // Allocate memory for depth image container on GPU
    gpuDepthImage.create(depthImage.rows, depthImage.cols, CV_16UC1);

    // Allocate memory for color image container on GPU
    gpuColorImage.create(colorImage.rows, colorImage.cols, CV_8UC3);

    // Calculate size of point cloud
    cloudSize = depthImage.cols * depthImage.rows;

    // Allocate memory for point cloud on GPU
    cudaMalloc(&gpuPointCloud, cloudSize * sizeof(SimplePoint));

    // Create simplified PHCPModel matrix
    double simplePHCPModel[3][3];

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            simplePHCPModel[i][j] = kinectModel.PHCPModel(i, j);
        }
    }

    // Send PHCPModel parameters matrix to constant GPU memory
    cudaMemcpyToSymbol(gpuPHCPModel, simplePHCPModel, sizeof(simplePHCPModel));

    // Calculate offsets
    int offsetX = int(depthImage.cols - 2 * kinectModel.focalAxis[0]);
    int offsetY = int(depthImage.rows - 2 * kinectModel.focalAxis[1]);

    // Initialize offsets values on GPU
    cudaMemcpyToSymbol(gpuOffsetX, &offsetX, sizeof(offsetX));
    cudaMemcpyToSymbol(gpuOffsetY, &offsetY, sizeof(offsetY));
}

/// Function used outside of CUDA code for deallocation of memory on GPU
__host__ void deallocateDepth2CloudMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, SimplePoint* &gpuPointCloud)
{
    // Deallocate memory of depth image container on GPU
    gpuDepthImage.release();

    // Deallocate memory of color image container on GPU
    gpuColorImage.release();

    // Deallocate memory of point cloud container on GPU
    cudaFree(gpuPointCloud);
}

/// Function to send data to allocated memory on GPU
__host__ void sendData2Memory(cv::cuda::GpuMat &gpuDepthImage, cv::Mat &depthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat &colorImage)
{
    // Send depth image to GPU
    gpuDepthImage.upload(cv::InputArray(depthImage));

    // Send color image to GPU
    gpuColorImage.upload(cv::InputArray(colorImage));
}

/// Function to get data back from GPU
__host__ void getDataFromMemory(SimplePoint* &gpuPointCloud, SimplePoint* &pointCloud, int &cloudSize)
{
    cudaMemcpy(pointCloud, gpuPointCloud, cloudSize * sizeof(SimplePoint), cudaMemcpyDeviceToHost);
}

/// Calculate point real coordinates
__device__ void getPoint(int &u, int &v, double &depth, double* point3D)
{
    // Create the point vector (u, v, 1)
    double point[3] = {static_cast<double>(u), static_cast<double>(v), 1.0};

    // Perform matrix-vector multiplication: point3D = depth * gpuPHCPModel * point
    for(int i = 0; i < 3; ++i)
    {
        point3D[i] = 0;
        for(int j = 0; j < 3; ++j)
        {
            point3D[i] += static_cast<double>(depth) * gpuPHCPModel[i][j] * point[j];
        }
    }
}

/// Kernel
__global__ void KernelDepth2Cloud(size_t boundingBoxXFirst, size_t boundingBoxXSecond, size_t boundingBoxYFirst, size_t boundingBoxYSecond,
                                  int imageRows, int imageCols, const uchar3* colorData, const uint16_t* depthData,
                                  double minDepth, double maxDepth, double depthFactor, SimplePoint* gpuPointCloud)
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
        if(coordU >= imageRows)
        {
            coordU = imageRows - 1;
        }
        if(coordV < 0)
        {
            coordV = 0;
        }
        if(coordV >= imageCols)
        {
            coordV = imageCols - 1;
        }

        // Calculate index
        int imageIndex = y * imageCols + x;

        // Access pixel data
        uchar3 colorPixel = colorData[imageIndex];
        uint16_t depthValue = depthData[imageIndex];
        double depth = static_cast<double>(depthValue) * depthFactor;

        // Get real coordinates
        double point3D[3];
        getPoint(coordU, coordV, depth, point3D);

        // Check limits of depth and add data to point cloud
        if(point3D[2] > minDepth && point3D[2] < maxDepth)
        {
            gpuPointCloud[imageIndex].x = (float)point3D[0];
            gpuPointCloud[imageIndex].y = (float)point3D[1];
            gpuPointCloud[imageIndex].z = (float)point3D[2];
            gpuPointCloud[imageIndex].r = colorPixel.z;
            gpuPointCloud[imageIndex].g = colorPixel.y;
            gpuPointCloud[imageIndex].b = colorPixel.x;
            gpuPointCloud[imageIndex].a = 255.f;
        }
    }
}

/// Main function
__host__ void CUDADepth2Cloud(cv::cuda::GpuMat &gpuDepthImage, cv::Mat& depthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat& colorImage,
                              grabber::CameraModel &kinectModel, double depthFactor, double minDepth, double maxDepth,
                              SimplePoint* &gpuPointCloud, SimplePoint* &pointCloud, int &cloudSize)
{
    // Send current data to GPU
    sendData2Memory(gpuDepthImage, depthImage, gpuColorImage, colorImage);

    // Kernel launch parameters
    dim3 blockSize(16, 16); // Example block size
    dim3 gridSize((depthImage.cols + blockSize.x - 1) / blockSize.x, (depthImage.rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    KernelDepth2Cloud<<<gridSize, blockSize>>>(0, depthImage.cols, 0, depthImage.rows,
                                               depthImage.rows, depthImage.cols, gpuColorImage.ptr<uchar3>(), gpuDepthImage.ptr<uint16_t>(),
                                               minDepth, maxDepth, depthFactor, gpuPointCloud);

    // Wait for all kernels to finish
    cudaDeviceSynchronize();

    // Get point cloud data from GPU
    getDataFromMemory(gpuPointCloud, pointCloud, cloudSize);
}
