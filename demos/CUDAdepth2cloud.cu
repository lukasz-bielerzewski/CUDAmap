#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/cuda/point_cloud.h>
#include <thrust/device_vector.h>
#include <pcl/point_types.h>
#include <omp.h>
#include <iostream>
#include <opencv2/core/cuda.hpp>

#include "CUDAmap.h"

__device__ void getPoint(int x, int y, uint16_t depth, float3& point, const double focalAxis1, const double focalAxis2, double depthFactor)
{
    double realDepth = depth * depthFactor;
    point.x = (static_cast<float>(x) - focalAxis1) * realDepth / focalAxis1;
    point.y = (static_cast<float>(y) - focalAxis2) * realDepth / focalAxis2;
    point.z = realDepth;
}

__global__ void depthToPointCloudKernel(
    const uint16_t* depthData, size_t depthStep,
    const uchar3* colorData, size_t colorStep,
    int width, int height,
    const double focalAxis1, const double focalAxis2, double depthFactor,
    double minDepth, double maxDepth,
    SimplePoint* outputCloud)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uint16_t depthValue = depthData[y * depthStep / sizeof(uint16_t) + x];

    float3 pointxyz;
    getPoint(x, y, depthValue, pointxyz, focalAxis1, focalAxis2, depthFactor);

    if (pointxyz.z > minDepth && pointxyz.z < maxDepth) {
        uchar3 color = colorData[y * colorStep / sizeof(uchar3) + x];

        int outputIdx = y * width + x;
        outputCloud[outputIdx] = {pointxyz.x, pointxyz.y, pointxyz.z, static_cast<float>(color.z), static_cast<float>(color.y), static_cast<float>(color.x), 255};
    }
}

__host__ grabber::PointCloud CUDAdepth2cloud(const cv::Mat& depthImage, const cv::Mat& colorImage, grabber::CameraModel &kinectModel, double depthFactor, double minDepth, double maxDepth)
{
    // Allocate memory on GPU and transfer data
    cv::cuda::GpuMat gpuDepthImage, gpuColorImage;

    gpuDepthImage.upload(cv::InputArray(depthImage));
    gpuColorImage.upload(cv::InputArray(colorImage));

    // Allocate memory for the output cloud on GPU
    int cloudSize = depthImage.cols * depthImage.rows;
    SimplePoint* gpuOutputCloud;
    cudaMalloc(reinterpret_cast<void**>(&gpuOutputCloud), cloudSize * sizeof(SimplePoint));

    // Launch kernel
    dim3 blockSize(16, 16); // Example block size
    dim3 gridSize((depthImage.cols + blockSize.x - 1) / blockSize.x, (depthImage.rows + blockSize.y - 1) / blockSize.y);

    //std::cout << "(uint16_t*)gpuDepthImage.ptr() " << (uint16_t*)gpuDepthImage.ptr() << std::endl;
    //std::cout << "gpuDepthImage.step " << gpuDepthImage.step << std::endl;
    //std::cout << "(uchar3*)gpuColorImage.ptr() " << (uchar3*)gpuColorImage.ptr() << std::endl;
    //std::cout << "gpuColorImage.step " << gpuColorImage.step << std::endl;
    //std::cout << "gpuDepthImage.cols " << gpuDepthImage.cols << std::endl;
    //std::cout << "gpuDepthImage.rows " << gpuDepthImage.rows << std::endl;
    //std::cout << "kinectModel.focalAxis " << kinectModel.focalAxis[0] << " " << kinectModel.focalAxis[1] << std::endl;
    //std::cout << "static_cast<float>(depthFactor) " << static_cast<float>(depthFactor) << std::endl;
    //std::cout << "static_cast<float>(minDepth) " << static_cast<float>(minDepth) << std::endl;
    //std::cout << "static_cast<float>(maxDepth) " << static_cast<float>(maxDepth) << std::endl;

    depthToPointCloudKernel<<<gridSize, blockSize>>>(
        (uint16_t*)gpuDepthImage.ptr(), gpuDepthImage.step,
        (uchar3*)gpuColorImage.ptr(), gpuColorImage.step,
        gpuDepthImage.cols, gpuDepthImage.rows,
        kinectModel.focalAxis[0], kinectModel.focalAxis[1], static_cast<float>(depthFactor), static_cast<float>(minDepth), static_cast<float>(maxDepth),
        gpuOutputCloud);

    cudaDeviceSynchronize();

    // Transfer results back to CPU
    std::vector<SimplePoint> outputCloudData(cloudSize);
    cudaMemcpy(outputCloudData.data(), gpuOutputCloud, cloudSize * sizeof(SimplePoint), cudaMemcpyDeviceToHost);

    // Convert the SimplePoint array to pcl::PointCloud<pcl::PointXYZRGBA>
    grabber::PointCloud cloud;

    #pragma omp parallel for
    for (const auto& pt : outputCloudData)
    {
        pcl::PointXYZRGBA pclPt;
        pclPt.x = pt.x;
        pclPt.y = pt.y;
        pclPt.z = pt.z;
        pclPt.r = pt.r;
        pclPt.g = pt.g;
        pclPt.b = pt.b;
        pclPt.a = pt.a;
        cloud.push_back(pclPt);
    }

    // Explicitly release GPU memory
    gpuDepthImage.release();
    gpuColorImage.release();

    // Cleanup GPU memory
    cudaFree(gpuOutputCloud);

    return cloud;
}
