#ifndef CUDAMAP_H
#define CUDAMAP_H

#include "Defs/defs.h"
#include <tinyxml2.h>
#include "Visualizer/Qvisualizer.h"
#include "Mapping/elevationMap.h"
//do usuniecia
#include "Mapping/gaussmap.h"
// simulation
#include "Defs/simulator_defs.h"
//planner
#include "include/Defs/planner_defs.h"
// Utilities
#include "Utilities/recorder.h"
// Filtering
#include <GL/glut.h>
#include <qapplication.h>
#include <iostream>
#include <thread>
#include <random>
#include <pcl/io/ply_io.h>
#include <chrono>
#include "Utilities/TimeRange.h"
#include "Utilities/observer.h"


struct SimplePoint
{
    float x, y, z;
    uchar r, g, b, a;
};

///NEW CODE

void allocateMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat &depthImage, cv::Mat &colorImage,
                    grabber::CameraModel &kinectModel, double depthFactor, double minDepth, double maxDepth,
                    SimplePoint* &gpuPointCloud);

void deallocateMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, SimplePoint* &gpuPointCloud);

void CUDAmap(cv::cuda::GpuMat &gpuDepthImage, cv::Mat& depthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat& colorImage,
             SimplePoint* &gpuPointCloud, SimplePoint* &pointCloud);

///END OF NEW CODE

/// Depth to point cloud conversion
// Memory allocation
void allocateDepth2CloudMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat &depthImage, cv::Mat &colorImage,
                               SimplePoint* &gpuPointCloud, grabber::CameraModel &kinectModel, int &cloudSize);

void allocateTransformPointsMemory(SimplePoint* &gpuInputPointCloud, SimplePoint* &outputPointCloud, int &cloudSize);

// Memory deallocation
void deallocateDepth2CloudMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, SimplePoint* &gpuPointCloud);

void deallocateTransformPointsMemory(SimplePoint* &gpuInputPointCloud, SimplePoint* &outputPointCloud);

// Main functionality - kernel
void CUDADepth2Cloud(cv::cuda::GpuMat &gpuDepthImage, cv::Mat& depthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat& colorImage,
                     grabber::CameraModel &kinectModel, double depthFactor, double minDepth, double maxDepth,
                     SimplePoint* &gpuPointCloud, SimplePoint* &pointCloud, int &cloudSize);

void CUDATransformPoints(double* camPoseArray, SimplePoint* &pointCloud, SimplePoint* &outputPointCloud,
                         SimplePoint *gpuInputPointCloud, SimplePoint* &gpuOutputPointCloud, int &cloudSize);

#endif // CUDAMAP_H
