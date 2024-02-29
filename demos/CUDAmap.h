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

void allocateMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat &depthImage, cv::Mat &colorImage,
                    grabber::CameraModel &kinectModel, double depthFactor, double minDepth, double maxDepth,
                    SimplePoint* &gpuPointCloud, SimplePoint* &gpuTransformedPointCloud);

void deallocateMemory(cv::cuda::GpuMat &gpuDepthImage, cv::cuda::GpuMat &gpuColorImage, SimplePoint* &gpuPointCloud, SimplePoint* &gpuTransformedPointCloud);

void CUDAmap(cv::cuda::GpuMat &gpuDepthImage, cv::Mat& depthImage, cv::cuda::GpuMat &gpuColorImage, cv::Mat& colorImage,
             SimplePoint* &gpuPointCloud, walkers::Mat34 &camPose, SimplePoint* &gpuTransformedPointCloud,
             TimeMeasurements &timepointtrans, TimeMeasurements &timedepth2cloud);

#endif // CUDAMAP_H
