#ifndef GRABBER_DEFS_H
#define GRABBER_DEFS_H

#include "Defs/defs.h"
#include "Defs/opencv.h"
#include "Defs/eigen3.h"
#include "Defs/pcl.h"

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>



// grabber name space
namespace grabber {

    /// 3D point representation
    typedef pcl::PointXYZRGBA Point3D;

    /// 3D point cloud representation
    typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloud;

    /// 3D point cloud representation
    typedef std::vector<std::vector<pcl::PointXYZRGBA>> PointCloudOrg;

    /// Sensor Frame representation
    class SensorFrame {
        public:
            /// sequence of images
            typedef std::vector<SensorFrame> Seq;

            /// 2D image
            cv::Mat image;

            /// 2D image
            cv::Mat depth;

            /// XYZRGBA point cloud
            PointCloud cloud;

            /// timestamp
            double timestamp;

            /// Default constructor
            inline SensorFrame() : timestamp(0){
            }
    };

    class CameraModel{
      public:
        CameraModel() :
            focalLength{582.64, 586.97},
            focalAxis{320.17, 260.0},
            resolution{640, 480},
            varU(1.1046), varV(0.64160),
            distVarCoefs{-8.9997e-06, 3.069e-003, 3.6512e-006, -0.0017512e-3}{

            initPHCPmodel();

            Ruvd << varU, 0, 0,
                    0, varV, 0,
                    0, 0, 0;
        }
        CameraModel(double fx, double fy, double cx, double cy, int rx, int ry, double _varU, double _varV, double k1, double k2, double k3, double k4) :
            focalLength{fx, fy},
            focalAxis{cx, cy},
            resolution{rx, ry},
            varU(_varU), varV(_varV),
            distVarCoefs{k1, k2, k3, k4}{

            initPHCPmodel();

            Ruvd << varU, 0, 0,
                    0, varV, 0,
                    0, 0, 0;
        }
        CameraModel(std::string configFilename);

        ///init PHCPmodel
        void initPHCPmodel(void);
        /// 3D point from pixel
        void getPoint(unsigned int u, unsigned int v, double depth, Eigen::Vector3d& point3D);
        /// pixel from 3D point
        walkers::Vec3 inverseModel(double x, double y, double z) const;
        /// pixel from 3D point
        walkers::Vec3 inverseModel(double x, double y, double z, size_t maxCols, size_t maxRows) const;
        /// compute covariance
        void computeCov(int u, int v, double depth, walkers::Mat33& cov);

        /// Convert disparity image to point cloud
        PointCloud depth2cloud(const cv::Mat& depthImage);

        /// Convert disparity image to point cloud
        PointCloud depth2cloud(const cv::Mat& depthImage, const cv::Mat& colorImage);

        /// Convert disparity image to point cloud
        PointCloud depth2cloud(const cv::Mat& depthImage, const cv::Mat& colorImage, double depthFactor, double minDepth, double maxDepth);

        /// Convert disparity image to point cloud
        PointCloud depth2cloud(const cv::Mat& depthImage, const cv::Mat& colorImage, double depthFactor, double minDepth, double maxDepth,
                               const std::pair<size_t, size_t>& boundingBoxX, const std::pair<size_t, size_t>& boundingBoxY);

        /// Convert disparity image to point cloud
        PointCloudOrg depth2cloudOrg(const cv::Mat& depthImage, const cv::Mat& colorImage);

        public:
            double focalLength[2];
            double focalAxis[2];
            int resolution[2];
            double varU, varV;// variance u,v
            double distVarCoefs[4];
            double depthScale;
            // Camera pose in robot's coordination frame
            walkers::Mat34 pose;
            //pin-hole camera projection model
            walkers::Mat33 PHCPModel;
            //covariance matrix for [u,v,disp]
            walkers::Mat33 Ruvd;

    };

}

#endif // GRABBER_DEFS_H
