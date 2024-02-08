#include "Defs/defs.h"
#include <tinyxml2.h>
#include "Defs/opencv.h"
#include "Defs/pcl.h"
#include <chrono>
#include <iostream>

using namespace walkers;

int main() {
    try {
        setlocale(LC_NUMERIC,"C");
        tinyxml2::XMLDocument config;
        config.LoadFile("../../resources/configGlobal.xml");
        if (config.ErrorID())
            std::cout << "unable to load config file.\n";

        std::string inputFile = "/home/dominik/uczelnia/badania/hierarchies/Dataset/ObjectDB-2014-Boris-v7/mug1/data2-points_raw-mug1-3.pcd";
        std::string outputFileDepth = "/home/dominik/uczelnia/badania/hierarchies/Dataset/ObjectDB-2014-Boris-v7/mug1/data2-points_raw-mug1-3.png";
        std::string outputFileColor = "/home/dominik/uczelnia/badania/hierarchies/Dataset/ObjectDB-2014-Boris-v7/mug1/data2-points_raw-mug1-3color.png";


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PCLPointCloud2 cloud_blob;
        pcl::io::loadPCDFile(inputFile, cloud_blob);
        pcl::fromPCLPointCloud2(cloud_blob, *cloud);

//        if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (inputFile, *cloud) == -1) {//* load the file
//            std::cout << "Couldn't read file test_pcd.pcd \n";
//            return (-1);
//        }
        std::cout << "Loaded "
                  << cloud->width * cloud->height
                  << " data points from test_pcd.pcd with the following fields: "
                  << std::endl;
        std::cout << "cloud->width * cloud->height: " << cloud->width << "x" << cloud->height << "\n";
        Mat34 cameraPose(Eigen::Translation<double, 3>(cloud->sensor_origin_(0), cloud->sensor_origin_(1), cloud->sensor_origin_(2))*Quaternion(cloud->sensor_orientation_));
        cv::Mat depthImage(cv::Size(640, 480), CV_16U);
        cv::Mat colorImage(cloud->height, cloud->width, CV_8UC3, cv::Scalar(0, 0, 0));
        size_t pointNo=0;
        for (size_t rowNo = 0; rowNo < cloud->height; ++rowNo){
            for (size_t colNo = 0; colNo < cloud->width; ++colNo){
                if (!std::isnan(cloud->points[pointNo].x)){
                    Mat34 pointTmp(Eigen::Translation<double, 3>(cloud->points[pointNo].x,cloud->points[pointNo].y,cloud->points[pointNo].z)*Mat33::Identity());
                    Mat34 pointCam = cameraPose.inverse()*pointTmp;
                    depthImage.at<uint16_t>((int)rowNo, (int)colNo) = (short unsigned int)(5000*pointCam(2,3));
                    colorImage.at<cv::Vec3b>((int)rowNo, (int)colNo)[0] = cloud->points[pointNo].b;
                    colorImage.at<cv::Vec3b>((int)rowNo, (int)colNo)[1] = cloud->points[pointNo].g;
                    colorImage.at<cv::Vec3b>((int)rowNo, (int)colNo)[2] = cloud->points[pointNo].r;
//                    std::cout << cloud->points[pointNo].z << " cloud->points[pointNo].b " << (int)cloud->points[pointNo].b << " " << (int)cloud->points[pointNo].g << " " << (int)cloud->points[pointNo].r << "\n";
                }
                pointNo++;
            }
        }
//        imwrite(outputFileDepth, depthImage );
//        namedWindow( "depth", cv::WINDOW_AUTOSIZE );
//        imshow( "depth", depthImage );

//        imwrite(outputFileColor, colorImage );
//        namedWindow( "rgb", cv::WINDOW_AUTOSIZE );
//        imshow( "rgb", colorImage );

//        cv::waitKey(-1);
        std::cout << "Done\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
