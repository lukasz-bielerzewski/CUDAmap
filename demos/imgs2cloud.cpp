#include "Defs/defs.h"
#include <tinyxml2.h>
#include "Defs/opencv.h"
#include "Defs/pcl.h"
#include "Grabber/grabber.h"

#ifdef BUILD_VISUALIZER
#include "Visualizer/Qvisualizer.h"
#include <GL/glut.h>
#include <qapplication.h>
#endif

#include <chrono>
#include <iostream>

using namespace walkers;

void processSimulation(void){
    char option = 0;
    while (option!='q'){
        std::cout << "Select option (type '?' for help): ";
        std::cin >> option;
        if (option=='q') {
            std::cout << "Quit.\n";
        }
        else if (option == '?'){
            std::cout << "Available options:\n"
                      << "v - visualize classes\n"
                      << "c - visualize curvature\n"
                      << "p - visualize footholds\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char** argv) {
    try {
        setlocale(LC_NUMERIC,"C");
        tinyxml2::XMLDocument config;
        config.LoadFile("../../resources/configGlobal.xml");
        if (config.ErrorID())
            std::cout << "unable to load config file.\n";
        auto rootXML = config.FirstChildElement( "configGlobal" );
        std::string simConfig(rootXML->FirstChildElement( "environment" )->FirstChildElement("config")->GetText());
        std::string simType(rootXML->FirstChildElement( "environment" )->FirstChildElement("type")->GetText());

        std::string plannerConfig(rootXML->FirstChildElement( "Planner" )->FirstChildElement("config")->GetText());
        std::string plannerType(rootXML->FirstChildElement( "Planner" )->FirstChildElement("type")->GetText());

        std::string motionCtrlConfig(rootXML->FirstChildElement( "MotionController" )->FirstChildElement("config")->GetText());
        std::string motionCtrlType(rootXML->FirstChildElement( "MotionController" )->FirstChildElement("type")->GetText());

        std::string robotConfig(rootXML->FirstChildElement( "Robot" )->FirstChildElement("config")->GetText());
        std::string robotType(rootXML->FirstChildElement( "Robot" )->FirstChildElement("type")->GetText());

        std::string mapConfig(rootXML->FirstChildElement( "Mapping" )->FirstChildElement("config")->GetText());
        std::string mapType(rootXML->FirstChildElement( "Mapping" )->FirstChildElement("type")->GetText());

        std::string visualizerConfig(rootXML->FirstChildElement( "Visualizer" )->FirstChildElement("config")->GetText());
        std::string visualizerType(rootXML->FirstChildElement( "Visualizer" )->FirstChildElement("type")->GetText());

        std::string gamepadConfig(rootXML->FirstChildElement( "HMIDevice" )->FirstChildElement("config")->GetText());
        std::string gamepadName(rootXML->FirstChildElement( "HMIDevice" )->FirstChildElement("name")->GetText());

        std::string coldetConfig(rootXML->FirstChildElement( "CollisionDetection" )->FirstChildElement("config")->GetText());
        std::string coldetType(rootXML->FirstChildElement( "CollisionDetection" )->FirstChildElement("type")->GetText());


        QApplication application(argc,argv);

        setlocale(LC_NUMERIC,"C");
        glutInit(&argc, argv);

        QGLVisualizer visu(visualizerConfig, robotConfig, robotType, coldetType, coldetConfig);
        visu.setWindowTitle("Simulator viewer");
        visu.show();

        std::string inputDepth1 = "/home/dominik/uczelnia/badania/Datasets/living_room_traj0_frei/depth_00000.png";
        std::string inputRGB1 = "/home/dominik/uczelnia/badania/Datasets/living_room_traj0_frei/rgb_00000.png";
        std::string inputDepth2 = "/home/dominik/uczelnia/badania/Datasets/living_room_traj0_frei/depth_00000.png";
        std::string inputRGB2 = "/home/dominik/uczelnia/badania/Datasets/living_room_traj0_frei/rgb_00000.png";

//        std::string inputPatchDepth1 = "/home/dominik/Sources/walkers/build/bin/dataset/depthIn00000yaw0obj0.png";
//        std::string inputPatchRGB1 = "/home/dominik/Sources/walkers/build/bin/dataset/rgbIn00000yaw0obj0.png";
//        std::string inputPatchDepth2 = "/home/dominik/Sources/walkers/build/bin/dataset/depthIn00000yaw-179obj0.png";
//        std::string inputPatchRGB2 = "/home/dominik/Sources/walkers/build/bin/dataset/rgbIn00000yaw-179obj0.png";

        cv::Mat depth1, depth2, patchDepth1, patchDepth2;
        cv::Mat rgb1, rgb2, patchrgb1, patchrgb2;

        depth1 = cv::imread(inputDepth1, cv::IMREAD_ANYDEPTH);
        rgb1 = cv::imread(inputRGB1, cv::IMREAD_COLOR);

        depth2 = cv::imread(inputDepth2, cv::IMREAD_ANYDEPTH);
        rgb2 = cv::imread(inputRGB2, cv::IMREAD_COLOR);

//        patchDepth1 = cv::imread(inputPatchDepth1, cv::IMREAD_ANYDEPTH);
//        patchrgb1 = cv::imread(inputPatchRGB1, cv::IMREAD_COLOR);

//        patchDepth2 = cv::imread(inputPatchDepth2, cv::IMREAD_ANYDEPTH);
//        patchrgb2 = cv::imread(inputPatchRGB2, cv::IMREAD_COLOR);

        namedWindow( "depth1", cv::WINDOW_AUTOSIZE );
        cv::imshow( "depth1", depth1 );
        namedWindow( "rgb1", cv::WINDOW_AUTOSIZE );
        cv::imshow( "rgb1", rgb1 );

        namedWindow( "depth2", cv::WINDOW_AUTOSIZE );
        cv::imshow( "depth2", depth2 );
        namedWindow( "rgb2", cv::WINDOW_AUTOSIZE );
        cv::imshow( "rgb2", rgb2 );

//        namedWindow( "patch depth1", cv::WINDOW_AUTOSIZE );
//        cv::imshow( "patch depth1", patchDepth1 );
//        namedWindow( "patch rgb1", cv::WINDOW_AUTOSIZE );
//        cv::imshow( "patch rgb1", patchrgb1 );

//        namedWindow( "patch depth2", cv::WINDOW_AUTOSIZE );
//        cv::imshow( "patch depth2", patchDepth2 );
//        namedWindow( "patch rgb2", cv::WINDOW_AUTOSIZE );
//        cv::imshow( "patch rgb2", patchrgb2 );

//        cv::Mat depth1fromPatch = depth1;
//        cv::Mat depth2fromPatch = depth2;
//        cv::Mat rgb1fromPatch = rgb1;
//        cv::Mat rgb2fromPatch = rgb2;
//        depth1fromPatch.setTo(cv::Scalar(0));
//        depth2fromPatch.setTo(cv::Scalar(0));
//        int x1 = 211; int y1 = 237;
//        int x2 = 434; int y2 = 261;
//        patchDepth1.copyTo(depth1fromPatch(cv::Rect(x1-patchDepth1.cols/2,
//                                                    y1-patchDepth1.rows/2,
//                                                    patchDepth1.cols, patchDepth1.rows)));
//        patchDepth2.copyTo(depth2fromPatch(cv::Rect(x2-patchDepth2.cols/2,
//                                                    y2-patchDepth2.rows/2,
//                                                    patchDepth2.cols, patchDepth2.rows)));
//        patchrgb1.copyTo(rgb1fromPatch(cv::Rect(x1-patchrgb1.cols/2,
//                                                y1-patchrgb1.rows/2,
//                                                patchrgb1.cols, patchrgb1.rows)));
//        patchrgb2.copyTo(rgb2fromPatch(cv::Rect(x2-patchrgb2.cols,
//                                                y2-patchrgb2.rows,
//                                                patchrgb2.cols, patchrgb2.rows)));

//        namedWindow( "depth1fromPatch", cv::WINDOW_AUTOSIZE );
//        cv::imshow( "depth1fromPatch", depth1fromPatch );
//        namedWindow( "rgb1fromPatch", cv::WINDOW_AUTOSIZE );
//        cv::imshow( "rgb1fromPatch", rgb1fromPatch );

//        cv::waitKey(-1);

        Mat34 camPose1(Mat34::Identity());
//        camPose1.matrix() << -0.202684, -0.631363, 0.748532, -0.877487,
//                             -0.979244, 0.13068, -0.154931, 0.136582,
//                                    -0, -0.764398, -0.644745, 1.26265,
//                0, 0, 0, 1;
        visu.addAxis(camPose1,3.0,1.0);
        std::cout << "Pose1: \n" << camPose1.matrix() << "\n";

        Mat34 camPose2(Mat34::Identity());
//        camPose2.matrix() << 0.202684, 0.631363, -0.748532, 1.0687,
//                             0.979244, -0.13068, 0.154931, -0.26624,
//                          0, -0.764398, -0.644745, 1.26265,
//                0, 0, 0, 1;
        visu.addAxis(camPose2,3.0,1.0);
        std::cout << "Pose2: \n" << camPose2.matrix() << "\n";

        std::string kinectModelConfig = "cameraModels/KinectModel.xml";
        grabber::CameraModel kinect2Model(kinectModelConfig);

        grabber::PointCloud cloud1 = kinect2Model.depth2cloud(depth1,rgb1,0.0002,0.0,8.0);

        /// point cloud
        mapping::PointCloud cloudMap1;
        std::cout << cloud1.size() << "\n";
        for (const auto& point : cloud1){
            //transform point
            walkers::Mat34 pointCL(Mat34::Identity());
            pointCL(0,3) = point.x; pointCL(1,3) = point.y; pointCL(2,3) = point.z;
            pointCL = camPose1*pointCL;
            mapping::Point3D point3D(pointCL(0,3), pointCL(1,3), pointCL(2,3), point.r, point.g, point.b);
//            mapping::Point3D point3D(point.x, point.y, point.z, point.r, point.g, point.b);
            cloudMap1.push_back(point3D);
        }
        visu.addCloud(cloudMap1);
std::cout << "S3\n";
        grabber::PointCloud cloud2 = kinect2Model.depth2cloud(depth2,rgb2,0.0002,0.0,8.0);

        /// point cloud
        mapping::PointCloud cloudMap2;
        std::cout << cloud2.size() << "\n";
        for (const auto& point : cloud2){
            walkers::Mat34 pointCL(Mat34::Identity());
            pointCL(0,3) = point.x; pointCL(1,3) = point.y; pointCL(2,3) = point.z;
            pointCL = camPose2*pointCL;
            mapping::Point3D point3D(pointCL(0,3), pointCL(1,3), pointCL(2,3), point.r, point.g, point.b);
//            mapping::Point3D point3D(point.x,point.y,point.z,point.r,point.g,point.b);
            cloudMap2.push_back(point3D);
        }
        visu.addCloud(cloudMap2);

        // Run main loop.
        std::thread processThr(processSimulation);
        application.exec();
        processThr.join();
//        imwrite(outputFileColor, colorImage );


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
