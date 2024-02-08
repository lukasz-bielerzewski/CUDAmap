#include "Defs/defs.h"
#include "Mapping/gaussmap.h"
#include <tinyxml2.h>
#include "Defs/opencv.h"
#include "Visualizer/Qvisualizer.h"
#include "Grabber/grabber.h"
#include "ImageProcessing/procRGBD.h"
#include "Utilities/weightedGraph.h"
#include <GL/glut.h>
#include "Defs/qt.h"
#include <iostream>
#include <thread>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <experimental/filesystem>
#include <regex>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
namespace fs = std::experimental::filesystem;

void processSimulation(void){//, Planner* planner){
    char option = 0;

    while (option!='q'){
        std::cout << "Select option (type '?' for help): ";
        std::cin >> option;
        if (option=='q') {
            std::cout << "Quit.\n";
        }
        else if (option == '?'){
            std::cout << "Available options:\n"
                      << "p - motion planning\n"
                      << "s - set servo position\n"
                      << "r - get servo position\n"
                      << "m - move platform\n"
                      << "h - move single leg\n"
                      << "o - tripod step\n"
                      << "G - Gamepad mode\n"
                      << "e - playground\n"
                      << "c - demo 2 - walking\n"
                      << "d - move foot on square\n"
                      << "f - demo 3 - greeting\n"
                      << "i - demo 1 - rise\n";
        }
    }
}

std::string toStringWithPrecision(double value, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << value;
    return out.str();
}

std::unordered_map<int,KeyFrame> keyframes;
walkers::WeightedGraph<double> covisibilityGraph;
std::vector<std::unique_ptr<mapping::Map>> localMaps;

int main(int argc, char** argv)
{
    try {
        //initialize rand
        std::srand((unsigned int)time(NULL));

        setlocale(LC_NUMERIC,"C");
        tinyxml2::XMLDocument config;
        config.LoadFile("../../resources/configGlobal.xml");
        if (config.ErrorID())
            std::cout << "unable to load config file.\n";
        auto rootXML = config.FirstChildElement( "configGlobal" );
        std::string robotConfig(rootXML->FirstChildElement( "Robot" )->FirstChildElement("config")->GetText());
        std::string robotType(rootXML->FirstChildElement( "Robot" )->FirstChildElement("type")->GetText());

        std::string visualizerConfig(rootXML->FirstChildElement( "Visualizer" )->FirstChildElement("config")->GetText());
        std::string visualizerType(rootXML->FirstChildElement( "Visualizer" )->FirstChildElement("type")->GetText());

        std::string coldetConfig(rootXML->FirstChildElement( "CollisionDetection" )->FirstChildElement("config")->GetText());
        std::string coldetType(rootXML->FirstChildElement( "CollisionDetection" )->FirstChildElement("type")->GetText());

        QApplication application(argc,argv);

        setlocale(LC_NUMERIC,"C");
        glutInit(&argc, argv);

        QGLVisualizer visu(visualizerConfig, robotConfig, robotType, coldetType, coldetConfig);
        visu.setWindowTitle("Simulator viewer");
        visu.show();

        int mapSize = 128;
        double resolution = 0.002;
        double raytraceFactor = 0.5*resolution;
        int pointThreshold = 10;
        std::unique_ptr<mapping::Map> map3D = mapping::createMapGauss(mapSize, resolution, raytraceFactor, pointThreshold);

        double maxObjectSize=0.3;
        map3D->setMapOccupied(walkers::Vec3(0.0,0.0,0.0), maxObjectSize);
        ((Gaussmap*)map3D.get())->attachVisualizer(&visu);
        map3D->setLimitUpdates(walkers::Vec3(0.0,0.0,0.0), maxObjectSize);

        std::cout << "Load data from experiments\n";

        grabber::CameraModel asusModel("../resources/cameraModels/cameraBlender.xml");

        // add borders
        bool addBorders = true;
        int borderSize=100;
        if (addBorders){
            asusModel.focalAxis[0]+=borderSize;
            asusModel.focalAxis[1]+=borderSize;
            asusModel.initPHCPmodel();
        }

//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/samplesReal/sample1/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/samplesReal/sample2/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/newSamples/gen_sample6/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/butelka_robot/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/newSamples/train_set_known_angles/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/newSamples/val_set_known_angles/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/newSamples/val_set_unknown_angles_denser/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/can_pipeline/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/bottle_pipeline/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/images/chair_pipeline/";

//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/results/rotate_net/mug/0/0.0_120.0.png/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/results/rotate_net/mug/1/0.0_0.0.png/";
        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/results/rotate_net/guitar/0/0.0_0.0.png/";
//        std::string path2sequence = "/home/dominik/uczelnia/badania/artykul_ICRA2019generative/results/rotate_net/guitar/0/0.0_120.0.png/";
//        std::vector<std::string> rgbImages = {"rgb_0_0.png", "rgb_0_12.png", "rgb_30_72.png", "rgb_30_24.png"};
//        std::vector<std::string> depthImages = {"depth_0_0.png", "depth_0_12.png", "depth_30_72.png", "depth_30_24.png"};
//        std::vector<std::pair<double,double>> rotAngles = {std::make_pair(0.0,0.0), std::make_pair(0.0,12.0), std::make_pair(30.0,72.0), std::make_pair(30.0,24.0)};

        /// extract roll and pitch angles from filenames
//        std::vector<std::string> rgbImages;
//        std::vector<std::string> depthImages;
//        std::vector<std::pair<double,double>> rotAngles;
//        for (const auto & entry : fs::directory_iterator(path2sequence)){
//            std::string input(entry.path());
//            std::size_t posDepth = input.find("depth");
//            if (posDepth>0&&posDepth<220){
//                double roll = atof(input.c_str() + posDepth+6);
//                std::size_t posPitch = input.find_last_of("_");
//                double pitch = atof(input.c_str() + posPitch+1);
//                std::string depthPath(input);
//                depthPath.erase(0,posDepth);
//                depthImages.push_back(depthPath);
//                std::cout << input << "\n";
//                std::string pathRGB = input.replace(posDepth, 5, "rgb");
//                pathRGB.erase(0,posDepth);
//                rgbImages.push_back(pathRGB);
//                std::cout << pathRGB << "\n";
//                rotAngles.push_back(std::make_pair(roll,pitch));
//            }
//        }

        //all images
        std::vector<std::string> rgbImages;
        std::vector<std::string> depthImages;
        std::vector<std::pair<double,double>> rotAngles;
        for (int roll = 0; roll< 4;roll++){
            for (int pitch = 0; pitch< 25; pitch++){
//                rgbImages.push_back("rgb_"+std::to_string(roll*10)+"_"+std::to_string(pitch*12)+".png");
//                depthImages.push_back("depth_"+std::to_string(roll*10)+"_"+std::to_string(pitch*12)+".png");
                rgbImages.push_back("rgb_"+toStringWithPrecision(roll*9,1)+"_"+toStringWithPrecision(pitch*12,1)+".png");
                depthImages.push_back("depth_"+toStringWithPrecision(roll*9,1)+"_"+toStringWithPrecision(pitch*12,1)+".png");
                rotAngles.push_back(std::make_pair(9.0*roll,12.0*pitch));
            }
        }

        namedWindow( "rgb", cv::WINDOW_AUTOSIZE );// Create a window for display.
        namedWindow( "depth", cv::WINDOW_AUTOSIZE );// Create a window for display.

        for (size_t frameNo=0;frameNo<rgbImages.size();frameNo++){
            std::ostringstream ossRGB;  ossRGB.str(std::string());
            ossRGB << path2sequence << rgbImages[frameNo];
            std::ostringstream ossDepth;  ossDepth.str(std::string());
            ossDepth << path2sequence << depthImages[frameNo];
            std::cout << "ossRGB.str() " << ossRGB.str() << "\n";
            std::cout << "ossDepth.str() " << ossDepth.str() << "\n";
            cv::Mat rgbImage = cv::imread(ossRGB.str(), cv::IMREAD_COLOR);
            cv::Mat depthImage = cv::imread(ossDepth.str(), cv::IMREAD_ANYDEPTH);
            cv::resize(rgbImage,rgbImage,cv::Size(512,512));
            cv::resize(depthImage,depthImage,cv::Size(512,512));

            // add borders
            if (addBorders){
                copyMakeBorder( depthImage, depthImage, borderSize, borderSize, borderSize, borderSize, cv::BORDER_CONSTANT, cv::Scalar(255));
                copyMakeBorder( rgbImage, rgbImage, borderSize, borderSize, borderSize, borderSize, cv::BORDER_CONSTANT, cv::Scalar(255,255,255));
            }

            imshow( "rgb", rgbImage );
            imshow( "depth", depthImage );                    // Show our image inside it.
            if (frameNo==0){
                cv::waitKey(-1);
                std::cout << "press any key\n";
//                getchar();
            }
            else
                cv::waitKey(30);

            grabber::Grabber::filterDepthImage(depthImage, depthImage, "median", 5);
            grabber::PointCloud cloud = asusModel.depth2cloud(depthImage, rgbImage, 0.005, 1.274, 1.3);
//            grabber::PointCloud cloud = asusModel.depth2cloud(depthImage, rgbImage, 0.005, 1.15, 1.3);
//            grabber::PointCloud cloudObject = asusModel.depth2cloud(depthImage, rgbImage, 0.005, 0.05, 1.05);
//            grabber::PointCloud cloudPositive = asusModel.depth2cloud(depthImage,rgbImage,0.005, 0.05, 1.05);
            std::cout << frameNo << "\n";
            walkers::Mat34 camPose;
            camPose = walkers::toRotationMat(walkers::Vec3(-rotAngles[frameNo].first*M_PI/180.0, rotAngles[frameNo].second*M_PI/180.0, 0.0));
            camPose(2,3) = 0.5;
            std::cout << "camPose\n" << camPose.matrix() << "\n";

            mapping::PointCloud pointCloud;
            //convert pcl cloud to mapping::PointCloud
            pointCloud.resize(cloud.size());
            size_t pointNo=0;
            for (auto& point : cloud){
                pointCloud[pointNo].position = walkers::Vec3(point.x, point.y, point.z);
                pointCloud[pointNo].color.r = point.r; pointCloud[pointNo].color.g = point.g; pointCloud[pointNo].color.b = point.b; pointCloud[pointNo].color.a = point.a;
                pointNo++;
            }

//            mapping::PointCloud pointCloudObject;
//            //convert pcl cloud to mapping::PointCloud
//            pointCloudObject.resize(cloudObject.size());
//            pointNo=0;
//            for (auto& point : cloudObject){
//                pointCloudObject[pointNo].position = walkers::Vec3(point.x, point.y, point.z);
////                std::cout << point.x << ", " << point.y << " , " << point.z << " b\n";
////                getchar();
//                pointCloudObject[pointNo].color.r = point.r; pointCloudObject[pointNo].color.g = point.g; pointCloudObject[pointNo].color.b = point.b; pointCloudObject[pointNo].color.a = point.a;
//                pointNo++;
//            }

//            mapping::PointCloud pointCloudPositive;
//            //convert pcl cloud to mapping::PointCloud
//            pointCloudPositive.resize(cloudPositive.size());
//            pointNo=0;
//            for (const auto& point : cloudPositive){
//                pointCloudPositive[pointNo].position = walkers::Vec3(point.x, point.y, point.z);
//                pointCloudPositive[pointNo].color.r = point.r; pointCloudPositive[pointNo].color.g = point.g; pointCloudPositive[pointNo].color.b = point.b; pointCloudPositive[pointNo].color.a = point.a;
//                pointNo++;
//            }

            std::cout << "cloud size " << pointCloud.size() << "\n";
            visu.updateCloud(pointCloud,0);
//            visu.updateCloud(pointCloudObject);

//            grabber::PointCloudOrg cloudOrg = asusModel.depth2cloudOrg(depthImage, rgbImage);
//            mapping::Ellipsoid::Seq ellipsoids = ImageProcessing::computeEllipsoids(cloudOrg, camPose, 15);
//            std::cout << "ellipsoids no " << ellipsoids.size() << "\n";

//            size_t samplesNo=150000;
//            ImageProcessing::filterWithEllipsoids(pointCloud, cloudOrg, ellipsoids, asusModel, rgbImage, samplesNo);
//            std::cout << "point cloud size " << pointCloud.size() << "\n";
//            //visu.updateCloud(pointCloud);

//            //transformEllipsoids(ellipsoids,camPose);
//            //visu.updateEllipsoids(ellipsoids);

//            pointCloud.erase(std::remove_if(pointCloud.begin(), pointCloud.end(), [](const mapping::Point3D& x)
//            {return (x.position.x()<0.3&&x.position.y()<0.3&&x.position.z()<0.3); }), pointCloud.end());

            //map3D->insertCloud(pointCloud,camPose.inverse(),mapping::updateMethodType::TYPE_NDTOM);
            map3D->raytraceMap(pointCloud,camPose.inverse());
            //map3D->insertCloud(pointCloudObject,camPose.inverse(),mapping::updateMethodType::TYPE_NDTOM, false);
//            ((Gaussmap*)(map3D.get()))->insertCloudPositive(pointCloudPositive,camPose.inverse());
//            getchar();
//            pcl::io::savePCDFile( "cloud.pcd", cloud);

        }
        mapping::VoxelVisu::Seq voxels;
//        ((Gaussmap*)(map3D.get()))->getVoxelsGlobal(voxels);
        std::cout << "get voxels\n";
        //getchar();
        ((Gaussmap*)(map3D.get()))->getVoxelsUnocupiedGlobal(voxels, walkers::Vec3(0,0,0), maxObjectSize);
        std::cout << "voxels size " << voxels.size() << "\n";
        std::cout << "get voxels done\n";
        //getchar();
        visu.addVoxels(voxels);
        std::cout << "visualized\n";
        //getchar();

        ///get point cloud
//        mapping::PointCloud cloudObject;
//        ((Gaussmap*)(map3D.get()))->getPointCloudOccupied(cloudObject, walkers::Vec3(0,0,0), maxObjectSize);
//        visu.updateCloud(cloudObject);

        map3D->saveMap("full3Dmap.dat");
        // Run main loop.
        std::thread processThr(processSimulation);
        application.exec();
        processThr.join();
        std::cout << "Finished\n";
        return 1;
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
