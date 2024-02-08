#include "Defs/defs.h"
#include <tinyxml2.h>
#include "Visualizer/Qvisualizer.h"
#include "Mapping/elevationMap.h"
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

using namespace std;

void processSimulation(QGLVisualizer* _visu){
    char option = 0;
    while (option!='q'){
        std::cout << "Select option (type '?' for help): ";
        std::cin >> option;
        if (option=='q') {
            _visu->closeWindow();
            std::cout << "Quit.\n";
        }
        else if (option == '?'){
            std::cout << "Available options:\n"
                      << "v - visualize classes\n"
                      << "c - visualize curvature\n"
                      << "p - visualize footholds\n"
                      << "j - snap RGB, depth images and point cloud\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char** argv)
{
    try {
        setlocale(LC_NUMERIC,"C");
        tinyxml2::XMLDocument config;
        config.LoadFile("../../resources/configGlobal.xml");
        if (config.ErrorID()){
            std::cout << "unable to load config file.\n";
            return 0;
        }
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

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
//        std::string inputFile = "/home/dominik/uczelnia/badania/artykul_objects_recon/images/kinect_v2_scenes/20210827-114754/20210827-114754_all_pcl.pcd";
//        std::string inputFile = "/home/dominik/uczelnia/badania/artykul_objects_recon/images/kinect_v2_scenes2/20210830-122111/all_pcl.pcd";
        std::string inputFile = "/home/dominik/uczelnia/badania/artykul_objects_recon/results/kinect2_robot_new/all_pcl_model200.pcd";
//        std::string inputFile = "/home/dominik/uczelnia/badania/artykul_objects_recon/images/kinect_v2_scenes/20210827-123334/all_pcl.pcd";
//        std::string inputFile = "/home/dominik/uczelnia/badania/artykul_objects_recon/results/z_siecia_Rafala/20210907-224124/all_pcl_new_ws.pcd";

        pcl::PCLPointCloud2 cloud_blob;
        pcl::io::loadPCDFile (inputFile, cloud_blob);
        pcl::fromPCLPointCloud2(cloud_blob, *cloud);

        /// point cloud
        mapping::PointCloud cloudVisu;
        for (size_t pointNo=0;pointNo<cloud->size();pointNo++){
            mapping::Point3D point(cloud->points[pointNo].x,cloud->points[pointNo].y,cloud->points[pointNo].z,
                                   cloud->points[pointNo].r,cloud->points[pointNo].g,cloud->points[pointNo].b);
            cloudVisu.push_back(point);
        }
        visu.addCloud(cloudVisu);

        // Run main loop.
        std::thread processThr(processSimulation,&visu);
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
