#include "Defs/defs.h"
#include "Mapping/gaussmap.h"
#include <tinyxml2.h>
#include "Defs/opencv.h"
#include "Visualizer/Qvisualizer.h"
#include "Grabber/grabber.h"
#include "ImageProcessing/procRGBD.h"
#include "Utilities/weightedGraph.h"
#include <GL/glut.h>
#include <qapplication.h>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <random>

using namespace std;

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

void getEllipsoids(const std::vector<std::unique_ptr<mapping::Map>>& localMaps, mapping::Ellipsoid::Seq& ellipsoids){
    ellipsoids.clear();
    for (const auto& localMap : localMaps){
        mapping::Ellipsoid::Seq localEllipsoids;
        ((Gaussmap*)(localMap.get()))->getEllipsoidsGlobal(localEllipsoids);
        ellipsoids.insert(ellipsoids.end(), localEllipsoids.begin(), localEllipsoids.end());
    }
}

void getVoxels(const std::vector<std::unique_ptr<mapping::Map>>& localMaps, mapping::VoxelVisu::Seq& voxels){
    voxels.clear();
    for (const auto& localMap : localMaps){
        mapping::VoxelVisu::Seq localVoxels;
        ((Gaussmap*)(localMap.get()))->getVoxelsGlobal(localVoxels);
        voxels.insert(voxels.end(), localVoxels.begin(), localVoxels.end());
    }
}

void getBoundingEllipsoids(const std::vector<std::unique_ptr<mapping::Map>>& localMaps, mapping::Ellipsoid::Seq& ellipsoids){
    ellipsoids.clear();
    for (size_t localMapNo=0; localMapNo<localMaps.size(); localMapNo++){
        Eigen::Vector3d boundEllipMean; walkers::Mat33 boundEllipCov;
        localMaps[localMapNo]->getBoundingEllipsoidGlobal(boundEllipMean, boundEllipCov);
        mapping::Ellipsoid ellipsoid;
        ellipsoid.position.vector() = boundEllipMean;
        ellipsoid.cov = boundEllipCov;
        ellipsoid.color.r = 150; ellipsoid.color.g = 150; ellipsoid.color.b = 150; ellipsoid.color.a = 150;
        ellipsoids.push_back(ellipsoid);
    }
}

/// compute the value of the multivariate gaussian function
double mahalanobis(const walkers::Vec3& mean, const walkers::Mat33& cov, const walkers::Vec3& pos){
    return std::exp(-0.5*(pos.vector()-mean.vector()).transpose()*cov.inverse()*(pos.vector()-mean.vector()));///std::sqrt(pow(2*M_PI,3.0)*cov.determinant());
}

void getPointCloud(const std::vector<std::unique_ptr<mapping::Map>>& localMaps, mapping::PointCloud& pointCloud, size_t samplesNo){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    pointCloud.clear();
    //initialize rand
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 mt(seed);
    mapping::Ellipsoid::Seq ellipsoids;
    getEllipsoids(localMaps, ellipsoids);
    std::uniform_int_distribution<size_t> distPointNo(0,(size_t)ellipsoids.size());
    for (size_t sampleNo=0; sampleNo<(size_t)samplesNo;sampleNo++){
        size_t sample = distPointNo(mt);
        mapping::Point3D point;
        point.position = Gaussmap::sampleFromEllipsoid(ellipsoids[sample]);

//        if (std::isnan(point.position.x())){
//            std::cout << point.position.vector() << "\n";
//            std::cout << sample << "/" << ellipsoids.size() << "\n";
//            std::cout << "sameplNo " << sampleNo << "\n";
//            std::cout << ellipsoids[sample].position.vector() << "\n";
//            std::cout << ellipsoids[sample].cov << "\n";
//            getchar();
//        }
        point.color = ellipsoids[sample].color;
        //point.color.a = (uint8_t)mahalanobis(point.position, ellipsoids[sample].cov, ellipsoids[sample].position)*255;
        if (!std::isnan(point.position.x()))
            pointCloud.push_back(point);
    }
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Creating point cloud duration time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;
    std::cout << "point cloud size " << pointCloud.size() << "\n";
}

bool isOccupied(const std::vector<std::unique_ptr<mapping::Map>>& localMaps, const walkers::Vec3& pos, mapping::RGBA& voxelColor){
    mapping::Ellipsoid::Seq ellipsoids;
    getBoundingEllipsoids(localMaps, ellipsoids);
    std::vector<int> submapsNumbers;
    for (size_t ellipsoidNo=0; ellipsoidNo<ellipsoids.size(); ellipsoidNo++){
//        double val = mahalanobis(ellipsoid.position, ellipsoid.cov, pos);
//        if (val>0.6){
            submapsNumbers.push_back((int)ellipsoidNo);
//        }
    }
    for (const auto mapNo : submapsNumbers){
        walkers::Mat34 mapPose;
        localMaps[mapNo]->getMapPose(mapPose);
        walkers::Vec3 posInMap;
        walkers::Mat34 mapPoseInv = mapPose.inverse();
        posInMap.vector() = mapPoseInv.matrix().block<3,3>(0,0)*pos.vector()+mapPoseInv.matrix().block<3,1>(0,3);
        if (localMaps[mapNo]->isOccupied(posInMap)){
            localMaps[mapNo]->getColor(posInMap, voxelColor);
            return true;
        }
    }
    return false;
}

void createOccupancyMap(const std::vector<std::unique_ptr<mapping::Map>>& localMaps, std::unique_ptr<mapping::Map>& globalMap){
    double resolution = globalMap->getResolution();
    int mapSize = 50;//globalMap->getMapSize();
    for (int x=-mapSize/2;x<mapSize/2;x++){
        for (int y=-mapSize/2;y<mapSize/2;y++){
            for (int z=-mapSize/2;z<mapSize/2;z++){
                walkers::Vec3 pos(x*resolution,y*resolution,z*resolution);
                mapping::RGBA voxelColor;
                if (isOccupied(localMaps, pos, voxelColor)){
                    globalMap->updateVoxelOccupancy(pos, voxelColor);
                }
            }
        }
    }
}

void createGlobalMap(const std::vector<std::unique_ptr<mapping::Map>>& localMaps, std::unique_ptr<mapping::Map>& globalMap){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    mapping::PointCloud pointCloud;
    size_t samplesNo = 100000*localMaps.size();
    getPointCloud(localMaps, pointCloud, samplesNo);
    size_t pointNo=0;
    for (const auto& p : pointCloud){
        if (std::isnan(p.position.x())){
            std::cout << p.position.vector() << "\n";
            std::cout << pointNo << "\n";
            getchar();
        }
        pointNo++;
    }
    globalMap->insertCloud(pointCloud, walkers::Mat34::Identity(), mapping::updateMethodType::TYPE_NDTOM, true);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << "Creating global map duration time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]" << std::endl;
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

        int mapSize = 1024;
        double resolution = 0.1;
        double raytraceFactor=0.08;
        int pointThreshold = 10;
        std::unique_ptr<mapping::Map> map3D = mapping::createMapGauss(mapSize, resolution, raytraceFactor, pointThreshold);

        ((Gaussmap*)map3D.get())->attachVisualizer(&visu);
        std::cout << "Loading 3D map...\n";
        map3D->loadMap("full3Dmap.dat");
//        covisibilityGraph.load("covisibilityGraph.dat");
//        size_t localMapsNo = 0;
//        std::ifstream mapfile("localMaps.dat");
//        mapfile >> localMapsNo;
//        std::cout << "localMapsNo " << localMapsNo << "\n";
//        mapping::Ellipsoid::Seq ellipsoids;
//        for (size_t localMapNo=0; localMapNo<localMapsNo; localMapNo++){
//            std::string filenameMap;
//            mapfile >> filenameMap;
//            std::cout << "filenameMap " << filenameMap << "\n";
//            localMaps.push_back(mapping::createMapGauss(mapSize, resolution, raytraceFactor, pointThreshold));
//            localMaps.back()->loadMap(filenameMap);
////            if (localMapNo==3){
////                ((Gaussmap*)(localMaps.back().get()))->getEllipsoidsGlobal(ellipsoids);
////            }
//        }
//        visu.updateEllipsoids(ellipsoids);
//        mapfile.close();
//                getEllipsoids(localMaps, ellipsoids);
//                visu.updateEllipsoids(ellipsoids);
        mapping::VoxelVisu::Seq voxels;
        getVoxels(localMaps, voxels);
//        std::cout << "vsize: " << voxels.size() << "\n";
        visu.addVoxels(voxels);

//        getBoundingEllipsoids(localMaps,ellipsoids);
//        visu.addEllipsoids(ellipsoids);

        /// create global octomap
//        createOccupancyMap(localMaps, map3D);
//        ((Gaussmap*)(map3D.get()))->getVoxelsGlobal(voxels);
//        visu.updateVoxels(voxels);

//        size_t mapNo=0;
//        for (auto& localMap : localMaps){
//            walkers::Mat34 mapPose;
//            localMap->getMapPose(mapPose);
//            walkers::Mat34 offset(walkers::Mat34::Identity());
//            offset = walkers::toRotationMat(walkers::Vec3(0,0,double(mapNo*0.03)));
//            offset(1,3) = double(mapNo*0.01);
//            mapPose = mapPose*offset;
//            localMap->updateMapPose(mapPose);
//            mapNo++;
//        }
//        createGlobalMap(localMaps, map3D);
//        map3D.get()->removeFloor(-0.06);
//        std::cout << "vsize: " << voxels.size() << "\n";
//        mapping::PointCloud pointCloud;
//        int samplesNo = 3000000;
//        getPointCloud(localMaps, pointCloud, samplesNo);
//        visu.updateCloud(pointCloud);
        std::cout << "Done.\n";
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
