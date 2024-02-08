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
#include <random>
#include <pcl/io/ply_io.h>

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


void readTrajectoryData(const std::string& filePath, std::vector<std::vector<double>>& trajectoryData)
{
    // Open the file
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cout << "Failed to open trajectory file: " << filePath.c_str();
        return;
    }

    // Read and parse the data line by line
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::vector<double> rowData;
        double value;

        // Skip the first column and read the rest
        iss >> value;
        while (iss >> value)
        {
            rowData.push_back(value);
        }

        if (!rowData.empty())
        {
            trajectoryData.push_back(rowData);
        }
    }

    // Close the file
    file.close();
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

        ///kinematic model of the robot
        std::vector<double> robotConfHexa = {0.7854,0.41888,-114*M_PI/180, 0.0,0.41888,-1.9897, -0.7854,0.41888,-1.9897, 0.7854,0.41888,-1.9897, 0.0,0.41888,-1.9897, -0.7854,0.41888,-1.9897};
        //std::vector<double> robotConfQuad = {-7*M_PI/180,24*M_PI/180,-74*M_PI/180, 7*M_PI/180,24*M_PI/180,-74*M_PI/180, 7*M_PI/180,24*M_PI/180,-74*M_PI/180, -7*M_PI/180,24*M_PI/180,-74*M_PI/180};
        std::vector<double> robotConfQuad = {7*M_PI/180,33*M_PI/180,-60*M_PI/180, -7*M_PI/180,33*M_PI/180,-60*M_PI/180, 7*M_PI/180,33*M_PI/180,-60*M_PI/180, -7*M_PI/180,33*M_PI/180,-60*M_PI/180};
        std::vector<double> robotConf;
        if (robotType=="MessorII" || robotType=="PhantomX"){
            robotConf = robotConfHexa;
        } else if (robotType=="Galgo" || robotType=="Anymal" || robotType=="Anymal_C"){
            robotConf = robotConfQuad;
        }

        walkers::Mat34 robotPose(walkers::Mat34::Identity());
        robotPose(0,3)=-6.0;         robotPose(1,3)=-6.0;         robotPose(2,3)=3.7;
        robotPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(walkers::Vec3(0.0,0.0,0.0));
        visu.updateKinematicModel(robotPose, robotConf);

        /// elevation map
        ElevationMap map(mapConfig);
        visu.loadMap(mapConfig);

//        int row, col;
//        map.toRaster(0.6,-0.5,row,col);
//        for (int coordx=col-20;coordx<col+20;coordx++){
//            for (int coordy=row-20;coordy<row+20;coordy++){
//                std::cout << map.get(coordx,coordy) << " ";
//            }
//            std::cout << "\n";
//        }

        /// point cloud
        mapping::PointCloud cloud;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudInput (new pcl::PointCloud<pcl::PointXYZRGBA>);

//        std::string inputFile =
//                "/home/dominik/Sources/walkers/build/bin/shapeNetGraspable_30instances_DexNet_Test/Mug/cloudGlobal00000.pcd";
////        pcl::PLYReader reader;
////        reader.read(inputFile,*cloudInput);
//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source(new pcl::PointCloud<pcl::PointXYZRGBA>());
//        pcl::PCLPointCloud2 cloud_blob;
//        pcl::io::loadPCDFile(inputFile, cloud_blob);
//        pcl::fromPCLPointCloud2(cloud_blob, *source);
//        std::cout << "Loaded \n";

//        for (size_t pointNo=0;pointNo<100;pointNo++){
//            mapping::Point3D point(0.0+double(pointNo)*0.002,0.0+double(pointNo)*0.001,0+double(pointNo)*0.01,0,255,0);
//        for (const auto& p : *source){
//            mapping::Point3D point(p.x,p.y,p.z,p.r,p.g,p.b);
//            cloud.push_back(point);
//        }
//        visu.addCloud(cloud);

        /// line 3D
        walkers::Line3D line;
        line.A=0.1;
        line.B=0;
        line.C=0.2;
        visu.addLine(line);

        line.A=-0.1;
        line.B=0;
        line.C=0.2;
        visu.addLine(line);

        /// arrow 3D
        walkers::Arrow3D arrow(walkers::Vec3(3.0,2.0,0.0),walkers::Vec3(3.1,2.1,1.0));
//        visu.addArrow(arrow,3.0,mapping::RGBA(20,20,200,255));

        /// ellipsoids
        mapping::Ellipsoid::Seq ellipsoids;
        mapping::Ellipsoid ellipsoid;
        ellipsoid.cov = walkers::Mat33::Identity()*0.1;
        ellipsoid.cov(0,0)=0.02;
        ellipsoid.position.x()=-0.2; ellipsoid.position.y()=-0.2; ellipsoid.position.z()=0.7;
        ellipsoid.color = mapping::RGBA(120,220,120,255);
        ellipsoids.push_back(ellipsoid);
        visu.addEllipsoids(ellipsoids);

        /// path
        std::vector<planner::PoseSE3> path;
        path.push_back(planner::PoseSE3(Eigen::Vector3d(0.5,0.6,0.0), walkers::Quaternion(1.0, 0.0, 0.0, 0.0)));
        path.push_back(planner::PoseSE3(Eigen::Vector3d(0.6,0.5,0.0), walkers::Quaternion(1.0, 0.0, 0.0, 0.0)));
        path.push_back(planner::PoseSE3(Eigen::Vector3d(0.7,0.6,0.0), walkers::Quaternion(1.0, 0.0, 0.0, 0.0)));
        path.push_back(planner::PoseSE3(Eigen::Vector3d(0.7,0.7,0.0), walkers::Quaternion(1.0, 0.0, 0.0, 0.0)));
        visu.addPath(path, 3.0, 0.005, mapping::RGBA(120,120,200,255),mapping::RGBA(200,120,120,255));

        /// robot path
        std::vector<planner::RobotState3D> robotPath;
        for (size_t pointNo=0;pointNo<30;pointNo++){
            planner::RobotState3D robotState;
            planner::PoseSE3 _robotPose;
            _robotPose.setPosition(Eigen::Vector3d(1.0+double(pointNo)*0.015, 0.0, 0.7));
            robotState.robotPose = _robotPose;
            planner::Foot foot;
            for (size_t footNo=0;footNo<2;footNo++){
                foot.pose(0,3)=1.0+double(pointNo)*0.015; foot.pose(1,3)=0.1; foot.pose(2,3)=0.4;
                if (pointNo%2)
                    foot.pose(2,3)+=0.1;
                if (footNo==1)
                    foot.pose(1,3)=-foot.pose(1,3);
                robotState.feet.push_back(foot);
            }
            robotPath.push_back(robotState);
        }
        visu.addRobotPath(robotPath,3.0,0.005,mapping::RGBA(120,120,200,255),mapping::RGBA(200,120,200,255),2.0,0.002,
                          mapping::RGBA(120,220,100,255),mapping::RGBA(100,220,120,255));

        /// Tree
        planner::RRTree tree;
        planner::RRTNode node;
        node.id=0; node.parent=-1; node.robotPose.body = walkers::Mat34::Identity(); node.robotPose.body(0,3)=0.1;
        tree.addNode(node);
        node.id=1; node.parent=0; node.robotPose.body = walkers::Mat34::Identity(); node.robotPose.body(0,3)=0.2;
        tree.addNode(node);
        visu.addTree(tree,3.0,0.005,mapping::RGBA(120,120,200,255),mapping::RGBA(200,120,120,255));

        tree.clear();
        node.id=0; node.parent=-1; node.robotPose.body = walkers::Mat34::Identity(); node.robotPose.body(0,3)=-0.1;
        tree.addNode(node);
        node.id=1; node.parent=0; node.robotPose.body = walkers::Mat34::Identity(); node.robotPose.body(0,3)=-0.2;
        tree.addNode(node);
        visu.addTree(tree,3.0,0.005,mapping::RGBA(120,200,120,255),mapping::RGBA(120,200,120,255));

        /// Voxels
        mapping::VoxelVisu::Seq voxels;
        mapping::VoxelVisu voxel;
        voxel.color = mapping::RGBA(50,255,50,255);
        voxel.pose = walkers::Mat34::Identity();
        voxel.pose(0,3) = -2.0; voxel.pose(1,3) = 1.0; voxel.pose(2,3) = 2.0;
        voxel.width = 0.1;
        voxels.push_back(voxel);
        voxel.pose(0,3) = -2.0; voxel.pose(1,3) = 1.2; voxel.pose(2,3) = 2.2;
        voxels.push_back(voxel);
        voxel.pose(0,3) = -2.0; voxel.pose(1,3) = 1.4; voxel.pose(2,3) = 2.3;
        voxels.push_back(voxel);
        visu.addVoxels(voxels);

        /// Planes
        walkers::Plane3D plane;
        plane.A = 1.0; plane.B = 2.0; plane.C = 3.0; plane.D = 4.0;
        plane.x0 = walkers::Vec3(-4.0, -2.0, 1.0);
        std::vector<walkers::Plane3D> planes;
        planes.push_back(plane);
        visu.addPlane(planes, mapping::RGBA(50,255,150,255));

        /// map NDTOM
        int mapSize = 1024;
        double resolution = 0.1;
        double raytraceFactor = 0.05;
        int pointThreshold = 10;
        std::unique_ptr<mapping::Map> map3D = mapping::createMapGauss(mapSize, resolution, raytraceFactor, pointThreshold);
        ((Gaussmap*)map3D.get())->attachVisualizer(&visu);
        cloud.clear();
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0,0.002);

        for (size_t pointNo=0;pointNo<1000;pointNo++){
            mapping::Point3D point(0.0+double(pointNo)*0.0002+distribution(generator),
                                   -2.0+double(pointNo)*0.0001+distribution(generator),
                                   0.3+double(pointNo)*0.001+distribution(generator),200,255,120);
            cloud.push_back(point);
            mapping::Point3D point1(0.0+double(pointNo)*0.0004 + distribution(generator),
                                    -2.0+double(pointNo)*0.0003+distribution(generator),
                                    0.3+double(pointNo)*0.002+distribution(generator),200,255,120);
            cloud.push_back(point1);
            mapping::Point3D point2(0.0+double(pointNo)*0.0003+distribution(generator),
                                    -2.0+double(pointNo)*0.0006+distribution(generator),
                                    0.3+double(pointNo)*0.003+distribution(generator),200,255,120);
            cloud.push_back(point2);
        }
        for (auto& point : cloud){
            point.color.r=255;
        }
//        map3D->insertCloud(cloud,walkers::Mat34::Identity(),mapping::updateMethodType::TYPE_NDTOM,false);
        visu.addCloud(cloud);

        /// Axis
        walkers::Mat34 axisPose(walkers::Mat34::Identity());
        axisPose = walkers::toRotationMat(walkers::Vec3(0.1,-0.2,0.3));
        axisPose(0,3) = -0.5; axisPose(1,3) = 0.6; axisPose(2,3) = 0.7;
        double axisScale = 0.1;
        double axisWidth = 3.0;
        visu.addAxis(axisPose, axisWidth, axisScale);

        simulator::RenderObject renderObj;
        std::vector<simulator::RenderObject> renderObjects;
        renderObj.meshFilename = "../../resources/models/hammer.dae";
        renderObj.mat = walkers::Mat34::Identity();
        renderObj.mat = walkers::toRotationMat(walkers::Vec3(0.0,0.0,0));
        renderObj.mat(0,3) = 0.4; renderObj.mat(1,3) = 0.3; renderObj.mat(2,3) = 0.3;
        renderObj.scaleX = 1.0; renderObj.scaleY = 1.0; renderObj.scaleZ = 1.0;
        renderObj.type = simulator::RenderObjectType::MODEL3D;
        renderObjects.clear();
        renderObjects.push_back(renderObj);
        visu.addRenderObjects(renderObjects);

        /// B-Spline 3D
        walkers::BSpline3D bspline3D(3);
        std::vector<walkers::Vec3> controlPoints;
        controlPoints.push_back(walkers::Vec3(0.0,1.0,1.0));
        controlPoints.push_back(walkers::Vec3(0.0,1.0,1.0));
        controlPoints.push_back(walkers::Vec3(0.0,1.5,2.0));
        controlPoints.push_back(walkers::Vec3(0.5,2.0,1.0));
        controlPoints.push_back(walkers::Vec3(0.0,2.5,2.0));
        controlPoints.push_back(walkers::Vec3(0.0,3.0,1.0));
        controlPoints.push_back(walkers::Vec3(0.0,3.0,1.0));
        bspline3D.setControlPoints(controlPoints);
        visu.updateBSpline3D(bspline3D,0, 50, 3, mapping::RGBA(255,0,0,255), mapping::RGBA(0,255,0,255), 7, true, true);

        /// B-Spline SE3
        walkers::BSplineSE3 bsplineSE3(3);
        std::vector<walkers::Mat34> controlPoses;
        walkers::Vec6 controlPose;
        controlPose << 1.0,1.0,1.0,0.0,0.0,0.0;
        controlPoses.push_back(walkers::fromTranslRPY(controlPose));
        controlPoses.push_back(walkers::fromTranslRPY(controlPose));
        controlPose << 2.0,2.0,1.0,3.0,0.0,0.0;
        controlPoses.push_back(walkers::fromTranslRPY(controlPose));
        controlPose << 2.5,1.5,1.0,-3.0,0.0,0.0;//crucial -- orientation change should be small
        controlPoses.push_back(walkers::fromTranslRPY(controlPose));
        controlPose << 3.0,1.0,1.0,0.0,0.0,0.0;
        controlPoses.push_back(walkers::fromTranslRPY(controlPose));
        controlPoses.push_back(walkers::fromTranslRPY(controlPose));
        bsplineSE3.setControlPoints(controlPoses);
        visu.updateBSplineSE3(bsplineSE3,0, 150, 3, mapping::RGBA(255,0,0,255), mapping::RGBA(0,255,0,255),
                              7, true, true, true, 0.05);

        std::vector<std::vector<double>> trajectoryData;
        readTrajectoryData("/home/dominik/uczelnia/badania/Datasets/living_room_traj0_frei/livingRoom0.gt.freiburg", trajectoryData);

        /// path
        std::vector<planner::PoseSE3> pathICL;
        for (const auto& point : trajectoryData){
            pathICL.push_back(planner::PoseSE3(Eigen::Vector3d(point[0],point[1],point[2]), walkers::Quaternion(point[6], point[3], point[4], point[5])));
        }
        visu.addPath(pathICL, 3.0, 0.005, mapping::RGBA(120,0,0,255),mapping::RGBA(0,255,0,255));

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
