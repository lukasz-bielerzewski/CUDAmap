
#include "Mapping/gaussmap.h"
#include "Utilities/samplerEllipsoid.h"
#include "Utilities/TimeRange.h"
#include <random>
#include <tinyxml2.h>

#include <thread>
#include <vector>
#include <mutex>

TimeMeasurements timeupdatemap;
TimeMeasurements timenotify;
TimeMeasurements for1;
TimeMeasurements for2;
TimeMeasurements rest;

double timeupdatemap_mean, timeupdatemap_SD;
double timenotify_mean, timenotify_SD;
double for1_mean, for1_SD;
double for2_mean, for2_SD;
double rest_mean, rest_SD;

/// A single instance of Octomap
//Gaussmap::Ptr gaussMap;

std::unique_ptr<mapping::Map> mapping::createMapGauss(int _mapSize, double resolution, double _raytraceFactor, int _pointThreshold) {
    return walkers::make_unique<Gaussmap>(_mapSize,  resolution, _raytraceFactor, _pointThreshold);
}

std::unique_ptr<mapping::Map> mapping::createMapGauss(int _mapSize, double resolution, double _raytraceFactor) {
    return walkers::make_unique<Gaussmap>(_mapSize,  resolution, _raytraceFactor);
}

std::unique_ptr<mapping::Map> mapping::createMapGauss(std::string mapPath, int _mapSize, double resolution, double _raytraceFactor) {
    return walkers::make_unique<Gaussmap>(mapPath, _mapSize,  resolution, _raytraceFactor);
}

//Multi-resolution surfel maps for efficient dense 3D modeling and tracking
//Careful treatment of the numerical stability is required when utilizing
//one-pass schemes for calculating the sample covariance [28]. We require a
//minimum sample size of |P| ≥ 10 to create surfels and stop incorporating
//new data points if |P| ≥ 10000^2. The discretization of disparity and color
//produced by the RGB-D sensor may cause degenerate sample covariances,
//which we robustly detect by thresholding the determinant of the covariance
//at a small constant.

Gaussmap::Gaussmap(int _mapSize, double resolution, double _raytraceFactor, int _pointThreshold) : map(_mapSize) {
    pointThreshold = _pointThreshold;
    mapSize = _mapSize;
    mapResolution = resolution;
    raytraceFactor = _raytraceFactor;
    xmin = ymin = zmin = -1.0 * mapSize * resolution/2.0;
    xmax = ymax = zmax = -1.0* xmin;
    indexes.clear();
    raytracing = true;
    limitUpdates = false;
    probabilityThreshold = 0.153;
    //preinitVoxels();
}

bool exists_test1 (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

Gaussmap::Gaussmap(std::string mapPath, int _mapSize, double resolution, double _raytraceFactor) : map(_mapSize)  {
    if(exists_test1(mapPath)) {
        std::cout<<"File found"<<std::endl;
    } else {
        std::cout<<"Map file not found"<<std::endl;
    }
    std::ifstream mapfile;
    mapfile.open(mapPath.c_str(), std::ios_base::binary);
    map.readBinary(mapfile);
    mapfile.close();
    mapSize = _mapSize;
    mapResolution = resolution;
    raytraceFactor = _raytraceFactor;
    xmin = ymin = zmin = -1 * mapSize * resolution/2;
    xmax = ymax = zmax = -1* xmin;
    indexes.clear();
    raytracing = true;
    loadIndexes(mapPath);
    limitUpdates = false;
    probabilityThreshold = 0.153;
}

/// set raytracing
void Gaussmap::setRaytracing(bool raytrace){
    raytracing = raytrace;
}

void Gaussmap::loadIndexes(std::string mapPath) {
    setlocale(LC_NUMERIC,"C");
    tinyxml2::XMLDocument doc;
    std::string xmlPath = mapPath + ".xml";
    doc.LoadFile(xmlPath.c_str());
    tinyxml2::XMLElement* parent = doc.FirstChildElement("indexes");
    tinyxml2::XMLElement *row = parent->FirstChildElement();
    while (row != NULL) {
        tinyxml2::XMLElement *col = row->FirstChildElement();
        Eigen::Vector3i index;
        while (col != NULL) {
            std::string sKey;
            int sVal;
            char *sTemp1 = (char *)col->Value();
            if (sTemp1 != NULL) {
                sKey = static_cast<std::string>(sTemp1);
            } else {
                sKey = "";
            }
            char *sTemp2 = (char *)col->GetText();
            if (sTemp2 != NULL) {
                std::string temp = static_cast<std::string>(sTemp2);
                sVal = std::stoi(temp);
            } else {
                sVal = 0;
            }
            if (sKey == "x") {
                index(0) = sVal;
            }
            if (sKey == "y") {
                index(1) = sVal;
            }
            if (sKey == "z") {
                index(2) = sVal;
            }
            col = col->NextSiblingElement();
        }
        std::string key = std::to_string(index(0)) + std::to_string(index(1)) + std::to_string(index(2));
        indexes[key] = index;
        row = row->NextSiblingElement();
    }
}

Gaussmap::Gaussmap(int _mapSize, double resolution, double _raytraceFactor) : map(_mapSize) {
    mapSize = _mapSize;
    mapResolution = resolution;
    raytraceFactor = _raytraceFactor;
    xmin = ymin = zmin = -1 * mapSize * resolution/2;
    xmax = ymax = zmax = -1* xmin;
    indexes.clear();
    limitUpdates = false;
    probabilityThreshold = 0.153;
    //preinitVoxels();
}

Gaussmap::Gaussmap(float vxmin, float vxmax, float vymin, float vymax, float vzmin, float vzmax, int _mapSize, double resolution, double _raytraceFactor) : map(_mapSize) {
    mapSize = _mapSize;
    mapResolution = resolution;
    raytraceFactor = _raytraceFactor;
    xmin = vxmin; xmax = vxmax;
    ymin = vymin; ymax = vymax;
    zmin = vzmin; zmax = vzmax;
    indexes.clear();
    limitUpdates = false;
    probabilityThreshold = 0.153;
    //preinitVoxels();
}

void Gaussmap::mapLoaded() {
    notifyMapNDTOM(map, indexes,0);
}

void Gaussmap::preinitVoxels() {
    for(int i = 0 ; i < map.size();  i++) {
        for(int j = 0 ; j < map.size();  j++) {
            for(int k = 0 ; k < map.size();  k++) {
                map(i, j, k).preinitParameters(mapResolution, Eigen::Vector3d(backwardXCoordinate(i), backwardYCoordinate(j), backwardZCoordinate(k)));
            }
        }
    }
}

/// Insert point cloud into map
void Gaussmap::insertCloud(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose, mapping::updateMethodType _updateType, bool raytrace)
{
    updateMap(pointCloud, _updateType, raytrace);

    notifyMapNDTOM(map, indexes,0);

    sceneCounter ++;
}

/* ===========================================================================================================================================================
   CUDA accelerated insertCloud() function
============================================================================================================================================================= */
// #include "demos/CUDAmap.h"
// void Gaussmap::insertCloudCUDA(SimplePoint* &pointCloud, const walkers::Mat34& cameraPose, mapping::updateMethodType _updateType, bool raytrace)
// {
//     currCameraFrame = cameraPose;

//     size_t size = sizeof(pointCloud) / sizeof(&pointCloud[0]);

//     uncertinatyErrors = std::vector<walkers::Mat33>(size, walkers::Mat33::Identity());

//     mapping::PointCloud cloud = pointCloudGlobal(pointCloud, cameraPose);

//     updateMap(cloud, _updateType, raytrace);

//     notifyMapNDTOM(map, indexes,0);

//     sceneCounter ++;
// }

/* ===========================================================================================================================================================
   END
============================================================================================================================================================= */

mapping::PointCloud Gaussmap::pointCloudGlobal(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose) const {
    mapping::PointCloud cloud(pointCloud);

    // Define a lambda function to be executed by each thread
    auto processPoint = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            walkers::Mat34 p(walkers::Mat34::Identity());
            p.matrix().block<3,1>(0,3) = cloud[i].position.vector();
            p = cameraPose * p;
            cloud[i].position.vector() = p.matrix().block<3,1>(0,3);
        }
    };

    // Get the number of available threads
    //const size_t numThreads = std::thread::hardware_concurrency();
    const size_t numThreads = 8;
    // Calculate the chunk size for each thread
    const size_t chunkSize = cloud.size() / numThreads;

    // Vector to store thread objects
    std::vector<std::thread> threads;

    // Launch threads
    for (size_t i = 0; i < numThreads - 1; ++i) {
        threads.emplace_back(processPoint, i * chunkSize, (i + 1) * chunkSize);
    }

    // Process the remaining elements in the main thread
    processPoint((numThreads - 1) * chunkSize, cloud.size());

    // Wait for threads to finish
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

    return cloud;
}


// /// point cloud from camera pose
// mapping::PointCloud Gaussmap::pointCloudGlobal(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose) const{
//    mapping::PointCloud cloud(pointCloud);
//    for (auto& point : cloud){
//        if (point.position.x()<0.3&&point.position.y()<0.3&&point.position.z()<0.3){
//            walkers::Mat34 p(walkers::Mat34::Identity());
//            p.matrix().block<3,1>(0,3) = point.position.vector();
//            p = cameraPose*p;
//            point.position.vector() = p.matrix().block<3,1>(0,3);
//        }
//    }
//    return cloud;
// }

/// raytrace map only without inserting points
void Gaussmap::raytraceMap(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose){
    currCameraFrame = cameraPose;
    uncertinatyErrors = std::vector<walkers::Mat33>(pointCloud.size(),walkers::Mat33::Identity());
    mapping::PointCloud cloud = pointCloudGlobal(pointCloud, cameraPose);
    int xCoor, yCoor, zCoor;
    for(mapping::Point3D &point : cloud) {
        xCoor = xCoordinate(point.position.x());
        yCoor = yCoordinate(point.position.y());
        zCoor = zCoordinate(point.position.z());
        if(xCoor >= map.size() || yCoor >=map.size() || zCoor>= map.size()) {
            raytracePoint(point);
        }
        else if(xCoor <0 || yCoor <0 || zCoor<0) {
            raytracePoint(point);
        }
        else {
            raytracePoint(point, xCoor, yCoor, zCoor);
        }
    }
}

/// Insert point cloud into map (increase probability)
void Gaussmap::insertCloudPositive(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose){
    currCameraFrame = cameraPose;
    mapping::PointCloud cloud = pointCloudGlobal(pointCloud, cameraPose);
    updateMapPositive(cloud);
}

/// load ellipsoids
void Gaussmap::loadEllipsoids(const std::string& filename){
    std::cout << "Load ellipsoids\n";
    std::ifstream ellifile(filename);
    size_t ellipsoidsNo =0;
    ellifile >> ellipsoidsNo;
    ellipsoids.resize(ellipsoidsNo);
    for (size_t ellipsoidNo = 0; ellipsoidNo<ellipsoidsNo; ellipsoidNo++){
        int r,g,b,a;
        ellifile >> r >> g >> b >> a;
        ellipsoids[ellipsoidNo].color.r = (uint8_t)r; ellipsoids[ellipsoidNo].color.g = (uint8_t)g; ellipsoids[ellipsoidNo].color.b = (uint8_t)b;
        ellipsoids[ellipsoidNo].color.a = (uint8_t)a;
        for (int row=0;row<3;row++){
            for (int col=0;col<3;col++){
                ellifile >> ellipsoids[ellipsoidNo].cov(row,col);
            }
        }
        size_t localMapNo;
        ellifile >> localMapNo;
    }
    ellifile.close();
    std::cout << "Ellipsoids loaded\n";
}

/// load map from file
void Gaussmap::loadMap(std::string filename){
//    std::ifstream mapfile;
//    mapfile.open(filename.c_str(), std::ios_base::binary);
//    map.readBinary(mapfile);
//    mapfile.close();
//    std::cout << "loaded\n";
//    notify3Dmap(map, mapResolution, indexes);
    std::ifstream mapfile(filename);
    mapfile >> mapResolution;
    for (int row=0;row<3;row++){
        for (int col=0;col<4;col++){
            mapfile >> mapFrame(row,col);
        }
    }
    size_t indicesSize;
    mapfile >> indicesSize;
//    std::cout << "indices size: " << indicesSize << "\n";
    for (size_t idx = 0; idx<indicesSize; idx++){
        Eigen::Vector3i index;
        mapfile >> index(0) >> index(1) >> index(2);
//        std::cout << index(0) << " " << index(1) << " " << index(2) << "\n";
        std::string key = std::to_string(index(0)) + std::to_string(index(1)) + std::to_string(index(2));
//        std::cout << "key " << key << "\n";
//        getchar();
//        indexes[key] = index;
        indexes.insert(std::make_pair(key, index));
        Eigen::Vector3d mean; walkers::Mat33 var; double probability; unsigned int sampNumber; Eigen::Vector3d sampMean; mapping::RGBA color; Eigen::Vector3d meanSum; walkers::Mat33 varSum;
        mapfile >> mean(0) >> mean(1) >> mean(2);
        for (int row=0; row<3;row++){
            for (int col=0; col<3;col++){
                mapfile >> var(row,col);
            }
        }
        mapfile >> probability;
        mapfile >> sampNumber;
        mapfile >> sampMean(0) >> sampMean(1) >> sampMean(2);
        int r; int g; int b; int a;
        mapfile >> r >> g >> b >> a;
        color.r = (uint8_t)r; color.g = (uint8_t)g; color.b = (uint8_t)b; color.a = (uint8_t)a;
//        mapfile >> meanSum(0) >> meanSum(1) >> meanSum(2);
        for (int row=0; row<3;row++){
            for (int col=0; col<3;col++){
                mapfile >> varSum(row,col);
            }
        }
        map(index(0), index(1), index(2)).setVoxelState(mean, var, probability, sampNumber, sampMean, color, meanSum, varSum);
    }
    mapfile >> boundEllipMean(0) >> boundEllipMean(1) >> boundEllipMean(2);
    for (int row=0; row<3;row++){
        for (int col=0; col<3;col++){
            mapfile >> boundEllipCov(row,col);
        }
    }
//    std::cout << indexes.size() << " size\n";
    mapfile.close();
//    notify3Dmap(map, indexes);
}

/// get bounding ellipsoid
void Gaussmap::getBoundingEllipsoid(Eigen::Vector3d& _boundEllipMean, walkers::Mat33& _boundEllipCov){
    _boundEllipMean = boundEllipMean;
    _boundEllipCov = boundEllipCov;
}

/// get bounding ellipsoid global
void Gaussmap::getBoundingEllipsoidGlobal(Eigen::Vector3d& _boundEllipMean, walkers::Mat33& _boundEllipCov){
    _boundEllipMean = mapFrame.matrix().block<3,3>(0,0)*boundEllipMean + mapFrame.matrix().block<3,1>(0,3);
    _boundEllipCov = mapFrame.matrix().block<3,3>(0,0)*boundEllipCov*mapFrame.matrix().block<3,3>(0,0).transpose();
}

/// save map in file
void Gaussmap::saveMap(std::string filename){
    std::cout << "save\n";
    std::ofstream mapfile(filename);
    mapfile << mapResolution << " ";
    for (int row=0;row<3;row++){
        for (int col=0;col<4;col++){
            mapfile << mapFrame(row,col) << " ";
        }
    }
    mapfile << indexes.size() << " ";
    for (const auto& idx : indexes){
        mapfile << idx.second(0) << " " << idx.second(1) << " " << idx.second(2) << " ";
        Eigen::Vector3d mean; walkers::Mat33 var; double probability; size_t sampNumber; Eigen::Vector3d sampMean; mapping::RGBA color; Eigen::Vector3d meanSum; walkers::Mat33 varSum;
        map(idx.second(0), idx.second(1), idx.second(2)).getVoxelState(mean, var, probability, sampNumber, sampMean, color, meanSum, varSum);
        mapfile << mean(0) << " " << mean(1) << " " << mean(2) << " ";
        for (int row=0; row<3;row++){
            for (int col=0; col<3;col++){
                mapfile << var(row,col) << " ";
            }
        }
        mapfile << (double)probability << " ";
        mapfile << sampNumber << " ";
        mapfile << sampMean(0) << " " << sampMean(1) << " " << sampMean(2) << " ";
        mapfile << (int)color.r << " "  << (int)color.g << " " << (int)color.b << " "  << (int)color.a << " ";
//        mapfile << (double)meanSum(0) << " " << (double)meanSum(1) << " " << (double)meanSum(2) << " ";
        for (int row=0; row<3;row++){
            for (int col=0; col<3;col++){
                if (std::isnan(varSum(row,col)))
                    mapfile << 0.0 << " ";
                else
                    mapfile << varSum(row,col) << " ";
            }
        }
    }
    mapfile << boundEllipMean(0) << " " << boundEllipMean(1) << " " << boundEllipMean(2) << " ";
    for (int row=0; row<3;row++){
        for (int col=0; col<3;col++){
            mapfile << boundEllipCov(row,col) << " ";
        }
    }
    mapfile.close();
    std::cout << "Done\n";
}

///set all cells to occupied
void Gaussmap::setMapOccupied(const walkers::Vec3& center, double distance){
    for(int i = 0 ; i < map.size();  i++) {
        for(int j = 0 ; j < map.size();  j++) {
            for(int k = 0 ; k < map.size();  k++) {
                mapping::VoxelVisu voxel;
                voxel.pose = walkers::Mat34::Identity();
                voxel.pose(0,3) = backwardXCoordinate(i);
                voxel.pose(1,3) = backwardYCoordinate(j);
                voxel.pose(2,3) = backwardZCoordinate(k);
                double dist=sqrt(pow(voxel.pose(0,3)-center.x(),2.0)+pow(voxel.pose(1,3)-center.y(),2.0)+pow(voxel.pose(2,3)-center.z(),2.0));
                if (dist<distance){
                    map(i, j, k).probability = 1.0;
                }
            }
        }
    }
}

/// sample from ellipsoid
walkers::Vec3 Gaussmap::sampleFromEllipsoid(const mapping::Ellipsoid& ellipsoid){
    SamplerEllipsoid sampler(ellipsoid);
    return sampler.getSample();
}

///Attach visualizer
void Gaussmap::attachVisualizer(Observer* visualizer){
    attach(visualizer);
}

/// print map
void Gaussmap::printMap(){

}

void Gaussmap::processFragment(const std::unordered_map<std::string, Eigen::Vector3i>& fragment, mapping::updateMethodType _updateType, int pointThreshold)
{
    for(const auto& n : fragment)
    {
        Eigen::Vector3i index = n.second;
        map(index.x(), index.y(), index.z()).updateWithSimpleMethod(_updateType, pointThreshold);
    }
}


void Gaussmap::processCloud(const mapping::PointCloud& cloud, size_t start, size_t end, bool raytrace)
{
    for(size_t i = start; i < end; ++i)
    {
        mapping::Point3D point = cloud[i];

        int xCoor, yCoor, zCoor;

        xCoor = xCoordinate(point.position.x());
        yCoor = yCoordinate(point.position.y());
        zCoor = zCoordinate(point.position.z());

        std::string key = std::to_string(xCoor) + std::to_string(yCoor) + std::to_string(zCoor);

        std::unordered_map<std::string, Eigen::Vector3i>::iterator got = simpleMethodIndexes.find(key);

        if(got == simpleMethodIndexes.end())
        {
            simpleMethodIndexes[key] = Eigen::Vector3i(xCoor, yCoor, zCoor);

            got = indexes.find(key);

            if(got == indexes.end())
            {
                indexes[key] = Eigen::Vector3i(xCoor, yCoor, zCoor);
            }
        }

        map(xCoor, yCoor, zCoor).insertPoint(point);
    }
}

void Gaussmap::updateMap(const mapping::PointCloud& cloud, mapping::updateMethodType _updateType, bool raytrace)
{
    simpleMethodIndexes.clear();

    size_t start = 0;
    size_t end = cloud.size();
    Gaussmap::processCloud(std::ref(cloud), start, end, raytrace);

    const int numThreads2 = 8;

    std::vector<std::thread> threads2;
    std::size_t fragmentSize = simpleMethodIndexes.size() / numThreads2;
    auto it = simpleMethodIndexes.begin();

    for(int i = 0; i < numThreads2; ++i)
    {
        auto fragmentBegin = it;
        std::advance(it, fragmentSize);
        auto fragmentEnd = (i == numThreads2 - 1) ? simpleMethodIndexes.end() : it;

        // Create a thread and pass the fragment to process
        threads2.emplace_back(&Gaussmap::processFragment, this, std::unordered_map<std::string, Eigen::Vector3i>(fragmentBegin, fragmentEnd), _updateType, pointThreshold);
    }

    // Wait for all threads to finish
    for(auto& thread : threads2)
    {
        thread.join();
    }
}

//---------------------------------------------------------- Poprawki prowadzącego koniec best and final-----------------------------------------------------------------------------------------------------


//update with positive raytracing
void Gaussmap::updateMapPositive(mapping::PointCloud cloud) {
    int xCoor, yCoor, zCoor;
    int i = 0;
    simpleMethodIndexes.clear();
    for(mapping::Point3D &point : cloud) {
        xCoor = xCoordinate(point.position.x());
        yCoor = yCoordinate(point.position.y());
        zCoor = zCoordinate(point.position.z());
        if(xCoor >= map.size() || yCoor >=map.size() || zCoor>= map.size()) {
            std::cout<<"Point out of bounds"<<std::endl;
        }
        else if(xCoor <0 || yCoor <0 || zCoor<0) {
            std::cout<<"Point out of bounds (negative)"<<std::endl;
            std::cout << point.position.vector() << "\n";
            std::cout << xCoor << " || " << yCoor << " || " << zCoor << "\n";
            std::cout << xmin << " " << xmax << "\n";
            std::cout << mapSize << "\n";
            std::cout << mapResolution << "\n";
            //getchar();
        }
        else {
            raytracePointPositive(point, 0.05);
            i++;
        }
    }
}

/// update bounding ellipsoid
void Gaussmap::computeBoundingEllipsoid(void){
    //mean
    boundEllipMean = Eigen::Vector3d(0, 0, 0);
    std::vector<Eigen::Vector3d> points(indexes.size());
    size_t pointNo = 0;
    for (const auto& coord : indexes){
        points[pointNo] = Eigen::Vector3d(backwardXCoordinate(coord.second(0)), backwardYCoordinate(coord.second(1)), backwardZCoordinate(coord.second(2)));
        boundEllipMean += points[pointNo];
        pointNo++;
    }
    boundEllipMean /= double(indexes.size());

    //cov
    boundEllipCov = walkers::Mat33::Zero();
    if(indexes.size() > 1) {
        for (const auto& point : points){
            boundEllipCov += (point - boundEllipMean) * (point - boundEllipMean).transpose();
        }
    }
    boundEllipCov =16*( boundEllipCov / double(indexes.size() - 1));
}

/// get all ellipsoids
void Gaussmap::getEllipsoids(mapping::Ellipsoid::Seq& _ellipsoids){
    _ellipsoids.clear();
    for (const auto& coord : indexes){
        if ((map(coord.second(0),coord.second(1),coord.second(2)).probability > probabilityThreshold)&&(map(coord.second(0),coord.second(1),coord.second(2)).var.determinant()!=1)) {
            mapping::Ellipsoid ellipsoid;
            ellipsoid.cov = map(coord.second(0),coord.second(1),coord.second(2)).var;
            ellipsoid.position.vector() = map(coord.second(0),coord.second(1),coord.second(2)).mean;
            ellipsoid.color = map(coord.second(0),coord.second(1),coord.second(2)).color;
            _ellipsoids.push_back(ellipsoid);
        }
    }
}

/// get all ellipsoids in global coordinate frame
void Gaussmap::getEllipsoidsGlobal(mapping::Ellipsoid::Seq& _ellipsoids){
    _ellipsoids.clear();
    for (const auto& coord : indexes){
        if ((map(coord.second(0),coord.second(1),coord.second(2)).probability > probabilityThreshold)&&(map(coord.second(0),coord.second(1),coord.second(2)).var.determinant()!=1)) {
            mapping::Ellipsoid ellipsoid;
            ellipsoid.cov = mapFrame.matrix().block<3,3>(0,0)*map(coord.second(0),coord.second(1),coord.second(2)).var*mapFrame.matrix().block<3,3>(0,0).transpose();
            ellipsoid.position.vector() = mapFrame.matrix().block<3,3>(0,0)*map(coord.second(0),coord.second(1),coord.second(2)).mean + mapFrame.matrix().block<3,1>(0,3);
            ellipsoid.color = map(coord.second(0),coord.second(1),coord.second(2)).color;
            _ellipsoids.push_back(ellipsoid);
        }
    }
}

void Gaussmap::getVoxels(mapping::VoxelVisu::Seq& voxels){
    voxels.clear();
    for (const auto& coord : indexes){
        if ((map(coord.second(0),coord.second(1),coord.second(2)).probability > probabilityThreshold)&&(map(coord.second(0),coord.second(1),coord.second(2)).var.determinant()!=1)) {
            mapping::VoxelVisu voxel;
            voxel.pose = walkers::Mat34::Identity();
            voxel.pose(0,3) = backwardXCoordinate(coord.second(0));
            voxel.pose(1,3) = backwardYCoordinate(coord.second(1));
            voxel.pose(2,3) = backwardZCoordinate(coord.second(2));
            voxel.color = map(coord.second(0),coord.second(1),coord.second(2)).color;
            voxel.width = mapResolution;
            voxels.push_back(voxel);
        }
    }
}

/// check if the voxel is occupied
bool Gaussmap::isOccupied(const walkers::Vec3& pos){
    int coord[3] = {xCoordinate(pos.x()), yCoordinate(pos.y()), zCoordinate(pos.z())};
    if ((map(coord[0], coord[1], coord[2]).probability > probabilityThreshold)&&(map(coord[0], coord[1], coord[2]).var.determinant()!=1))
        return true;
    else
        return false;
}

/// get color of the voxel
void Gaussmap::getColor(const walkers::Vec3& pos, mapping::RGBA& color){
    int coord[3] = {xCoordinate(pos.x()), yCoordinate(pos.y()), zCoordinate(pos.z())};
    color = map(coord[0], coord[1], coord[2]).color;
}

///update occupancy
void Gaussmap::updateVoxelOccupancy(const walkers::Vec3& pos, const mapping::RGBA& color){
    int x = xCoordinate(pos.x()); int y = yCoordinate(pos.y()); int z = zCoordinate(pos.z());
    map(x,y,z).probability = 1.0;
    map(x,y,z).var = walkers::Mat33::Identity();
    map(x,y,z).color = color;
    if(x >= map.size() || y >=map.size() || z>= map.size()) {
        std::cout<<"Point out of bounds"<<std::endl;
    }
    else if(x <0 || y <0 || z<0) {
        std::cout<<"Point out of bounds (negative)"<<std::endl;
    }
    else {
        std::string key = std::to_string(x) + std::to_string(y) + std::to_string(z);
        std::unordered_map<std::string, Eigen::Vector3i>::iterator got = indexes.find(key);
        if(got == indexes.end()) {
            indexes[key] = Eigen::Vector3i(x, y, z);
        }
        got = simpleMethodIndexes.find(key);
        if(got == simpleMethodIndexes.end()) {
            simpleMethodIndexes[key] = Eigen::Vector3i(x, y, z);
        }
    }
}

/// get all voxels in global coordinate frame
void Gaussmap::getVoxelsGlobal(mapping::VoxelVisu::Seq& voxels){
    voxels.clear();
    for (const auto& coord : indexes){
        if ((map(coord.second(0),coord.second(1),coord.second(2)).probability > probabilityThreshold)) {
            mapping::VoxelVisu voxel;
            voxel.pose = walkers::Mat34::Identity();
            voxel.pose(0,3) = backwardXCoordinate(coord.second(0));
            voxel.pose(1,3) = backwardYCoordinate(coord.second(1));
            voxel.pose(2,3) = backwardZCoordinate(coord.second(2));
            voxel.pose.matrix().block<3,3>(0,0) = mapFrame.matrix().block<3,3>(0,0)*voxel.pose.matrix().block<3,3>(0,0);
            voxel.pose.matrix().block<3,1>(0,3) = mapFrame.matrix().block<3,3>(0,0)*voxel.pose.matrix().block<3,1>(0,3) + mapFrame.matrix().block<3,1>(0,3);
            voxel.color = map(coord.second(0),coord.second(1),coord.second(2)).color;
            voxel.width = mapResolution;
            voxels.push_back(voxel);
        }
    }
}

// number of neighbours
size_t Gaussmap::getNumberOfNeigh(int x, int y, int z){
    size_t neighNo = 0;
    for (int i=-1;i<2;i++){
        for (int j=-1;j<2;j++){
            for (int k=-1;k<2;k++){
                if (i!=0&&j!=0&&k!=0){
                    if ((map(x+i,y+j,z+k).probability > probabilityThreshold)) {
                        neighNo++;
                    }
                }
            }
        }
    }
    return neighNo;
}

/// get all voxels in global coordinate frame
void Gaussmap::getVoxelsUnocupiedGlobal(mapping::VoxelVisu::Seq& voxels, const walkers::Vec3& center, double distance){
    voxels.clear();
    for(int i = 0 ; i < map.size();  i++) {
        for(int j = 0 ; j < map.size();  j++) {
            for(int k = 0 ; k < map.size();  k++) {
                mapping::VoxelVisu voxel;
                voxel.pose = walkers::Mat34::Identity();
                voxel.pose(0,3) = backwardXCoordinate(i);
                voxel.pose(1,3) = backwardYCoordinate(j);
                voxel.pose(2,3) = backwardZCoordinate(k);
                double dist=sqrt(pow(voxel.pose(0,3)-center.x(),2.0)+pow(voxel.pose(1,3)-center.y(),2.0)+pow(voxel.pose(2,3)-center.z(),2.0));
                if (dist<distance){
                    if ((map(i,j,k).probability > probabilityThreshold)) {
//                        std::cout << "ijk " << i << ", " << j << ", " << k << " " << map(i,j,k).probability<< "\n";
//                        std::cout << " map(i,j,k).probability  " << map(i,j,k).probability << "\n";
//                        getchar();
                        voxel.pose.matrix().block<3,3>(0,0) = mapFrame.matrix().block<3,3>(0,0)*voxel.pose.matrix().block<3,3>(0,0);
                        voxel.pose.matrix().block<3,1>(0,3) = mapFrame.matrix().block<3,3>(0,0)*voxel.pose.matrix().block<3,1>(0,3) + mapFrame.matrix().block<3,1>(0,3);
                        voxel.color = mapping::RGBA(120,120,120,255);
                        voxel.width = mapResolution;
                        if (getNumberOfNeigh(i, j, k)>3)
                            voxels.push_back(voxel);

                        //update indexes
                        std::string key = std::to_string(i) + std::to_string(j) + std::to_string(k);
                        std::unordered_map<std::string, Eigen::Vector3i>::iterator got = indexes.find(key);
                        if(got == indexes.end()) {
                            indexes[key] = Eigen::Vector3i(i, j, k);
                        }
                    }
                }
            }
        }
    }
}

/// get point cloud in global coordinate frame
void Gaussmap::getPointCloudOccupied(mapping::PointCloud& cloud, const walkers::Vec3& center, double distance){
    cloud.clear();
    std::default_random_engine generator;
    for(int i = 0 ; i < map.size();  i++) {
        for(int j = 0 ; j < map.size();  j++) {
            for(int k = 0 ; k < map.size();  k++) {
                mapping::VoxelVisu voxel;
                voxel.pose = walkers::Mat34::Identity();
                voxel.pose(0,3) = backwardXCoordinate(i);
                voxel.pose(1,3) = backwardYCoordinate(j);
                voxel.pose(2,3) = backwardZCoordinate(k);
                double dist=sqrt(pow(voxel.pose(0,3)-center.x(),2.0)+pow(voxel.pose(1,3)-center.y(),2.0)+pow(voxel.pose(2,3)-center.z(),2.0));
                if (dist<distance){
                    mapping::Point3D point;
                    if ((map(i,j,k).probability > 0.0)) {
                        size_t pointsNo = size_t(map(i,j,k).probability*100);
                        std::uniform_real_distribution<double> distribution(-mapResolution/2.0,mapResolution/2.0);
                        for (size_t pointNo=0;pointNo<pointsNo;pointNo++){
                            point.position.x() = voxel.pose(0,3) + distribution(generator);
                            point.position.y() = voxel.pose(1,3) + distribution(generator);
                            point.position.z() = voxel.pose(2,3) + distribution(generator);
                            point.color = map(i,j,k).color;
                            cloud.push_back(point);
                        }
                    }
                }
            }
        }
    }
}

void Gaussmap::raytracePoint(mapping::Point3D point, int x, int y, int z) {
//    Eigen::Vector3d incrementValue = Eigen::Vector3d((point.position.x() - currCameraFrame(0,3))/raytraceFactor, (point.position.y() - currCameraFrame(1,3))/raytraceFactor, (point.position.z() - currCameraFrame(2,3))/raytraceFactor);
    int maxIter = int(fabs(point.position.x() - currCameraFrame(0,3))/raytraceFactor);
    if (int(fabs((point.position.y() - currCameraFrame(1,3))/raytraceFactor))>maxIter)
        maxIter = int(fabs(point.position.y() - currCameraFrame(1,3))/raytraceFactor);
    if ((int)fabs((point.position.z() - currCameraFrame(2,3))/raytraceFactor)>maxIter)
        maxIter = int(fabs(point.position.z() - currCameraFrame(2,3))/raytraceFactor);
    Eigen::Vector3d incrementValue = Eigen::Vector3d((point.position.x() - currCameraFrame(0,3))/maxIter, (point.position.y() - currCameraFrame(1,3))/maxIter, (point.position.z() - currCameraFrame(2,3))/maxIter);
    Eigen::Vector3d incrementedPoint = currCameraFrame.matrix().block<3,1>(0,3);
    int i = 0;
    while (i < maxIter) {
        int xCoor = xCoordinate(incrementedPoint[0]);
        int yCoor = yCoordinate(incrementedPoint[1]);
        int zCoor = zCoordinate(incrementedPoint[2]);
        if (isInsideMap(xCoor, yCoor, zCoor)){
            if(xCoor != x && yCoor != y && zCoor != z ) {
                if (limitUpdates){
                    double dist = sqrt(pow(incrementedPoint[0]-centerUpdates.x(),2.0)+pow(incrementedPoint[1]-centerUpdates.y(),2.0)+pow(incrementedPoint[2]-centerUpdates.z(),2.0));
                    if (dist<distanceUpdates){
                        map(xCoor, yCoor, zCoor).updateNullOccupancy();
                    }
                }
                else
                    map(xCoor, yCoor, zCoor).updateNullOccupancy();
            }
        }
        incrementedPoint += incrementValue;
        i++;
    }
}

/// raytrace to point which is outside the map
void Gaussmap::raytracePoint(mapping::Point3D point) {
//    Eigen::Vector3d incrementValue = Eigen::Vector3d((point.position.x() - currCameraFrame(0,3))/raytraceFactor, (point.position.y() - currCameraFrame(1,3))/raytraceFactor, (point.position.z() - currCameraFrame(2,3))/raytraceFactor);
    int maxIter = int(fabs(point.position.x() - currCameraFrame(0,3))/raytraceFactor);
    if (int(fabs((point.position.y() - currCameraFrame(1,3))/raytraceFactor))>maxIter)
        maxIter = int(fabs(point.position.y() - currCameraFrame(1,3))/raytraceFactor);
    if ((int)fabs((point.position.z() - currCameraFrame(2,3))/raytraceFactor)>maxIter)
        maxIter = int(fabs(point.position.z() - currCameraFrame(2,3))/raytraceFactor);
    Eigen::Vector3d incrementValue = Eigen::Vector3d((point.position.x() - currCameraFrame(0,3))/maxIter, (point.position.y() - currCameraFrame(1,3))/maxIter, (point.position.z() - currCameraFrame(2,3))/maxIter);
    Eigen::Vector3d incrementedPoint = currCameraFrame.matrix().block<3,1>(0,3);
    int i = 0;
//    std::cout << incrementValue << "\n";
//    std::cout << "maxIter " << maxIter << "\n";
//    std::cout << incrementedPoint << " cam\n";
//    std::cout << point.position.vector() << " point\n";
    bool isInside=false;
    while (i < maxIter) {
//        std::cout << incrementedPoint << "\n";
        int xCoor = xCoordinate(incrementedPoint[0]);
        int yCoor = yCoordinate(incrementedPoint[1]);
        int zCoor = zCoordinate(incrementedPoint[2]);
//        std::cout << "xcor " << xCoor << ", " << yCoor << " " << zCoor << "\n";
        if (isInsideMap(xCoor, yCoor, zCoor)){
            isInside = true;
            if (limitUpdates){
                double dist = sqrt(pow(incrementedPoint[0]-centerUpdates.x(),2.0)+pow(incrementedPoint[1]-centerUpdates.y(),2.0)+pow(incrementedPoint[2]-centerUpdates.z(),2.0));
//                                    std::cout << "dist " << dist << " \n";
                if (dist<distanceUpdates){
//                    std::cout << "update\n";
                    map(xCoor, yCoor, zCoor).updateNullOccupancy();
                }
            }
            else
                map(xCoor, yCoor, zCoor).updateNullOccupancy();
//                            std::cout << xCoor << ", " << yCoor << ", " << zCoor << " " << map(xCoor, yCoor, zCoor).probability << "\n";
//                            getchar();
        }
        else{
            if (isInside)//again outside the map
                return;
        }
        incrementedPoint += incrementValue;
        i++;
//        getchar();
    }
}

/// positive raytracing
void Gaussmap::raytracePointPositive(mapping::Point3D point, double updateValue) {
//    Eigen::Vector3d incrementValue = Eigen::Vector3d((point.position.x() - currCameraFrame(0,3))/raytraceFactor, (point.position.y() - currCameraFrame(1,3))/raytraceFactor, (point.position.z() - currCameraFrame(2,3))/raytraceFactor);
    int maxIter = int(fabs(point.position.x() - currCameraFrame(0,3))/raytraceFactor);
    if (int(fabs((point.position.y() - currCameraFrame(1,3))/raytraceFactor))>maxIter)
        maxIter = int(fabs(point.position.y() - currCameraFrame(1,3))/raytraceFactor);
    if ((int)fabs((point.position.z() - currCameraFrame(2,3))/raytraceFactor)>maxIter)
        maxIter = int(fabs(point.position.z() - currCameraFrame(2,3))/raytraceFactor);
    Eigen::Vector3d incrementValue = Eigen::Vector3d((point.position.x() - currCameraFrame(0,3))/maxIter, (point.position.y() - currCameraFrame(1,3))/maxIter, (point.position.z() - currCameraFrame(2,3))/maxIter);
    Eigen::Vector3d incrementedPoint = currCameraFrame.matrix().block<3,1>(0,3);
    int i = maxIter-5;
    incrementedPoint += i*incrementValue;

    while (i < maxIter+5) {
        //std::cout << incrementedPoint << "\n";
        int xCoor = xCoordinate(incrementedPoint[0]);
        int yCoor = yCoordinate(incrementedPoint[1]);
        int zCoor = zCoordinate(incrementedPoint[2]);
//        if(xCoor != x && yCoor != y && zCoor != z ) {
            if (limitUpdates){
                double dist = sqrt(pow(incrementedPoint[0]-centerUpdates.x(),2.0)+pow(incrementedPoint[1]-centerUpdates.y(),2.0)+pow(incrementedPoint[2]-centerUpdates.z(),2.0));
                if (dist<distanceUpdates){//&&map(xCoor, yCoor, zCoor).probability>probabilityThreshold){
                    map(xCoor, yCoor, zCoor).probability+=updateValue;
                    break;
                }
            }
            //std::cout << xCoor << ", " << yCoor << ", " << zCoor << " " << map(xCoor, yCoor, zCoor).probability << "\n";
            //getchar();
//        }
        incrementedPoint += incrementValue;
        i++;
    }
}

double Gaussmap::normalize(double p, double min) {
    return (fmod((p - min), mapResolution) * 2/mapResolution) - 1.0;
}


int Gaussmap::xCoordinate(double x) {
    if (x>xmax||x<xmin)
        return -1;
    double  a = mapSize/(xmax-xmin);
    double b = -1 * a*xmin;
    return int(a*x + b);
}

int Gaussmap::yCoordinate(double y) {
    if (y>ymax||y<ymin)
        return -1;
    double  a = mapSize/(ymax-ymin);
    double b = -1 * a*ymin;
    return int(a*y + b);
}

int Gaussmap::zCoordinate(double z) {
    if (z>zmax||z<zmin)
        return -1;
    double  a = mapSize/(zmax-zmin);
    double b = -1 * a*zmin;
    return int(a*z + b);
}

double Gaussmap::backwardXCoordinate(int x) {
    double  a = mapSize/(xmax-xmin);
    double b = -1 * a*xmin;
    return (x - b) / a;
}

double Gaussmap::backwardYCoordinate(int y) {
    double  a = mapSize/(ymax-ymin);
    double b = -1 * a*ymin;
    return (y - b) / a;
}

double Gaussmap::backwardZCoordinate(int z) {
    double  a = mapSize/(zmax-zmin);
    double b = -1 * a*zmin;
    return (z - b) / a;
}

std::string Gaussmap::currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d-%X", &tstruct);

    return buf;
}

std::vector<std::string> Gaussmap::split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

/// remove floor
void Gaussmap::removeFloor(double maxHeight){
//    double probabilityThreshold(65.3);
    for (const auto& coord : indexes){
//        if ((map(coord.second(0),coord.second(1),coord.second(2)).probability < probabilityThreshold)) {
            if (map(coord.second(0),coord.second(1),coord.second(2)).mean(2)<maxHeight){
                map(coord.second(0),coord.second(1),coord.second(2)).probability = 0;
            }
//        }
    }
    std::vector<mapping::Ellipsoid> cleanedFloor;
    cleanedFloor.reserve(ellipsoids.size());
    size_t eliNo=0;
    for (const auto& eli : ellipsoids){
        if (eli.position.z()<maxHeight){
        }
        else{
            cleanedFloor.push_back(eli);
            eliNo++;
        }
    }
    ellipsoids = cleanedFloor;
    notifyMapNDTOM(map, indexes,0);
}

///is inside the map
bool Gaussmap::isInsideMap(int x, int y, int z) const{
    if (x>=0 && x<map.size() && y>=0 && y<map.size() && z>=0 && z<map.size())
        return true;
    else
        return false;
}

///set all cells to occupied
void Gaussmap::setLimitUpdates(const walkers::Vec3& center, double distance){
    centerUpdates = center;
    distanceUpdates = distance;
    limitUpdates = true;
}

/// initialize mesh model from the map
void Gaussmap::initializeMeshModel(ObjectsMesh& objects3DS){
    ObjTypeLoader obj;
    for (const auto& coord : indexes){
        if ((map(coord.second(0),coord.second(1),coord.second(2)).probability > probabilityThreshold)) {
            for (int i=-1;i<2;i=i+2){
                for (int j=-1;j<2;j=j+2){
                    for (int k=-1;k<2;k=k+2){
                        Vertex vert;
                        vert.vertex.x = float(map(coord.second(0),coord.second(1),coord.second(2)).mean(0) + (i*mapResolution/2.0));
                        vert.vertex.y = float(map(coord.second(0),coord.second(1),coord.second(2)).mean(1) + (j*mapResolution/2.0));
                        vert.vertex.z = float(map(coord.second(0),coord.second(1),coord.second(2)).mean(2) + (k*mapResolution/2.0));
                        obj.vertices.push_back(vert);
                    }
                }
            }
            PolygonType pol;
            pol.a = (short unsigned int)(obj.vertices.size() - 8); pol.b = (short unsigned int)(obj.vertices.size() - 7); pol.c = (short unsigned int)(obj.vertices.size() - 6); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 7); pol.b = (short unsigned int)(obj.vertices.size() - 5); pol.c = (short unsigned int)(obj.vertices.size() - 6); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 8); pol.b = (short unsigned int)(obj.vertices.size() - 3); pol.c = (short unsigned int)(obj.vertices.size() - 7); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 8); pol.b = (short unsigned int)(obj.vertices.size() - 4); pol.c = (short unsigned int)(obj.vertices.size() - 3); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 7); pol.b = (short unsigned int)(obj.vertices.size() - 3); pol.c = (short unsigned int)(obj.vertices.size() - 5); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 5); pol.b = (short unsigned int)(obj.vertices.size() - 3); pol.c = (short unsigned int)(obj.vertices.size() - 1); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 8); pol.b = (short unsigned int)(obj.vertices.size() - 6); pol.c = (short unsigned int)(obj.vertices.size() - 4); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 6); pol.b = (short unsigned int)(obj.vertices.size() - 2); pol.c = (short unsigned int)(obj.vertices.size() - 4); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 6); pol.b = (short unsigned int)(obj.vertices.size() - 5); pol.c = (short unsigned int)(obj.vertices.size() - 2); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 5); pol.b = (short unsigned int)(obj.vertices.size() - 1); pol.c = (short unsigned int)(obj.vertices.size() - 2); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 2); pol.b = (short unsigned int)(obj.vertices.size() - 1); pol.c = (short unsigned int)(obj.vertices.size() - 3); obj.polygons.push_back(pol);
            pol.a = (short unsigned int)(obj.vertices.size() - 4); pol.b = (short unsigned int)(obj.vertices.size() - 2); pol.c = (short unsigned int)(obj.vertices.size() - 3); obj.polygons.push_back(pol);
        }
    }
    objects3DS.objects.push_back(obj);
}

/// initialize mesh model from the map
void Gaussmap::initializeMeshModelEllipsoids(ObjectsMesh& objects3DS){
    ObjTypeLoader obj;
    double voxelSize = 0.05;
    size_t count =0;
    for (const auto& eli : ellipsoids){
        for (int i=-1;i<2;i=i+2){
            for (int j=-1;j<2;j=j+2){
                for (int k=-1;k<2;k=k+2){
                    Vertex vert;
                    vert.vertex.x = float(eli.position.x() + (i*voxelSize/2.0));
                    vert.vertex.y = float(eli.position.y() + (j*voxelSize/2.0));
                    vert.vertex.z = float(eli.position.z() + (k*voxelSize/2.0));
                    obj.vertices.push_back(vert);
                }
            }
        }
        PolygonType pol;
        pol.a = (size_t)(obj.vertices.size() - 8); pol.b = (size_t)(obj.vertices.size() - 7); pol.c = (size_t)(obj.vertices.size() - 6); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 7); pol.b = (size_t)(obj.vertices.size() - 5); pol.c = (size_t)(obj.vertices.size() - 6); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 8); pol.b = (size_t)(obj.vertices.size() - 3); pol.c = (size_t)(obj.vertices.size() - 7); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 8); pol.b = (size_t)(obj.vertices.size() - 4); pol.c = (size_t)(obj.vertices.size() - 3); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 7); pol.b = (size_t)(obj.vertices.size() - 3); pol.c = (size_t)(obj.vertices.size() - 5); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 5); pol.b = (size_t)(obj.vertices.size() - 3); pol.c = (size_t)(obj.vertices.size() - 1); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 8); pol.b = (size_t)(obj.vertices.size() - 6); pol.c = (size_t)(obj.vertices.size() - 4); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 6); pol.b = (size_t)(obj.vertices.size() - 2); pol.c = (size_t)(obj.vertices.size() - 4); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 6); pol.b = (size_t)(obj.vertices.size() - 5); pol.c = (size_t)(obj.vertices.size() - 2); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 5); pol.b = (size_t)(obj.vertices.size() - 1); pol.c = (size_t)(obj.vertices.size() - 2); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 2); pol.b = (size_t)(obj.vertices.size() - 1); pol.c = (size_t)(obj.vertices.size() - 3); obj.polygons.push_back(pol);
        pol.a = (size_t)(obj.vertices.size() - 4); pol.b = (size_t)(obj.vertices.size() - 2); pol.c = (size_t)(obj.vertices.size() - 3); obj.polygons.push_back(pol);
        count++;
    }
    objects3DS.objects.push_back(obj);
}
