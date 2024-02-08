#ifndef _GAUSSMAP_H
#define _GAUSSMAP_H
#include "Defs/defs.h"
#include "Mapping/voxel.h"
#include "3rdParty/octree/octree.h"
#include "Utilities/observer.h"
#include "map.h"

extern double timeupdatemap_mean, timeupdatemap_SD;
extern double timenotify_mean, timenotify_SD;
extern double for1_mean, for1_SD;
extern double for2_mean, for2_SD;
extern double rest_mean, rest_SD;

class QGLVisualizer;//forward declaration

namespace mapping {
    /// create a single Maping object
    std::unique_ptr<mapping::Map> createMapGauss(int _mapSize, double resolution, double _raytraceFactor, int _pointThreshold);
    std::unique_ptr<mapping::Map> createMapGauss(int _mapSize, double resolution, double _raytraceFactor);
    std::unique_ptr<mapping::Map> createMapGauss(std::string mapPath, int _mapSize, double resolution, double _raytraceFactor);
}

class Gaussmap : public mapping::Map, public Subject {
private:

    std::mutex simpleMethodIndexes_mutex;
    std::mutex Indexes_mutex;

    Octree<mapping::Voxel> map;
    std::vector<walkers::Mat33> uncertinatyErrors;

    std::unordered_map<std::string, Eigen::Vector3i> indexes;
    std::unordered_map<std::string, Eigen::Vector3i> simpleMethodIndexes;

    //Map borders
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;

    double timeSum = 0.0;
    double sceneCounter = 0;
    std::vector<double> times;

    int mapSize;
    double mapResolution;
    double raytraceFactor;
    int pointThreshold;

    /// bounding ellipsoid mean
    Eigen::Vector3d boundEllipMean;
    /// bounding ellipsoid cov
    walkers::Mat33 boundEllipCov;
    /// map pose
    walkers::Mat34 currCameraFrame;
    /// raytracing
    bool raytracing;

    void preinitVoxels();
    void raytracePoint(mapping::Point3D point, int x, int y, int z);
    /// raytrace to point which is outside the map
    void raytracePoint(mapping::Point3D point);

    /// positive raytracing
    void raytracePointPositive(mapping::Point3D point, double updateValue);

    // number of neighbours
    size_t getNumberOfNeigh(int x, int y, int z);

    /// limit updates
    bool limitUpdates;
    ///center of updates
    walkers::Vec3 centerUpdates;
    ///distance from center;
    double distanceUpdates;
    /// double probabilityThreshold(0.153);
    double probabilityThreshold;
    /// ellipsoids
    std::vector<mapping::Ellipsoid> ellipsoids;

protected:
    //void processCloud(const mapping::PointCloud& cloud, int start, int end, bool raytrace,
    //                  std::unordered_map<std::string, Eigen::Vector3i>& localIndexes,
    //                  std::unordered_map<std::string, Eigen::Vector3i>& localSimpleMethodIndexes);

    //void updateMap(const mapping::PointCloud& cloud, mapping::updateMethodType _updateType, bool raytrace);

    //Protected functions
    void updateMap(const mapping::PointCloud& cloud, mapping::updateMethodType updateType, bool raytrace);
    //void processCloud_index(const mapping::PointCloud& cloud, int start, int end, bool raytrace);
    //void processCloud_simpleMethodIndexes(const mapping::PointCloud& cloud, int start, int end, bool raytrace);
    void processCloud(const mapping::PointCloud& cloud, size_t start, size_t end, bool raytrace);
    void processFragment(const std::unordered_map<std::string, Eigen::Vector3i>& fragment, mapping::updateMethodType _updateType, int pointThreshold);


    //update with positive raytracing
    void updateMapPositive(mapping::PointCloud cloud);
    /// point cloud from camera pose
    mapping::PointCloud pointCloudGlobal(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose) const;
    /// update bounding ellipsoid
    void computeBoundingEllipsoid(void);
    double normalize(double p, double min);
    int xCoordinate(double x);
    int yCoordinate(double y);
    int zCoordinate(double z);
    double backwardXCoordinate(int x);
    double backwardYCoordinate(int y);
    double backwardZCoordinate(int z);
    ///is inside the map
    bool isInsideMap(int x, int y, int z) const;

public:
    /// Pointer
    typedef std::unique_ptr<Gaussmap> Ptr;

    Gaussmap(int mapSize, double _resolution, double _raytraceFactor, int _pointThreshold);
    Gaussmap(int mapSize, double _resolution, double _raytraceFactor);
    Gaussmap(std::string mapPath, int mapSize, double _resolution, double _raytraceFactor);
    Gaussmap(float vxmin, float vxmax, float vymin, float vymax, float vzmin, float vzmax, int mapSize, double _resolution, double _raytraceFactor);

    /// load ellipsoids
    void loadEllipsoids(const std::string& filename);

    /// get ellipsoids
    mapping::Ellipsoid::Seq getEllipsoids() const {return ellipsoids;};

    /// Name of the map
    const std::string& getName() const {return name;}

    /// Insert point cloud into map
    void insertCloud(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose, mapping::updateMethodType _updateType, bool raytrace);

    /// Insert point cloud into map (increase probability)
    void insertCloudPositive(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose);

    /// save map in file
    void saveMap(std::string filename);

    /// load map from file
    void loadMap(std::string filename);

    /// print map
    void printMap();

    ///Attach visualizer
    void attachVisualizer(Observer* visualizer);

    void mapLoaded();

    /// get bounding ellipsoid
    void getBoundingEllipsoid(Eigen::Vector3d& _boundEllipMean, walkers::Mat33& _boundEllipCov);

    /// get bounding ellipsoid global
    void getBoundingEllipsoidGlobal(Eigen::Vector3d& _boundEllipMean, walkers::Mat33& _boundEllipCov);

    /// sample from ellipsoid
    static walkers::Vec3 sampleFromEllipsoid(const mapping::Ellipsoid& ellipsoid);

    /// descrutor
    ~Gaussmap() {}

    void loadIndexes(std::string mapPath);

    ///set all cells to occupied
    void setMapOccupied(const walkers::Vec3& center, double distance);

    ///set all cells to occupied
    void setLimitUpdates(const walkers::Vec3& center, double distance);

    std::string currentDateTime();

    std::vector<std::string> split(const std::string &s, char delim);

    /// get all ellipsoids
    void getEllipsoids(mapping::Ellipsoid::Seq& ellipsoids);
    /// get all ellipsoids global
    void getEllipsoidsGlobal(mapping::Ellipsoid::Seq& ellipsoids);

    /// get all voxels
    void getVoxels(mapping::VoxelVisu::Seq& voxels);
    /// get all voxels global
    void getVoxelsGlobal(mapping::VoxelVisu::Seq& voxels);
    /// get point cloud in global coordinate frame
    void getPointCloudOccupied(mapping::PointCloud& cloud, const walkers::Vec3& center, double distance);
    /// get voxels for unnocupied cells global
    void getVoxelsUnocupiedGlobal(mapping::VoxelVisu::Seq& voxels, const walkers::Vec3& center, double distance);

    ///get resolution
    double getResolution() {
        return mapResolution;
    }

    ///get map size
    int getMapSize() { return mapSize; }

    double getXmin(){ return xmin; }

    double getYmin(){ return ymin; }

    double getXmax(){ return xmax; }

    double getYmax(){ return ymax;}

    /// check if the voxel is occupied
    bool isOccupied(const walkers::Vec3& pos);

    /// get color of the voxel
    void getColor(const walkers::Vec3& pos, mapping::RGBA& color);

    ///update occupancy
    void updateVoxelOccupancy(const walkers::Vec3& pos, const mapping::RGBA& color);

    /// set raytracing
    void setRaytracing(bool raytrace);

    /// remove floor
    void removeFloor(double maxHeight);

    /// initialize mesh model from the map
    void initializeMeshModel(ObjectsMesh& objects3DS);

    /// initialize mesh model for the ellipsoids
    void initializeMeshModelEllipsoids(ObjectsMesh& objects3DS);

    /// raytrace map only without inserting points
    void raytraceMap(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose);
};

#endif //_GAUSSMAP_H
