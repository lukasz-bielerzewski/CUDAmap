/** @file map.h
 *
 * Map interface
 *
 */

#ifndef _MAP_H_
#define _MAP_H_
#include "Defs/defs.h"
#include "Defs/mapping_defs.h"

/// Map interface
namespace mapping {
    class Map {
    public:

        /// Map type
        enum Type {
            /// RGB camera
            TYPE_OCTOMAP,
            /// 2D Depth sensor
            TYPE_PROB,
        };

        ///constructor
        Map(void) : name("unknown"), mapFrame(walkers::Mat34::Identity()){}

        /// overloaded constructor
        Map(const std::string _name, Type _type) : name(_name), type(_type), mapFrame(walkers::Mat34::Identity()) {}

        /// Name of the map
        virtual const std::string& getName() const = 0;

        /// Insert point cloud into map
        virtual void insertCloud(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose, mapping::updateMethodType _updateType, bool raytrace) = 0;

        /// save map in file
        virtual void saveMap(std::string filename) = 0;

        /// load map from file
        virtual void loadMap(std::string filename) = 0;

        /// print map
        virtual void printMap() = 0;

        ///Attach visualizer
        //virtual void attachVisualizer(QGLVisualizer* visualizer) = 0;

        virtual void mapLoaded() = 0;

        /// get bounding ellipsoid
        virtual void getBoundingEllipsoid(Eigen::Vector3d& _boundEllipMean, walkers::Mat33& _boundEllipCov) = 0;

        /// get bounding ellipsoid global
        virtual void getBoundingEllipsoidGlobal(Eigen::Vector3d& _boundEllipMean, walkers::Mat33& _boundEllipCov) = 0;

        ///get resolution
        virtual double getResolution() = 0;

        ///get map size
        virtual int getMapSize() = 0;

        ///set all cells to occupied
        virtual void setMapOccupied(const walkers::Vec3& center, double distance) = 0;

        ///set all cells to occupied
        virtual void setLimitUpdates(const walkers::Vec3& center, double distance) = 0;

        /// check if the voxel is occupied
        virtual bool isOccupied(const walkers::Vec3& pos) = 0;

        /// raytrace map only without inserting points
        virtual void raytraceMap(const mapping::PointCloud& pointCloud, const walkers::Mat34& cameraPose) = 0;

        /// update map pose
        void updateMapPose(const walkers::Mat34& mapPose){
            mapFrame = mapPose;
        }

        /// update map pose
        void getMapPose(walkers::Mat34& mapPose){
            mapPose = mapFrame;
        }

        /// Get color of the voxel
        virtual void getColor(const walkers::Vec3& pos, mapping::RGBA& color) = 0;

        ///update occupancy
        virtual void updateVoxelOccupancy(const walkers::Vec3& pos, const mapping::RGBA& color) = 0;

        /// set raytracing
        virtual void setRaytracing(bool raytrace) = 0;

        /// remove floor
        virtual void removeFloor(double maxHeight) = 0;

        /// initialize mesh model from the map
//        virtual void initializeMeshModel(Objects3DS& objects3DS) = 0;

        /// Virtual descrutor
        virtual ~Map() {}

    protected:
        /// Map name
        const std::string name;

        /// Map type
        Type type;

        /// Map frame
        walkers::Mat34 mapFrame;
    };
}


#endif // _MAP_H_
