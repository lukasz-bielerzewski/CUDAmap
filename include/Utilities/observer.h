#ifndef OBSERVER_H_
#define OBSERVER_H_

#include "3rdParty/octree/octree.h"
#include "Defs/simulator_defs.h"
#include "Defs/planner_defs.h"
#include "Mapping/voxel.h"
#include "Utilities/objectsMesh.h"

extern double notifyupdatemap_mean, notifyupdatemap_SD;

class Observer {
public:

    virtual void addRenderObjects(std::vector<simulator::RenderObject>& renderObjects) = 0;
    virtual void updateRenderObjects(const std::vector<simulator::RenderObject>& renderObjects) = 0;
    virtual void initGround(const simulator::RenderObjectHeightmap& ground) = 0;
    virtual void updateSimTime(int simTime) = 0;
    virtual void addMesh(const ObjectsMesh& objects3DS) = 0;
    virtual void setShadowsVisible(bool shadowVisible) = 0;
    virtual void updateTree(const planner::RRTree& tree, size_t treeNo, double _lineWidth, double _pointSize,
                            const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor) = 0;
    virtual void updateKinematicModel(const walkers::Mat34& pose, const std::vector<double>& configuration) = 0;
    /// RRT - update random position
    virtual void updateRandomPosition(const walkers::Vec3& randPos) = 0;
    /// RRT - update planned path
    virtual void addRobotPath(const std::vector<planner::RobotState3D>& robotPath, double _lineWidthRobot,
                              double _pointSizeRobot,
                              const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                              double _lineWidthFeet, double _pointSizeFeet,
                              const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet) = 0;
    /// RRT - update planned path
    virtual void updateRobotPath(const std::vector<planner::RobotState3D>& robotPath, size_t pathNo,
                         double _lineWidthRobot, double _pointSizeRobot,
                         const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                         double _lineWidthFeet, double _pointSizeFeet,
                         const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet) = 0;
    /// clear rrt trees
    virtual void clearRRTTrees(void) = 0;
    /// add path
    virtual void addPath(const std::vector<planner::PoseSE3>& path, double _lineWidth, double _pointSize,
                 const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor) = 0;
    /// update path
    virtual void updatePath(const std::vector<planner::PoseSE3>& path, size_t pathNo, double _lineWidth, double _pointSize,
                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor) = 0;
    /// update 3D map
    virtual void updateMapNDTOM(const Octree<mapping::Voxel>& map3D, const std::unordered_map<std::string, Eigen::Vector3i> indexes,
                                size_t mapNo) = 0;
    ///update robot mesh pose and map mesh pose
    virtual void updateRobotMapPose(const std::vector<walkers::Mat34>& _robotAndMapPose) = 0;
    /// update Axis
    virtual void updateAxis(const walkers::Mat34& _line, size_t axisNo, double lineWidth, double scale) = 0;
    /// update ellipsoids
    virtual void updateEllipsoids(const mapping::Ellipsoid::Seq& _ellipsoids, size_t ellipsoidNo) = 0;
};

class Subject
{
    //Lets keep a track of all the shops we have observing
    std::vector<Observer*> list;

public:
    void attach(Observer *observer);
    void detach(Observer *observer);

    void addRenderObjects(std::vector<simulator::RenderObject>& renderObjects);
    void notifyRenderObjects(const std::vector<simulator::RenderObject>& renderObjects);
    void notifyGround(const simulator::RenderObjectHeightmap& ground);
    void notifySimTime(int simTime);
    void add3DSModel(const ObjectsMesh& objects3DS);
    void setShadowsVisible(bool shadowVisible);
    void updateTree(const planner::RRTree& tree, size_t treeNo, double _lineWidth, double _pointSize,
                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor) const;
    void updateKinematicModel(const walkers::Mat34& pose, const std::vector<double>& configuration);
    /// RRT - update random position
    void updateRandomPosition(const walkers::Vec3& randPos);
    /// RRT - update planned path
    void addRobotPath(const std::vector<planner::RobotState3D>& robotPath, double _lineWidthRobot, double _pointSizeRobot,
                      const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                      double _lineWidthFeet, double _pointSizeFeet,
                      const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet);
    /// RRT - update planned path
    void updateRobotPath(const std::vector<planner::RobotState3D>& robotPath, size_t pathNo,
                         double _lineWidthRobot, double _pointSizeRobot,
                         const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                         double _lineWidthFeet, double _pointSizeFeet,
                         const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet);
    /// update ellipsoids
    void updateEllipsoids(const mapping::Ellipsoid::Seq& _ellipsoids, size_t ellipsoidNo);
    /// clear rrt trees
    void clearRRTTrees(void);
    void addPath(const std::vector<planner::PoseSE3>& path, double _lineWidth, double _pointSize,
                 const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
    /// update path
    void updatePath(const std::vector<planner::PoseSE3>& path, size_t pathNo, double _lineWidth, double _pointSize,
                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
    /// update 3D map
    void notifyMapNDTOM(Octree<mapping::Voxel>& map3D, std::unordered_map<std::string, Eigen::Vector3i> indexes, size_t mapNo);
    ///update robot mesh pose and map mesh pose
    void updateRobotMapPose(const std::vector<walkers::Mat34>& _robotAndMapPose);
    /// update Axis
    void updateAxis(const walkers::Mat34& _line, size_t axisNo, double lineWidth, double scale);

private:

};

#endif // OBSERVER_H_
