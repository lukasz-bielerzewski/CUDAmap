//CPP File
#include "Utilities/observer.h"
#include "Utilities/TimeRange.h"
#include <algorithm>
#include <iostream>

using namespace std;

TimeMeasurements notifyupdatemap;
double notifyupdatemap_mean, notifyupdatemap_SD;

void Subject::attach(Observer *observer){
    list.push_back(observer);
}

void Subject::detach(Observer *observer){
    list.erase(std::remove(list.begin(), list.end(), observer), list.end());
}

void Subject::addRenderObjects(std::vector<simulator::RenderObject>& renderObjects){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter) {
        if(*iter != 0) {
            (*iter)->addRenderObjects(renderObjects);
        }
    }
}

void Subject::notifyRenderObjects(const std::vector<simulator::RenderObject>& renderObjects){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter) {
        if(*iter != 0) {
            (*iter)->updateRenderObjects(renderObjects);
        }
    }
}

void Subject::notifySimTime(int simTime){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter) {
        if(*iter != 0) {
            (*iter)->updateSimTime(simTime);
        }
    }
}

void Subject::notifyGround(const simulator::RenderObjectHeightmap& ground) {
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter) {
        if(*iter != 0) {
            (*iter)->initGround(ground);
        }
    }

}
void Subject::add3DSModel(const ObjectsMesh& objects3DS){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->addMesh(objects3DS);
//            (*iter)->init3DS(objects3DS);
        }
    }

}

void Subject::setShadowsVisible(bool shadowVisible) {
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->setShadowsVisible(shadowVisible);
        }
    }

}

void Subject::updateTree(const planner::RRTree& tree, size_t treeNo, double _lineWidth, double _pointSize,
                         const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor) const{
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->updateTree(tree, treeNo, _lineWidth, _pointSize, _lineColor, _pointColor);
        }
    }
}

void Subject::updateKinematicModel(const walkers::Mat34& pose, const std::vector<double>& configuration){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->updateKinematicModel(pose, configuration);
        }
    }
}

/// RRT - update random position
void Subject::updateRandomPosition(const walkers::Vec3& randPos){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->updateRandomPosition(randPos);
        }
    }
}

/// RRT - add planned path
void Subject::addRobotPath(const std::vector<planner::RobotState3D>& robotPath, double _lineWidthRobot, double _pointSizeRobot,
                             const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                                  double _lineWidthFeet, double _pointSizeFeet,
                                  const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->addRobotPath(robotPath, _lineWidthRobot, _pointSizeRobot, _lineColorRobot, _pointColorRobot,
                                    _lineWidthFeet, _pointSizeFeet, _lineColorFeet, _pointColorFeet);
        }
    }
}

/// RRT - update planned path
void Subject::updateRobotPath(const std::vector<planner::RobotState3D>& robotPath, size_t pathNo,
                           double _lineWidthRobot, double _pointSizeRobot,
                           const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                           double _lineWidthFeet, double _pointSizeFeet,
                           const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->updateRobotPath(robotPath, pathNo, _lineWidthRobot, _pointSizeRobot, _lineColorRobot, _pointColorRobot,
                                    _lineWidthFeet, _pointSizeFeet, _lineColorFeet, _pointColorFeet);
        }
    }
}

/// update ellipsoids
void Subject::updateEllipsoids(const mapping::Ellipsoid::Seq& _ellipsoids, size_t ellipsoidNo){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->updateEllipsoids(_ellipsoids, ellipsoidNo);
        }
    }
}

/// add path
void Subject::addPath(const std::vector<planner::PoseSE3>& path, double _lineWidth, double _pointSize,
                      const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->addPath(path, _lineWidth, _pointSize, _lineColor, _pointColor);
        }
    }
}

/// update path
void Subject::updatePath(const std::vector<planner::PoseSE3>& path, size_t pathNo, double _lineWidth,
                         double _pointSize, const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->updatePath(path, pathNo, _lineWidth, _pointSize, _lineColor, _pointColor);
        }
    }
}

/// clear rrt trees
void Subject::clearRRTTrees(void){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter){
        if(*iter != 0) {
            (*iter)->clearRRTTrees();
        }
    }
}

void Subject::notifyMapNDTOM(Octree<mapping::Voxel>& map3D, std::unordered_map<std::string, Eigen::Vector3i> indexes,
                          size_t mapNo){


    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter) {
        if(*iter != 0) {

           //std::chrono::steady_clock::time_point begin_update = std::chrono::steady_clock::now();

           (*iter)->updateMapNDTOM(map3D, indexes, mapNo);

           //std::chrono::steady_clock::time_point end_update = std::chrono::steady_clock::now();
           //notifyupdatemap.addMeasurement(end_update-begin_update);
        }
    }

    //notifyupdatemap_mean=notifyupdatemap.calculateMean();
    //notifyupdatemap_SD=notifyupdatemap.calculateStandardDeviation();

}

///update robot mesh pose and map mesh pose
void Subject::updateRobotMapPose(const std::vector<walkers::Mat34>& _robotAndMapPose) {
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter) {
        if(*iter != 0) {
           (*iter)->updateRobotMapPose(_robotAndMapPose);
        }
    }
}

/// update Axis
void Subject::updateAxis(const walkers::Mat34& _line, size_t axisNo, double lineWidth, double scale){
    for(vector<Observer*>::const_iterator iter = list.begin(); iter != list.end(); ++iter) {
        if(*iter != 0) {
           (*iter)->updateAxis(_line, axisNo, lineWidth, scale);
        }
    }
}
