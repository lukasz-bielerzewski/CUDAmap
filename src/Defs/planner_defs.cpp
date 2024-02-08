/** @file planner_defs.cpp
*
* Planner definitions
*
*/

#include "Defs/planner_defs.h"
#include <iostream>

using namespace planner;

void planner::robotPoseSE32Mat34(const planner::PoseSE3& poseSE3, walkers::Mat34& pose34){
    pose34 = walkers::Mat34::Identity();
    pose34(0,3) = poseSE3.getPosition().x();
    pose34(1,3) = poseSE3.getPosition().y();
    pose34(2,3) = poseSE3.getPosition().z();
    pose34.matrix().block<3,3>(0,0) = poseSE3.getRotation().matrix();
}

/// conver robotState3D 2 RobotPose
void planner::robotState3D2RobotPose(const RobotState3D& robotState3D, RobotPose& robotPose){
    robotPose.feet = robotState3D.feet;
    walkers::Mat34 body(walkers::Mat34::Identity());
    robotPoseSE32Mat34(robotState3D.robotPose,body);
    robotPose.body = body;
}

/// get pos
Eigen::Vector3d PoseSE3::getPosition() const{
    return pos;
}

/// get rot
walkers::Quaternion PoseSE3::getRotation() const{
    return rot;
}

/// get pose
walkers::Mat34 PoseSE3::getPose() const{
    return pose;
}

/// set pos
void PoseSE3::setPosition(const Eigen::Vector3d& _pos){
    pos=_pos;
    pose = walkers::toSE3(pos,rot);
}

/// set rot
void PoseSE3::setRotation(const walkers::Quaternion& _rot){
    rot=_rot;
    pose = walkers::toSE3(pos,rot);
}

/// set pose
void PoseSE3::setPose(const walkers::Mat34& _pose){
    pose=_pose;
    pos(0) = _pose(0,3); pos(1) = _pose(1,3); pos(2) = _pose(2,3);
    rot = walkers::rotMat2quat(pose);
}

/// show foot state
void Foot::show(void) const {
    std::cout << "pose: \n" << pose.matrix() << std::endl;
    (isFoothold) ? std::cout << "is foothold\n" : std::cout << "is not foothold\n";
}

/// show current pose
void RobotPose::show(void) const {
    std::cout << "body: \n" << body.matrix() << std::endl;
    int footNo = 0;
    for (Foot::Seq::const_iterator it = feet.begin(); it!=feet.end(); it++){
        std::cout << "foot " << footNo << ": \n"; it->show();
        footNo++;
    }
}

/// show node properties
void RRTNode::show(void) const {
    std::cout << "id: " << id << ", parent: " << parent << std::endl;
    std::cout << "Robot pose \n" << robotPose.body.matrix() << "\n";
//                int stateNo = 0;
//                for (RobotPose::Seq::const_iterator it = path.begin(); it!=path.end(); it++){
//                    std::cout << "State Number " << stateNo << ": \n"; it->show();
//                    stateNo++;
//                }
}

RRTree::RRTree(){
    branchesNo = 1;
    nodesNo=0;
    nodes.clear();
}

/// add node
void RRTree::addNode(RRTNode& node){
    if (node.parent>=0){
        auto it = nodes.find(node.parent);
        if (it != nodes.end()){
            node.cost2root = it->second.cost2root + dist2parent(node);
            it->second.children.insert((int)nodesNo); //add child
        }
        else{
            std::cout << "this->branchesNo " << this->branchesNo << "\n";
            std::cout << "this->nodesNo " << this->nodesNo << "\n";
            std::cout << "mymap contains:\n";
            for (it=nodes.begin(); it!=nodes.end(); ++it)
                std::cout << it->first << " => " << it->second.id << "parent id: " << it->second.parent << '\n';
            std::cout << "nodes.size() " << nodes.size() << "\n";
            std::cout << "parent no: " << node.parent << "\n";
            throw std::runtime_error("RRTree (add): parent index out of bounds.\n");
        }
    }
    else
        node.cost2root = dist2parent(node);
    node.id = nodesNo;
    nodes.insert(std::pair<size_t,RRTNode>(nodesNo,node));
    nodesNo++;
}

/// get node
RRTNode RRTree::getNode(size_t nodeNo) const{
    auto it = nodes.find(nodeNo);
    if (it != nodes.end()){
        return it->second;
    }
    else {
        std::cout << "this->branchesNo " << this->branchesNo << "\n";
        std::cout << "this->nodesNo " << this->nodesNo << "\n";
        std::cout << "mymap contains:\n";
        for (it=nodes.begin(); it!=nodes.end(); ++it)
            std::cout << it->first << " => " << it->second.id << '\n';
        std::cout << "nodes.size() " << nodes.size() << "\n";
        std::cout << "index no: " << nodeNo << "\n";
        throw std::runtime_error("RRTree (getter): index out of bounds.\n");
    }
}

/// set joint
void RRTree::setJoint(size_t nodeNo, bool isJoint) {
    RRTNode node = getNodeRef(nodeNo);
    node.isJoint = isJoint;
}

/// get node
RRTNode& RRTree::getNodeRef(size_t nodeNo) {
    auto it = nodes.find(nodeNo);
    if (it != nodes.end()){
        return it->second;
    }
    else {
        std::cout << "index no: " << nodeNo << "\n";
        throw std::runtime_error("RRTree (getter ref): index out of bounds.\n");
    }
}

/// remove node
bool RRTree::removeNode(size_t nodeNo){
    RRTNode node = getNodeRef(nodeNo);
    int parentId = node.parent;
    RRTNode nodeParent = getNodeRef(parentId);
    // erase nodeNo from children list in the parent node
    nodeParent.children.erase((int)nodeNo);
    // erase node
    nodes.erase(nodeNo);
    return true;
}

/// remove branch
bool RRTree::removeBranch(size_t nodeNo){
    auto it = nodes.find(nodeNo);
    if (it != nodes.end()){
        bool isJoint = false;
        int parentId = it->second.parent;
        int nodeId = (int)nodeNo;
        while (!isJoint){
            auto itParent = nodes.find(parentId);
            if (itParent->second.children.size()>1||itParent->second.isJoint||parentId==0){//joint or root
                // parent is a joint
                isJoint = true;
                itParent->second.children.erase(nodeId);
            }
            // remove node
            removeNode(nodeId);
            nodeId = parentId;
            parentId = itParent->second.parent;
        }
        return true;
    }
    else {
        std::cout << "index no: " << nodeNo << "\n";
        throw std::runtime_error("RRTree (remove branch): Could not remove branch. Index out of bounds.\n");
    }
    return false;
}

/// get size
size_t RRTree::getSize(void) const {
    return nodes.size();
}

/// update count
void RRTree::updateBranchesNo(void){
    branchesNo++;
}

/// get count
size_t RRTree::getBranchesNo(void) const{
    return branchesNo;
}

/// clear tree
void RRTree::clear(void){
    nodes.clear();
    nodesNo = 0;
    branchesNo = 1;
}

/// compute distance to parent node
double RRTree::dist2parent(const RRTNode& node) const{
    if (node.parent<0)
        return 0;
    auto parent = nodes.find(node.parent);
    if (parent != nodes.end()){
        return sqrt(pow(node.robotPose.body(0,3)-parent->second.robotPose.body(0,3),2.0)+
                pow(node.robotPose.body(1,3)-parent->second.robotPose.body(1,3),2.0)+
                pow(node.robotPose.body(2,3)-parent->second.robotPose.body(2,3),2.0));
    }
    else{
        throw std::runtime_error("dist2parent: parent does not exist\n");
    }
    return -1.0;
}
