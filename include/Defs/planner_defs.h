/** @file planner_defs.h
*
* Planner definitions
*
*/

#ifndef PLANNER_DEFS_H_INCLUDED
#define PLANNER_DEFS_H_INCLUDED

#include "defs.h"
#include <set>

/// planner name space
namespace planner {

    /// Leg type
    enum GaitType {
        /// Tripod
        GAIT_TRIPOD,
        /// Wave gait
        GAIT_WAVE,
    };

    /// Foot
    class Foot {
        public:
            /// set of feet
            typedef std::vector<Foot> Seq;

            /// foot pose
            walkers::Mat34 pose;

            /// is foothold
            bool isFoothold;

            /// Constructor
            Foot(void) : pose(walkers::Mat34::Identity()){
                isFoothold = false;
            }

            /// Constructor
            Foot(walkers::Mat34 _footPose, bool _isFoothold) : pose(_footPose), isFoothold(_isFoothold){
            }

            /// show foot state
            void show(void) const;
    };

    /// Robot
    class RobotPose {
        public:
            /// set of features
            typedef std::vector<RobotPose> Seq;

            /// feet poses
            Foot::Seq feet;

            /// feet poses
            Foot::Seq feetInit;

            /// body pose
            walkers::Mat34 body;

            /// gait type
            GaitType gaitType;

            /// show current pose
            void show(void) const;

            ///construction
            RobotPose(void) : body(walkers::Mat34::Identity()), gaitType(GAIT_TRIPOD){}
    };

    /// RRT Node
    class RRTNode {
        public:
        RRTNode () : stepIter(0), extendFailuresNo(0), cost2root(0.0){
        }
            /// set of nodes
            typedef std::vector<RRTNode> Seq;

            /// transition to parent node
            RobotPose::Seq path;

            /// robotPose and legs position for the node
            RobotPose robotPose;

            /// int parent node
            int parent;

            /// step iterator
            int stepIter;

            /// number of failure extend operations
            int extendFailuresNo;

            /// identifier
            size_t id;

            /// cost to root
            double cost2root;

            /// is joint
            bool isJoint;

            /// children
            std::set<int> children;

            /// show node properties
            void show(void) const;
    };

    class RRTree {
    public:
        RRTree();
        /// add node
        void addNode(RRTNode& node);
        /// get node
        RRTNode getNode(size_t nodeNo) const;
        /// get node
        RRTNode& getNodeRef(size_t nodeNo);
        /// get size
        size_t getSize(void) const;
        /// update count
        void updateBranchesNo(void);
        /// remove node
        bool removeNode(size_t nodeNo);
        /// remove branch
        bool removeBranch(size_t nodeNo);
        /// get count
        size_t getBranchesNo(void) const;
        /// clear
        void clear(void);
        /// set joint
        void setJoint(size_t nodeNo, bool isJoint);
    private:
        /// tree container
        std::map<size_t,RRTNode> nodes;
        /// number of nodes
        size_t nodesNo;
        /// number of updates (it is not equal to the number of nodes)
        size_t branchesNo;
        /// compute distance to parent node
        double dist2parent(const RRTNode& node) const;
    };

    class RobotState2D {
    public:
        /// position
        double x;
        double y;
        double yaw;

        /// Construction
        RobotState2D(void) : x(0), y(0), yaw(0) {
        }

        /// Construction
        RobotState2D(double _x, double _y, double _yaw) : x(_x), y(_y), yaw(_yaw) {
        }
    };

    class PoseSE3 {
    private:
        /// position
        Eigen::Vector3d pos;
        /// orientation
        walkers::Quaternion rot;
        /// pose
        walkers::Mat34 pose;

    public:
        /// Construction
        PoseSE3(void) : pos(0,0,0), rot(walkers::Quaternion::Identity()) {
            pose = walkers::toSE3(pos,rot);
        }

        /// Construction
        PoseSE3(const Eigen::Vector3d& _pos, const walkers::Quaternion _rot) : pos(_pos), rot(_rot) {
            pose = walkers::toSE3(pos,rot);
        }

        /// get pos
        Eigen::Vector3d getPosition() const;
        /// get rot
        walkers::Quaternion getRotation() const;
        /// get pose
        walkers::Mat34 getPose() const;

        /// set pos
        void setPosition(const Eigen::Vector3d& _pos);
        /// set rot
        void setRotation(const walkers::Quaternion& _rot);
        /// set pose
        void setPose(const walkers::Mat34& _pose);
    };

    class RobotState {
    public:
        /// Robot pose
        PoseSE3 robotPose;

        /// Joint angles
        std::vector<double> jointAngles;

        ///Construction
        RobotState(void) : robotPose(Eigen::Vector3d(0,0,0),walkers::Quaternion::Identity()){}

        ///Construction
        RobotState(const PoseSE3& _robotPose) : robotPose(_robotPose){}
    };

    class RobotState3D {
    public:
        /// Robot pose
        PoseSE3 robotPose;

        /// Joint angles
        std::vector<Foot> feet;

        /// Joint angles
        std::vector<Foot> feetInit;

        /// gait type
        GaitType gaitType;

        ///Construction
        RobotState3D(void) : robotPose(Eigen::Vector3d(0,0,0),walkers::Quaternion::Identity()), gaitType(GAIT_TRIPOD){}

        ///Construction
        RobotState3D(const PoseSE3& _robotPose) : robotPose(_robotPose), gaitType(GAIT_TRIPOD){}
    };


    void robotPoseSE32Mat34(const planner::PoseSE3& poseSE3, walkers::Mat34& pose34);
    /// conver robotState3D 2 RobotPose
    void robotState3D2RobotPose(const RobotState3D& robotState3D, RobotPose& robotPose);
}

#endif // PLANNER_DEFS_H_INCLUDED
