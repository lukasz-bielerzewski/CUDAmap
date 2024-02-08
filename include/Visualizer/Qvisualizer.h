/** @file QVisualizer.h
 *
 * implementation - QGLVisualizer
 *
 */

#ifndef QVISUALIZER_H_INCLUDED
#define QVISUALIZER_H_INCLUDED

#include "Defs/defs.h"
#include "Defs/planner_defs.h"
#include "Utilities/observer.h"
#include "Utilities/stopwatch.h"
#include "Utilities/recorder.h"
#include "Mapping/elevationMap.h"
#include "Utilities/objectsMesh.h"
#include "Visualizer/DisplayObject.h"

/// displayed object
#include "Visualizer/Displays.h"
#include "Visualizer/PointCloud.h"
#include "Visualizer/Line.h"
#include "Visualizer/Arrow.h"
#include "Visualizer/BSpline3D.h"
#include "Visualizer/BSplineSE3.h"
#include "Visualizer/Axis.h"
#include "Visualizer/ElevationMesh.h"
#include "Visualizer/Ellipsoids.h"
#include "Visualizer/Path.h"
#include "Visualizer/RobotPath.h"
#include "Visualizer/Tree.h"
#include "Visualizer/Mesh.h"
#include "Visualizer/Polynomial.h"
#include "Visualizer/Voxels.h"
#include "Visualizer/Plane.h"
#include "Visualizer/Triangle.h"
#include "Visualizer/MapNDTOM.h"
#include "Visualizer/TrajectorySE3.h"

#include "Defs/mapping_defs.h"
#include <QGLViewer/qglviewer.h>

extern double time_mutex_mean, time_mutex_SD;
extern double timedisplaylist_mean, timedisplaylist_SD;

/// Map implementation
class QGLVisualizer: public QGLViewer, public Observer{
public:
    /// Pointer
    typedef std::unique_ptr<QGLVisualizer> Ptr;

    class Config
    {

      public:
        Config(){
        }

        Config(std::string configFilename){
            load(configFilename);
        }

        void load(std::string configFilename);

        public:
            /// verbose
            bool verbose;
            /// Background color
            QColor backgroundColor;
            /// rrtBegin color
            QColor rrtBeginColor;
            /// draw rrtBegin tree
            bool drawRRTbegin;
            /// rrtBegin line width
            double rrtBeginWidth;
            /// draw rrtFinish tree
            bool drawRRTfinish;
            /// rrtFinish col
            QColor rrtFinishColor;
            /// rrtBegin line width
            double rrtFinishWidth;
            /// visualize collisions
            bool visualizeCollisions;
            /// visualize footholds
            bool visualizeFootholds;
            /// point size
            double pointSize;
            /// line color
            mapping::RGBA lineColor;
            /// line width
            double lineWidth;
            /// draw normal vectors to the terrain surface
            bool drawNormals;
            /// draw dynamic engine robot
            bool drawObjects;
            /// draw kinematic model of the robot
            bool drawKinematicObjects;
            /// draw Polynomial Fitting
            bool drawPolyFitt;
            /// display text status
            bool displayTextStatus;
    };

    /// Construction
    QGLVisualizer(void);

    /// Construction
    QGLVisualizer(std::string configFilename);

    /// Construction
    QGLVisualizer(std::string configFilename, std::string& robotConfig, std::string& _robotType, std::string& _coldetType,
                  std::string& _coldetConfig);

    /// Destruction
    ~QGLVisualizer(void);

    /// Observer update
    void updateRobotStateTraj(const RecorderRobotState& _robotStateTraj);
    /// add ellipsoids
    void addEllipsoids(const mapping::Ellipsoid::Seq& _ellipsoids);
    /// update ellipsoids
    void updateEllipsoids(const mapping::Ellipsoid::Seq& _ellipsoids, size_t ellipsoidNo);
    /// update voxels
    void updateVoxels(const mapping::VoxelVisu::Seq& _voxels, size_t voxelsNo);
    /// add voxels
    void addVoxels(const mapping::VoxelVisu::Seq& _voxels);
    /// add render objects (ids of added objects are in the structure RenderObjects)
    void addRenderObjects(std::vector<simulator::RenderObject>& _renderObjects);
    /// update render objects
    void updateRenderObjects(const std::vector<simulator::RenderObject>& _renderObjects);
    void renderObject(simulator::RenderObject& object);
    void initGround(const simulator::RenderObjectHeightmap& ground);
    /// update color of the ground
    void updateGroundColor(const std::vector<std::vector<std::vector<double>>>& groundColors);
    void setShadowsVisible(bool shadowVisible);
    void updateSimTime(int simTime);
    /// update robot trajectory
    void updateRobotTrajectories(const std::vector<walkers::Mat34>& _bodyPath, const std::vector<planner::Foot::Seq>& _feetPaths,
                                 size_t trajNo);
    /// initialize kinematic model
    void initializeKinematicModel(const walkers::Mat34& pose, const std::vector<double>& configuration);
    /// update kinematic model
    void updateKinematicModel(const walkers::Mat34& pose, const std::vector<double>& configuration);
    /// load map
    void loadMap(std::string configFilename);
    /// draw 3D map
    void drawMapNDTOM();
    /// draw ellipsoids
    void drawEllipsoids(void);
    /// draw voxels
    void drawVoxels(void);
    /// add point cloud
    void addCloud(const mapping::PointCloud& _cloud);
    /// update cloud
    void updateCloud(const mapping::PointCloud& _cloud, size_t cloudNo);
    /// modify existing point cloud
    void updateCloud(const mapping::PointCloud& _cloud, size_t cloudNo, double pointSize);
    /// modify existing point cloud
    void updateCloud(const mapping::PointCloud& _cloud, size_t cloudNo, double pointSize, const mapping::RGBA& color);
    /// add path
    void addPath(const std::vector<planner::PoseSE3>& path, double _lineWidth, double _pointSize,
                 const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
    /// update path
    void updatePath(const std::vector<planner::PoseSE3>& path, size_t pathNo, double _lineWidth, double _pointSize,
                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
    /// add trajectory
    void addTrajectory(const Recorder6D& path, double _lineWidth, double _pointSize,
                 const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
    /// update path
    void updateTrajectory(const Recorder6D& path, size_t pathNo, double _lineWidth, double _pointSize,
                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
    /// add robot path
    void addRobotPath(const std::vector<planner::RobotState3D>& robotPath, double _lineWidthRobot,
                      double _pointSizeRobot,
                      const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                      double _lineWidthFeet, double _pointSizeFeet,
                      const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet);
    /// update robot path
    void updateRobotPath(const std::vector<planner::RobotState3D>& robotPath, size_t pathNo,
                         double _lineWidthRobot, double _pointSizeRobot,
                         const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                         double _lineWidthFeet, double _pointSizeFeet,
                         const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet);
    /// add polynomial
    void addPolynomial(regression::Regression* _poly, double _centerX, double _centerY, double _offsetZ, double _surfWidth,
                       double _surfLength, size_t _verticesNo);
    /// update polynomial
    void updatePolynomial(regression::Regression* _poly, size_t polyNo, double _centerX, double _centerY, double _offsetZ,
                          double _surfWidth, double _surfLength, size_t _verticesNo);
    /// add mapNDTOM
    void addMapNDTOM(const Octree<mapping::Voxel>& map, const std::unordered_map<std::string, Eigen::Vector3i> indexes);
    /// update mapNDTOM
    void updateMapNDTOM(const Octree<mapping::Voxel>& map, const std::unordered_map<std::string, Eigen::Vector3i> indexes,
                     size_t mapNo);

    /// RRT - update random position
    void updateRandomPosition(const walkers::Vec3& randPos);
    /// clear rrt trees
    void clearRRTTrees(void);

    ///update robot mesh pose and map mesh pose
    void updateRobotMapPose(const std::vector<walkers::Mat34>& robotAndMapPose);

    /// add Arrow 3D
    void addArrow(const walkers::Arrow3D& _arrow, double lineWidth,
                  const mapping::RGBA& lineColor);
    /// update Arrow 3D
    void updateArrow(const walkers::Arrow3D& _arrow, size_t arrowNo);
    /// update Arrow 3D
    void updateArrow(const walkers::Arrow3D& _arrow, size_t arrowNo, double arrowWidth,
                     const mapping::RGBA& arrowColor);

    /// add line 3D
    void addLine(const walkers::Line3D& _line);
    /// update Lines 3D
    void updateLine(const walkers::Line3D& _line, size_t lineNo);
    /// update Lines 3D
    void updateLine(const walkers::Line3D& _line, size_t lineNo, const mapping::RGBA& color);

    /// add BSpline
    void addBSpline3D(const walkers::BSpline3D& _bspline);
    /// update BSplines
    void updateBSpline3D(const walkers::BSpline3D& _bspline, size_t bsplineNo);
    /// update BSplines
    void updateBSpline3D(const walkers::BSpline3D& _bspline, size_t bsplineNo, size_t pointsNo, double lineWidth,
                         const mapping::RGBA& lineColor, const mapping::RGBA& ctrlLineColor, double ctrlPointSize,
                         bool drawCtrlPoints, bool drawCtrlLines);

    /// add BSpline SE3
    void addBSplineSE3(const walkers::BSplineSE3& _bspline);
    /// update BSplines SE3
    void updateBSplineSE3(const walkers::BSplineSE3& _bspline, size_t bsplineNo);
    /// update BSplines SE3
    void updateBSplineSE3(const walkers::BSplineSE3& _bspline, size_t bsplineNo, size_t pointsNo, double lineWidth,
                         const mapping::RGBA& lineColor, const mapping::RGBA& ctrlLineColor, double ctrlPointSize,
                         bool drawCtrlPoints, bool drawCtrlLines, bool drawAxis, double framesScale);

    /// add axis
    void addAxis(const walkers::Mat34& _axis, double lineWidth, double scale);
    /// update Axis
    void updateAxis(const walkers::Mat34& _line, size_t axisNo, double lineWidth, double scale);

    /// add plane
    void addPlane(const std::vector<walkers::Plane3D>& _plane, const mapping::RGBA& color);
    /// update planes
    void updatePlane(const std::vector<walkers::Plane3D>& _plane, size_t planeNo, const mapping::RGBA& color);

    /// add triangle
    void addTriangle(const std::vector<walkers::Triangle3D>& _triangle, const mapping::RGBA& color);
    /// update triangles
    void updateTriangle(const std::vector<walkers::Triangle3D>& _triangle, size_t triangleNo, const mapping::RGBA& color);

    /// add rrt trees
    void addTree(const planner::RRTree& tree, double _lineWidth, double _pointSize,
                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
    /// update rrt trees
    void updateTree(const planner::RRTree& tree, size_t treeNo, double _lineWidth, double _pointSize,
                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
    /// draw trees
    void drawTrees(void);

    /// set current robot pose to the last added trajectory
    void setCurrentRobotPose(void);

    /// displa text status
    void setTextStatus(bool state);

    /// set camera pose
    void setCameraPose(const walkers::Mat34& cameraPose);

    /// generate dataset - camera views
    void generateViews(const std::vector<walkers::Mat34>& cameraPoses2save,
                       const std::vector<int>& _parentIds, const std::vector<double>& _pitchs,
                       const std::vector<double>& _yaws, size_t offsetImageNo);
    /// close window
    void closeWindow(void);

    /// get object pose
    walkers::Mat34 getObjectPose(const std::string& objectName);

    /// get object position on the image
    std::pair<int,int> getObjectPositionUV(const std::string& objectName, double& distance2surface);

    /// save screenshot
    void saveScreenshot(const std::string& filename);

    /// is busy with snap
    bool isBusyWithSnap(void);

private:
    ///config
    Config config;
    /// update state of the kinematic model
    bool updateStateKinemModel = false;
    /// pose of the kinematic robot's model
    walkers::Mat34 kinemRobotPose;

    /// robot and map pose (mesh)
    std::vector<walkers::Mat34> robotAndMapPose;

    ///objects 2 draw
    std::vector<simulator::RenderObject> renderObjects;

    ///ground
    simulator::RenderObjectHeightmap groundObj;
    bool groundInit = false;
    bool normalsInit = false;

    ///mutex
    std::mutex mtxObjects;
    bool init3DSModel = false;
    /// draw objects
    void draw();

    /// draw objects
    void animate();

    ///render
    void renderScene();
    void renderGround();
    void renderNormals();

    /// add mesh
    void addMesh(const ObjectsMesh& _mesh);
    /// update mesh
    void updateMesh(const ObjectsMesh& _mesh, size_t meshNo);

    /// initialize visualizer
    void init();

    /// generate help string
    std::string help() const;

    bool shadowFlag = false;

    /// simulation time from start
    int simulationTime;// miliseconds

    /// simulation time mutex
    std::mutex mtxSimTime;

    /// simulation time from start
    Stopwatch<std::chrono::milliseconds> realTime;// miliseconds

    /// recorders
    std::vector<Recorder6D> recorders6D;

    ///robot state trajectory
    RecorderRobotState robotStateTraj;
    /// robot state traj mutex
    std::mutex mtxRobotStateTraj;

    /// type of the robot
    std::string robotType;
    /// ref values for kinematic model
    std::vector<double> refAnglesKinem;
    /// selected leg
    size_t selectedLeg;

    std::list<std::pair<double,walkers::RobotState>>::iterator currentRobotState;
    std::list<std::pair<double,walkers::Mat34>>::iterator currentRobotPose;

    /// Map of the environment
    std::unique_ptr<ElevationMap> elevationMap;

    /// RRT- random position
    walkers::Vec3 randPosRRT;
    /// rand pos RRT mtx
    std::mutex mtxRandPosRRT;
    /// rand pos RRT mtx
    std::mutex mtxPlannedPath;
    /// planned path
    std::vector<planner::RobotState3D> plannedPath;
    /// pose iterator
    size_t planedPoseIterator;
    /// visualize classes
    bool visualizeClasses;
    /// visualize classes
    bool visualizeCurvature;
    /// visualize footholds
    bool visualizeFootholdsCost;
    /// 3D map
    Octree<mapping::Voxel> map3D;
    std::unordered_map<std::string, Eigen::Vector3i> updatedVoxels;

    /// display elevation maps/classes/footholds/curvature
    Displays<DisplayElevationMesh> displayElevationMeshes;
    /// display point cloud
    Displays<DisplayPointCloud> displayPointClouds;
    /// display lines
    Displays<DisplayLine> displayLines;
    /// display BSplines
    Displays<DisplayBSpline3D> displayBSplines3D;
    /// display BSplines SE3
    Displays<DisplayBSplineSE3> displayBSplinesSE3;
    /// display arrows
    Displays<DisplayArrow> displayArrows;
    /// display axis
    Displays<DisplayAxis> displayAxis;
    /// display ellipsoids
    Displays<DisplayEllipsoids> displayEllipsoids;
    /// display paths
    Displays<DisplayPath> displayPaths;
    /// display robot paths
    Displays<DisplayRobotPath> displayRobotPaths;
    /// display trees
    Displays<DisplayTree> displayTrees;
    /// display meshes
    Displays<DisplayMesh> displayMeshes;
    /// display polynomials
    Displays<DisplayPolynomial> displayPolynomials;
    /// display voxels
    Displays<DisplayVoxels> displayVoxels;
    /// display planes
    Displays<DisplayPlanes> displayPlanes;
    /// display triangles
    Displays<DisplayTriangles> displayTriangles;
    /// display mapNDTOM
    Displays<DisplayMapNDTOM> displayMapNDTOM;
    /// display trajectories SE3
    Displays<DisplayTrajectorySE3> displayTrajectories;

    /// display normals to the terrain surface
//    DisplayObject displayTerrainNormals;

    /// capture the key
    void keyPressEvent(QKeyEvent* key);
    /// draw text status
    void drawTextStatus(void);
    /// drawRobotPlannedPaths
    void drawRobotPlannedPaths(void);
    /// update kinematic objects
//    void updateKinematicObjects(const std::vector<simulator::RenderObject>& kinemObjects);
    ///draw registered robot state
    void drawRegisteredRobotState(void);
    /// visualize collisions
    void visualizeCollisions(void);
    /// visualize footholds
    void visualizeFootholds(void);
    /// visualize random position RRT
    void drawRandPosRRT(void);
    ///show planned pose
    void showPlannedPose(size_t _poseNo);
    /// get class color
    std::array<double,3> getClassColor(int classId);
    /// draw point cloud
    void drawPointClouds(void);
    /// draw lines
    void drawLines(void);
    /// draw arrows
    void drawArrows(void);
    /// draw BSplines 3D
    void drawBSplines3D(void);
    /// draw BSplines SE3
    void drawBSplinesSE3(void);
    /// draw axes
    void drawAxes(void);
    /// draw normals to the surface
    void drawTerrainNormals();
    ///draw polynomial fitting example
    void drawPolynomialFitting();
    /// draw planes
    void drawPlanes(void);
    /// draw triangles
    void drawTriangles(void);
    /// draw trajectories
    void drawTrajectories(void);
    /// resize
    void resizeWindow(int width, int height);
    /// snap
    void snap(int width, int height, const std::string& filenameDepth, const std::string& filenameRGB,
                   const std::string& filenameSegment, const std::string& filenameCloud);
    /// find mesh id
    size_t findMeshId(const std::string& filename);
    /// append camera pose
    void appendCameraPose(size_t poseNo, int parentNo, double pitch, double yaw, const walkers::Mat34& cameraPose, const std::string& filename);
    /// append objects list
    void appendObjectsList(const std::string& filename);
    /// get camera pose
    walkers::Mat34 getCameraPose(void);
    /// turn off visualization of objects
    void turnOffObjsVisualization(void);
    /// get distance to surface/object uder pixel
    double getDepthUnderPixel(int u, int v);
    /// visualize support polygon
    void visualizeSupportPolygon(size_t _poseNo, const walkers::Mat34& robotPose, const std::vector<double>& conf);

    bool snapFlag = false;
    bool resizeFlag = false;
    double animateTime = 0;
    size_t poseNo2save = 0;
    std::vector<walkers::Mat34> cameraPoses2save;
    std::string depth2save = "snap_depth.png";
    std::string rgb2save = "snap_rgb.png";
    std::string segment2save = "snap_segment.png";
    std::string cloud2save = "snap_cloud.pcd";
    std::vector<int> parentIds2save;
    std::vector<double> pitchs2save;
    std::vector<double> yaws2save;
    size_t offsetImageNo2save;

    /// get depth under pixel
    bool getPointUnderPixel = false;
    QPoint pointPixel;
    qglviewer::Vec pointUnderPixel;


    /// camera model
    std::unique_ptr<grabber::CameraModel> cameraModel;
    /// busy with snap
    bool busySnap = false;

};

#endif // QVISUALIZER_H_INCLUDED
