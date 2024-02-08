#include "Visualizer/Qvisualizer.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>
#include <QPoint>
#include <QGLWidget>
#include <QKeyEvent>
#include <Visualizer/Qvisualizer.h>
#include "Utilities/convexHull2D.h"
#include "Utilities/TimeRange.h"
//#include "Defs/opencv.h"

#include <filesystem>
#include <tinyxml2.h>

using namespace simulator;

TimeMeasurements time_inside_mutex;
TimeMeasurements timedisplaylist;
TimeMeasurements timemutex;

double time_mutex_mean, time_mutex_SD;
double timedisplaylist_mean, timedisplaylist_SD;

/// A single instance of Visualizer
QGLVisualizer::Ptr visualizer;

QGLVisualizer::QGLVisualizer(void) : robotStateTraj(-1,"",""), selectedLeg(0), randPosRRT(0,0,0),
    planedPoseIterator(0), visualizeClasses(false), visualizeCurvature(false), visualizeFootholdsCost(false), map3D(1024){
    simulationTime = 0;
    updateStateKinemModel = false;
}

/// Construction
QGLVisualizer::QGLVisualizer(std::string configFilename) : config(configFilename),
    robotStateTraj(-1,"",""), selectedLeg(0), randPosRRT(0,0,1e6), planedPoseIterator(0),
    visualizeClasses(false), visualizeCurvature(false), visualizeFootholdsCost(false), map3D(1024){
    simulationTime = 0;
    updateStateKinemModel = false;
}

/// Construction
QGLVisualizer::QGLVisualizer(std::string configFilename, std::string& robotConfig, std::string& _robotType,
                             std::string& _coldetType, std::string& _coldetConfig) : config(configFilename),
    robotStateTraj(-1,"",""), robotType(_robotType), selectedLeg(0), randPosRRT(0,0,1e6),
    planedPoseIterator(0), visualizeClasses(false), visualizeCurvature(false), visualizeFootholdsCost(false), map3D(1024){
    simulationTime = 0;
    updateStateKinemModel = false;

    refAnglesKinem.resize(18);
    //    for (auto& ang : refAnglesKinem)
    //        ang = 0;
    if ((robotType=="MessorII")||(robotType=="PhantomX")){
        refAnglesKinem = std::vector<double> ({0.7854,0.41888,-114*M_PI/180, 0.0,0.41888,-1.9897, -0.7854,0.41888,-1.9897, 0.7854,0.41888,-1.9897, 0.0,0.41888,-1.9897, -0.7854,0.41888,-1.9897});
    }
    else if ((robotType=="Anymal")||(robotType=="Anymal_C")||(robotType=="Galgo")){
        refAnglesKinem = std::vector<double> ({7*M_PI/180,33*M_PI/180,-60*M_PI/180, -7*M_PI/180,33*M_PI/180,-60*M_PI/180,
                                               7*M_PI/180,33*M_PI/180,-60*M_PI/180, -7*M_PI/180,33*M_PI/180,-60*M_PI/180});
    }
    else if (robotType=="Laikago"){
        refAnglesKinem = std::vector<double> ({0.12, 0.67, -1.3, -0.12, 0.67, -1.3, 0.12, 0.67, -1.3, -0.12, 0.67, -1.3});
    }

    ObjectsMesh objects3DS;
    //    addMesh(objects3DS);
    kinemRobotPose(0,3) = 0.0; kinemRobotPose(1,3) = 2.0; kinemRobotPose(2,3) = 0.75;
    initializeKinematicModel(kinemRobotPose, refAnglesKinem);
    //    initKinematicModel(objects3DS);
    //    init3DS(objects3DS);

    /// camera model
    cameraModel.reset(new grabber::CameraModel("../resources/cameraModels/KinectModel.xml"));
}

/// Destruction
QGLVisualizer::~QGLVisualizer(void) {
    //#ifdef DEBUG
    //std::cout << "QGLVisualizer destructor\n";
    //#endif
}

void QGLVisualizer::Config::load(std::string configFilename) {
    setlocale(LC_NUMERIC,"C");
    tinyxml2::XMLDocument config;
    std::string filename = "../../resources/" + configFilename;
    config.LoadFile(filename.c_str());
    if (config.ErrorID())
        std::cout << "unable to load Visualizer config file: " + filename << std::endl;

    tinyxml2::XMLElement * model = config.FirstChildElement( "VisualizerConfig" );
    model->FirstChildElement( "parameters" )->QueryBoolAttribute("verbose", &verbose);
    model->FirstChildElement( "parameters" )->QueryBoolAttribute("visualizeCollisions", &visualizeCollisions);
    model->FirstChildElement( "parameters" )->QueryBoolAttribute("visualizeFootholds", &visualizeFootholds);
    model->FirstChildElement( "parameters" )->QueryBoolAttribute("drawNormals", &drawNormals);
    model->FirstChildElement( "parameters" )->QueryBoolAttribute("drawPolyFitt", &drawPolyFitt);
    model->FirstChildElement( "parameters" )->QueryBoolAttribute("drawKinematicObjects", &drawKinematicObjects);
    model->FirstChildElement( "parameters" )->QueryBoolAttribute("displayTextStatus", &displayTextStatus);

    double rgba[4]={0,0,0,0};
    model->FirstChildElement( "background" )->QueryDoubleAttribute("red", &rgba[0]);
    model->FirstChildElement( "background" )->QueryDoubleAttribute("green", &rgba[1]);
    model->FirstChildElement( "background" )->QueryDoubleAttribute("blue", &rgba[2]);
    model->FirstChildElement( "background" )->QueryDoubleAttribute("alpha", &rgba[3]);
    backgroundColor.setRedF(rgba[0]); backgroundColor.setGreenF(rgba[1]);
    backgroundColor.setBlueF(rgba[2]); backgroundColor.setAlphaF(rgba[3]);

    /// point cloud
    model->FirstChildElement( "pointCloud" )->QueryDoubleAttribute("pointSize", &pointSize);
    /// line
    model->FirstChildElement( "line" )->QueryDoubleAttribute("lineWidth", &lineWidth);
    model->FirstChildElement( "line" )->QueryDoubleAttribute("red", &rgba[0]);
    model->FirstChildElement( "line" )->QueryDoubleAttribute("green", &rgba[1]);
    model->FirstChildElement( "line" )->QueryDoubleAttribute("blue", &rgba[2]);
    model->FirstChildElement( "line" )->QueryDoubleAttribute("alpha", &rgba[3]);
    lineColor.r = (uint8_t)(rgba[0]*255); lineColor.g = (uint8_t)(rgba[1]*255);
    lineColor.b = (uint8_t)(rgba[2]*255); lineColor.a = (uint8_t)(rgba[3]*255);

    model->FirstChildElement( "rrtBegin" )->QueryBoolAttribute("draw", &drawRRTbegin);
    model->FirstChildElement( "rrtBegin" )->QueryDoubleAttribute("red", &rgba[0]);
    model->FirstChildElement( "rrtBegin" )->QueryDoubleAttribute("green", &rgba[1]);
    model->FirstChildElement( "rrtBegin" )->QueryDoubleAttribute("blue", &rgba[2]);
    model->FirstChildElement( "rrtBegin" )->QueryDoubleAttribute("alpha", &rgba[3]);
    rrtBeginColor.setRedF(rgba[0]); rrtBeginColor.setGreenF(rgba[1]);
    rrtBeginColor.setBlueF(rgba[2]); rrtBeginColor.setAlphaF(rgba[3]);
    model->FirstChildElement( "rrtBegin" )->QueryDoubleAttribute("width", &rrtBeginWidth);

    model->FirstChildElement( "rrtFinish" )->QueryBoolAttribute("draw", &drawRRTfinish);
    model->FirstChildElement( "rrtFinish" )->QueryDoubleAttribute("red", &rgba[0]);
    model->FirstChildElement( "rrtFinish" )->QueryDoubleAttribute("green", &rgba[1]);
    model->FirstChildElement( "rrtFinish" )->QueryDoubleAttribute("blue", &rgba[2]);
    model->FirstChildElement( "rrtFinish" )->QueryDoubleAttribute("alpha", &rgba[3]);
    rrtFinishColor.setRedF(rgba[0]); rrtFinishColor.setGreenF(rgba[1]);
    rrtFinishColor.setBlueF(rgba[2]); rrtFinishColor.setAlphaF(rgba[3]);
    model->FirstChildElement( "rrtFinish" )->QueryDoubleAttribute("width", &rrtFinishWidth);

    drawObjects = true;
}

/// displa text status
void QGLVisualizer::setTextStatus(bool state){
    config.displayTextStatus = state;
}

/// clear rrt trees
void QGLVisualizer::clearRRTTrees(void){
    displayTrees.clear();
}

/// update robot trajectory
void QGLVisualizer::updateRobotTrajectories(const std::vector<walkers::Mat34>& _bodyPath, const std::vector<planner::Foot::Seq>& _feetPaths, size_t trajNo){
    /// robot path
    std::vector<planner::RobotState3D> robotPath;
    for (size_t pointNo=0;pointNo<_bodyPath.size();pointNo++){
        planner::RobotState3D robotState;
        robotState.robotPose.setPosition(Eigen::Vector3d(_bodyPath[pointNo](0,3),_bodyPath[pointNo](1,3),_bodyPath[pointNo](2,3)));
                                         robotState.feet = _feetPaths[pointNo];
                robotPath.push_back(robotState);
    }
    updateRobotPath(robotPath, trajNo, 3.0, 7.0, mapping::RGBA(255,0,0,255), mapping::RGBA(255,0,0,255),
                    3.0, 5.0, mapping::RGBA(0,255,0,255), mapping::RGBA(0,255,0,255));
}

void QGLVisualizer::updateRobotStateTraj(const RecorderRobotState& _robotStateTraj){
    mtxRobotStateTraj.lock();
    robotStateTraj = _robotStateTraj;
    currentRobotState = robotStateTraj.container.begin();
    mtxRobotStateTraj.unlock();
}

/// draw trees
void QGLVisualizer::drawTrees(void){
    glPushMatrix();
    displayTrees.draw();
    glPopMatrix();
}

/// draw 3D map
void QGLVisualizer::drawMapNDTOM() {
    glPushMatrix();
//    std::cout << "start draw ndt\n";
    displayMapNDTOM.draw();
//    std::cout << "end draw ndt\n";
    glPopMatrix();
}

/// draw Ellipsoids
void QGLVisualizer::drawEllipsoids() {
    glPushMatrix();
    displayEllipsoids.draw();
    glPopMatrix();
}

/// draw voxels
void QGLVisualizer::drawVoxels() {
    glPushMatrix();
    displayVoxels.draw();
    glPopMatrix();
}

/// draw point clouds
void QGLVisualizer::drawPointClouds() {
    glPushMatrix();
    displayPointClouds.draw();
    glPopMatrix();
}

/// draw lines
void QGLVisualizer::drawLines() {
    glPushMatrix();
    displayLines.draw();
    glPopMatrix();
}

/// draw lines
void QGLVisualizer::drawArrows() {
    glPushMatrix();
    displayArrows.draw();
    glPopMatrix();
}

/// draw B-Splines3D
void QGLVisualizer::drawBSplines3D() {
    glPushMatrix();
    displayBSplines3D.draw();
    glPopMatrix();
}

/// draw B-SplinesSE3
void QGLVisualizer::drawBSplinesSE3() {
    glPushMatrix();
    displayBSplinesSE3.draw();
    glPopMatrix();
}

/// draw 3D map
void QGLVisualizer::drawAxes() {
    glPushMatrix();
    displayAxis.draw();
    glPopMatrix();
}

/// draw Planes
void QGLVisualizer::drawPlanes() {
    glPushMatrix();
    displayPlanes.draw();
    glPopMatrix();
}

/// draw Triangles
void QGLVisualizer::drawTriangles() {
    glPushMatrix();
    displayTriangles.draw();
    glPopMatrix();
}

/// draw Trajectories
void QGLVisualizer::drawTrajectories() {
    glPushMatrix();
    displayTrajectories.draw();
    glPopMatrix();
    for (auto& trajectory : displayTrajectories.displays){
        trajectory.setCurrentTime(realTime.now());
    }
}

/// find mesh id
size_t QGLVisualizer::findMeshId(const std::string& filename){
    displayMeshes.mutex.lock();
    int meshId = 0;
    int id=0;
    for (const auto& mesh : displayMeshes.displays){
        if (mesh.filename==filename){
            meshId = id;
        }
        id++;
    }
    displayMeshes.mutex.unlock();
    return meshId;
}

/// add render objects (ids of added objects are in the structure RenderObjects)
void QGLVisualizer::addRenderObjects(std::vector<simulator::RenderObject>& _renderObjects)
{
    mtxObjects.lock();
    for (auto& renderObject : _renderObjects){
        renderObjects.push_back(renderObject);
        renderObject.id = renderObjects.size()-1;
        if (renderObject.type == RenderObjectType::MODEL3D){
            ObjectsMesh objectMesh;
            objectMesh.scale[0] = renderObject.scaleX;
            objectMesh.scale[1] = renderObject.scaleY;
            objectMesh.scale[2] = renderObject.scaleZ;
            objectMesh.filenames.push_back(renderObject.meshFilename);
            addMesh(objectMesh);
            renderObject.meshId = findMeshId(renderObject.meshFilename);
            renderObjects.back().meshId = findMeshId(renderObject.meshFilename);
        }
        renderObjects.back().id = renderObjects.size()-1;
    }
    mtxObjects.unlock();
}

/// update render objects
void QGLVisualizer::updateRenderObjects(const std::vector<simulator::RenderObject>& _renderObjects)
{
    mtxObjects.lock();
    for (const auto& renderObject : _renderObjects){
        if (renderObject.id<renderObjects.size()){
            renderObjects[renderObject.id] = renderObject;
        }
        else{
            std::cout << "unregistered render object id: " << renderObject.id << "\n";
        }
    }
    mtxObjects.unlock();
}

/// draw kinematic model
void QGLVisualizer::initializeKinematicModel(const walkers::Mat34& pose, const std::vector<double>& configuration){

}

/// draw kinematic model
void QGLVisualizer::updateKinematicModel(const walkers::Mat34& pose, const std::vector<double>& configuration){

}

//void QGLVisualizer::updateKinematicObjects(const std::vector<simulator::RenderObject>& kinemObjects) {
//    mtxObjects.lock();
//    kinematicObj = kinemObjects;
//    for (auto& ko : kinematicObj){
//        ko.color[0]=0.7; ko.color[1]=0.7; ko.color[2]=0.5; ko.color[3]=0.8;
//    }
//    mtxObjects.unlock();
//}

void QGLVisualizer::initGround(const simulator::RenderObjectHeightmap& ground) {
    mtxObjects.lock();
    groundObj = ground;
    mtxObjects.unlock();
    groundInit = true;
}

/// update color of the ground
void QGLVisualizer::updateGroundColor(const std::vector<std::vector<std::vector<double>>>& groundColors) {
    mtxObjects.lock();
    groundObj.heightmapColors = groundColors;
    mtxObjects.unlock();
    groundInit = true;
}

void QGLVisualizer::setShadowsVisible(bool shadowVisible) {
    shadowFlag = shadowVisible;
}

/// close window
void QGLVisualizer::closeWindow(void){
    this->close();
}

/// turn off visualization of objects
void QGLVisualizer::turnOffObjsVisualization(void){
    if (!(snapFlag&&poseNo2save==1)){
        for (auto& mesh : displayMeshes.displays){
            mesh.isVisible = true;
        }
    }
    else{
        for (const auto& renderObject : renderObjects){
            if (renderObject.name != "table"){
                displayMeshes.displays[renderObject.meshId].isVisible = false;
            }
        }
    }
}

/// draw objects
void QGLVisualizer::draw(){
    // ModelView represents the world coordinate system at the beginning of draw()
    glClearStencil(0); // this is the default value
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

    /* Enable stencil operations */
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

//    renderGround();
//    if (updateStateKinemModel){
//        updateKinematicModel(kinemRobotPose,refAnglesKinem);
//        updateStateKinemModel = false;
//    }

//    turnOffObjsVisualization();
    glPushMatrix();
    renderScene();

    glPopMatrix();
//    renderNormals();
//    drawTrees();

    if (config.displayTextStatus)
        drawTextStatus();
//    drawRegisteredRobotState();
    drawRobotPlannedPaths();
//    if (config.visualizeCollisions)
//        visualizeCollisions();
    //    if (config.visualizeFootholds)// TODO does not work if terrain or robot is not loaded
    //        visualizeFootholds();

//    drawRandPosRRT();
    // draw ellipsoids 3D map
    drawMapNDTOM();
    // draw bounding ellipsoids
    drawEllipsoids();

    // draw Octomap
    drawVoxels();
    drawPointClouds();
//    if (config.drawPolyFitt){
//        drawPolynomialFitting();
//    }

    drawLines();
    drawArrows();
    drawBSplines3D();
    drawBSplinesSE3();
    drawAxes();

    drawPlanes();
    drawTriangles();
    drawTrajectories();

    if (getPointUnderPixel){
        bool foundPixel;
        pointUnderPixel = camera()->pointUnderPixel(pointPixel, foundPixel);
        getPointUnderPixel = false;
    }
    if (snapFlag){
        snap(640, 480, depth2save, rgb2save, segment2save, cloud2save);
        snapFlag = false;
    }
    if (resizeFlag){
        resizeWindow(640,480);
        snapFlag = true;
        resizeFlag = false;
    }

    GLenum err;
    while((err = glGetError()) != GL_NO_ERROR) {
        std::cout << "OpenGL error: " << err << "\n";
    }

    if (poseNo2save < cameraPoses2save.size()&&!snapFlag){
        setCameraPose(cameraPoses2save[poseNo2save]);
        std::ostringstream ss;
        ss << std::setw(5) << std::setfill('0') << poseNo2save+offsetImageNo2save;
        std::string number(ss.str());
        depth2save = "dataset/depth" + number + ".png";
        rgb2save = "dataset/rgb" + number + ".png";
        segment2save = "dataset/segment" + number + ".png";
        cloud2save = "";//dataset/cloud" + number + ".pcd";
        int parentNo = (int)parentIds2save[poseNo2save];
        double pitch = pitchs2save[poseNo2save];
        double yaw = yaws2save[poseNo2save];
        appendCameraPose(poseNo2save+offsetImageNo2save, parentNo, pitch, yaw, cameraPoses2save[poseNo2save],"dataset/camPoses.txt");
        std::string objectsList2save = "dataset/objects" + number + ".dat";
        appendObjectsList(objectsList2save);
        resizeFlag=true;
        poseNo2save++;
        std::cout << "poseNo2save " << poseNo2save << "\n";
    }
}

/// is busy with snap
bool QGLVisualizer::isBusyWithSnap(void){
    return busySnap;
}

/// visualize random position RRT
void QGLVisualizer::drawRandPosRRT(void){
    glPushMatrix();
    mtxRandPosRRT.lock();
    glTranslated(randPosRRT.x(), randPosRRT.y(), randPosRRT.z());
    mtxRandPosRRT.unlock();
    glColor3d(0.5,1,0.5);
    glutSolidSphere(0.08, 10, 10);
    glPopMatrix();
}

/// visualize footholds
void QGLVisualizer::visualizeFootholds(void){

}

/// visualize collisions
void QGLVisualizer::visualizeCollisions(void){

}

/// draw robot planned paths
void QGLVisualizer::drawRobotPlannedPaths(){
    glPushMatrix();
    displayPaths.draw();
    glPopMatrix();

    glPushMatrix();
    displayRobotPaths.draw();
    glPopMatrix();
}

/// draw text status
void QGLVisualizer::drawTextStatus(){
    int minutesFromStart = (int)realTime.stop()/60000;
    int secondsFromStart = ((int)realTime.stop()/1000)%60;
    std::string realTimeTxt = "Real time: " + std::string( 2-std::to_string(minutesFromStart).length(), '0').append(std::to_string(minutesFromStart)) + ":" + std::string( 2-std::to_string(secondsFromStart).length(), '0').append(std::to_string(secondsFromStart));
    //this->setD
    drawText(10, this->height()-4,QString(realTimeTxt.c_str()));
    mtxSimTime.lock();
    minutesFromStart = (int)simulationTime/60000;
    secondsFromStart = ((int)simulationTime/1000)%60;
    std::string simTimeTxt = "Sim. time: " + std::string( 2-std::to_string(minutesFromStart).length(), '0').append(std::to_string(minutesFromStart)) + ":" + std::string( 2-std::to_string(secondsFromStart).length(), '0').append(std::to_string(secondsFromStart));
    //this->setD
    drawText(10, this->height()-15,QString(simTimeTxt.c_str()));
    mtxSimTime.unlock();
}

void QGLVisualizer::updateSimTime(int simTime){
    mtxSimTime.lock();
    simulationTime = simTime;
    mtxSimTime.unlock();
}

///update robot mesh pose and map mesh pose
void QGLVisualizer::updateRobotMapPose(const std::vector<walkers::Mat34>& _robotAndMapPose){
    robotAndMapPose = _robotAndMapPose;
}

void QGLVisualizer::renderScene() {
    //    if(shadowFlag) {
    //        for(unsigned int i=0;i<dynamicObj.size();++i) {
    //            if(dynamicObj[i].type != RenderObjectType::LINE) {
    //                const double shadowMat[]={ 1,0,0,0, 0,1,0,0, 0,-1,0,0, 0,0,0,1 };
    //                if(groundObj.type == RenderObjectType::HEIGHTFIELD) {
    //                    uint k = (uint)round((dynamicObj[i].mat(0,3)-groundObj.x)/groundObj.scaleX);
    //                    uint j = (uint)round((dynamicObj[i].mat(1,3)-groundObj.y)/groundObj.scaleY);
    //                    if(k<groundObj.heightmap.size()) {
    //                        if(j<groundObj.heightmap[k].size()) {
    //                            double z = groundObj.z+groundObj.heightmap[groundObj.heightmap.size()-j][k]*groundObj.scaleZ;
    //                            glDisable(GL_DEPTH_TEST);
    //                            glDisable(GL_LIGHTING);

    //                            glPushMatrix();
    //                            glColor4d(0.0,0.0,0.0,1.0);
    //                            glTranslated(0.0,0.0,z);
    //                            glMultMatrixd(shadowMat);
    //                            glTranslated(0.0,0.0,-z);
    //                            renderObject(dynamicObj[i]);
    //                            glPopMatrix();

    //                            glEnable(GL_DEPTH_TEST);
    //                            glEnable(GL_LIGHTING);
    //                        }
    //                    }
    //                }
    //                else {
    //                    glDisable(GL_DEPTH_TEST);
    //                    glDisable(GL_LIGHTING);

    //                    glPushMatrix();
    //                    glColor4d(0.0,0.0,0.0,1.0);
    //                    glMultMatrixd(shadowMat);
    //                    renderObject(dynamicObj[i]);
    //                    glPopMatrix();

    //                    glEnable(GL_DEPTH_TEST);
    //                    glEnable(GL_LIGHTING);
    //                }
    //            }
    //        }
    //    }
    mtxObjects.lock();
    if (config.drawObjects){
        for(unsigned int i=0;i<renderObjects.size();++i) {
            glStencilFunc(GL_ALWAYS, i, -1);
            glColor4d(renderObjects[i].color[0],renderObjects[i].color[1],renderObjects[i].color[2],renderObjects[i].color[3]);
            renderObject(renderObjects[i]);
        }
    }
    mtxObjects.unlock();
    //    if (config.drawKinematicObjects){
    //        for(unsigned int i=0;i<kinematicObj.size();++i) {
    //            glColor4d(kinematicObj[i].color[0],kinematicObj[i].color[1],kinematicObj[i].color[2],kinematicObj[i].color[3]);
    //            renderObject(kinematicObj[i]);
    //        }
    //    }
}

void QGLVisualizer::renderGround() {
    if(groundInit) {
        std::cout << "updateGround\n";
        displayElevationMeshes.displays.resize(4);
        displayElevationMeshes.displays[0].createDisplayList(*(elevationMap.get()),groundObj,false,false,false);
        displayElevationMeshes.displays[1].createDisplayList(*(elevationMap.get()),groundObj,true,false,false);
        displayElevationMeshes.displays[2].createDisplayList(*(elevationMap.get()),groundObj,false,true,false);
        displayElevationMeshes.displays[3].createDisplayList(*(elevationMap.get()),groundObj,false,false,true);
        groundInit = false;
        normalsInit = true;
    }
    if (displayElevationMeshes.displays.size()>3){
        glPushMatrix();
//        glDisable(GL_TEXTURE_2D);
//        GLfloat diffuseColor[] = {0.4, 0.4, 0.4, 1.0};
//        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseColor);

//        GLfloat specular_color[4] = {0.2, 0.2, 0.2, 1.0};
//        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular_color);

//        float reflectColor[] = { 0.8f, 0.8f, 0.8f, 1.0f };
//        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, reflectColor);
//        GLfloat emissiveLight[] = { 0.1f, 0.1f, 0.1f, 1.0f };
//        glMaterialfv(GL_FRONT, GL_EMISSION, emissiveLight);

        if (visualizeClasses){
            displayElevationMeshes.displays[1].draw();
        }
        else {
            if (visualizeCurvature){
                displayElevationMeshes.displays[2].draw();
            }
            else {
                if (visualizeFootholdsCost){
                    displayElevationMeshes.displays[3].draw();
                }
                else{
                    displayElevationMeshes.displays[0].draw();
                }
            }
        }
        glPopMatrix();
    }
    if (config.drawNormals){
        //        glCallList(displayTerrainNormals.listId);
    }
}

void QGLVisualizer::renderNormals() {
    if(normalsInit) {
        //        displayTerrainNormals.mutex.lock();
        //        displayTerrainNormals.listId = glGenLists(1);
        //        glNewList(displayTerrainNormals.listId, GL_COMPILE);
        //            drawTerrainNormals();
        //        glEndList();
        //        displayTerrainNormals.mutex.unlock();
        normalsInit = false;
    }
    if (config.drawNormals){
        //        glCallList(displayTerrainNormals.listId);
    }
}

/// draw normals to the surface
void QGLVisualizer::drawTerrainNormals(){
    std::vector<std::vector<walkers::Vec3>> normals = elevationMap->getNormals();
    for (size_t rowNo=1;rowNo<normals.size()-1;rowNo++){
        for (size_t colNo=1;colNo<normals[rowNo].size()-1;colNo++){
            walkers::Vec3 normal = normals[rowNo][colNo];
            double cellx = elevationMap->toRealX((int)colNo);
            double celly = elevationMap->toRealY((int)rowNo);
            double cellz = elevationMap->get((int)colNo, (int)rowNo);
            glLineWidth(2.5);
            glColor3f(0.5, 0.5, 0.5);
            glBegin(GL_LINES);
            glVertex3f((GLfloat)cellx, (GLfloat)celly, (GLfloat)cellz);
            glVertex3f((GLfloat)(cellx+normal.x()*0.1), (GLfloat)(celly+normal.y()*0.1), (GLfloat)(cellz+normal.z()*0.1));
            glEnd();
        }
    }
}
/// draw objects
void QGLVisualizer::animate(){
    //    qglviewer::Vec position;
    //    animateTime = animateTime+0.01;
    //    double radius = 3.5;
    //    position.setValue(float(radius * sin(animateTime)), float(radius * cos(animateTime)), 1.75);
    //    camera()->setPosition(position);
    //    qglviewer::Vec view(float(-radius * sin(animateTime)), float(-radius * cos(animateTime)), -1.75);
    //    camera()->setOrientation(0.0,-1.57);
    //    camera()->setViewDirection(view);

    //    qglviewer::Vec position;
    //    animateTime = animateTime+0.01;
    //    double radius = 0.075;
    //    position.setValue(float(radius * sin(animateTime)), float(-0.075), float(radius * cos(animateTime)));
    //    camera()->setPosition(position);
    //    qglviewer::Vec view(float(-radius * sin(animateTime)), float(0.075), float(-radius * cos(animateTime)));
    //    camera()->setOrientation(1.57,0);
    //    camera()->setViewDirection(view);
}

/// initialize visualizer
void QGLVisualizer::init(){
    // Light setup
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 258.0);
    GLfloat specular_color[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  specular_color);

    //Set global ambient light
    GLfloat black[] = {0.1f, 0.1f, 0.1f, 1.0f};
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

    glEnable(GL_AUTO_NORMAL);
    glEnable(GL_NORMALIZE);
    //    glEnable(GL_DEPTH_TEST);
    //    glEnable(GL_TEXTURE_2D);
    // Restore previous viewer state.

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // Light0 is the default ambient light
    glEnable(GL_LIGHT0);

    // Light1 is a spot light
//    glEnable(GL_LIGHT1);
//    const GLfloat light_ambient[4] = {0.5f, 0.5f, 0.5f, 1.0};
//    const GLfloat light_diffuse[4] = {0.4f, 0.4f, 0.4f, 1.0};
//    const GLfloat light_specular[4] = {0.2f, 0.2f, 0.2f, 1.0};
//    const GLfloat light1_position[3] = {0.0, 0.0, 2.0};

//    glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
//    glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, 3.0);
//    glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, 20.0);
//    glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 0.5);
//    glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 1.0);
//    glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 1.5);
//    glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
//    glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular);
//    glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);

//    glEnable(GL_LIGHT2);
//    const GLfloat light2_position[3] = {-4.0, 0.0, 2.0};

//    glLightfv(GL_LIGHT2, GL_POSITION, light2_position);
//    glLightf(GL_LIGHT2, GL_SPOT_EXPONENT, 3.0);
//    glLightf(GL_LIGHT2, GL_SPOT_CUTOFF, 20.0);
//    glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION, 0.5);
//    glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION, 1.0);
//    glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, 1.5);
//    glLightfv(GL_LIGHT2, GL_AMBIENT, light_ambient);
//    glLightfv(GL_LIGHT2, GL_SPECULAR, light_specular);
//    glLightfv(GL_LIGHT2, GL_DIFFUSE, light_diffuse);

//    glEnable(GL_LIGHT3);
//    const GLfloat light3_position[3] = {4.0, 0.0, 2.0};

//    glLightfv(GL_LIGHT3, GL_POSITION, light3_position);
//    glLightf(GL_LIGHT3, GL_SPOT_EXPONENT, 3.0);
//    glLightf(GL_LIGHT3, GL_SPOT_CUTOFF, 20.0);
//    glLightf(GL_LIGHT3, GL_CONSTANT_ATTENUATION, 0.5);
//    glLightf(GL_LIGHT3, GL_LINEAR_ATTENUATION, 1.0);
//    glLightf(GL_LIGHT3, GL_QUADRATIC_ATTENUATION, 1.5);
//    glLightfv(GL_LIGHT3, GL_AMBIENT, light_ambient);
//    glLightfv(GL_LIGHT3, GL_SPECULAR, light_specular);
//    glLightfv(GL_LIGHT3, GL_DIFFUSE, light_diffuse);

//    glEnable(GL_LIGHT4);
//    const GLfloat light4_position[3] = {0.0, -4.0, 2.0};

//    glLightfv(GL_LIGHT4, GL_POSITION, light4_position);
//    glLightf(GL_LIGHT4, GL_SPOT_EXPONENT, 3.0);
//    glLightf(GL_LIGHT4, GL_SPOT_CUTOFF, 20.0);
//    glLightf(GL_LIGHT4, GL_CONSTANT_ATTENUATION, 0.5);
//    glLightf(GL_LIGHT4, GL_LINEAR_ATTENUATION, 1.0);
//    glLightf(GL_LIGHT4, GL_QUADRATIC_ATTENUATION, 1.5);
//    glLightfv(GL_LIGHT4, GL_AMBIENT, light_ambient);
//    glLightfv(GL_LIGHT4, GL_SPECULAR, light_specular);
//    glLightfv(GL_LIGHT4, GL_DIFFUSE, light_diffuse);

//    glEnable(GL_LIGHT5);
//    const GLfloat light5_position[3] = {0.0, 4.0, 2.0};

//    glLightfv(GL_LIGHT5, GL_POSITION, light5_position);
//    glLightf(GL_LIGHT5, GL_SPOT_EXPONENT, 3.0);
//    glLightf(GL_LIGHT5, GL_SPOT_CUTOFF, 20.0);
//    glLightf(GL_LIGHT5, GL_CONSTANT_ATTENUATION, 0.5);
//    glLightf(GL_LIGHT5, GL_LINEAR_ATTENUATION, 1.0);
//    glLightf(GL_LIGHT5, GL_QUADRATIC_ATTENUATION, 1.5);
//    glLightfv(GL_LIGHT5, GL_AMBIENT, light_ambient);
//    glLightfv(GL_LIGHT5, GL_SPECULAR, light_specular);
//    glLightfv(GL_LIGHT5, GL_DIFFUSE, light_diffuse);

//    // Light1 is a spot light
//    glEnable(GL_LIGHT6);
//    const GLfloat light6_position[3] = {0.0f, 0.0f, 0.2f};

//    glLightfv(GL_LIGHT6, GL_POSITION, light6_position);
//    glLightf(GL_LIGHT6, GL_SPOT_EXPONENT, 3.0);
//    glLightf(GL_LIGHT6, GL_SPOT_CUTOFF, 20.0);
//    glLightf(GL_LIGHT6, GL_CONSTANT_ATTENUATION, 0.5);
//    glLightf(GL_LIGHT6, GL_LINEAR_ATTENUATION, 1.0);
//    glLightf(GL_LIGHT6, GL_QUADRATIC_ATTENUATION, 1.5);
//    glLightfv(GL_LIGHT6, GL_AMBIENT, light_ambient);
//    glLightfv(GL_LIGHT6, GL_SPECULAR, light_specular);
//    glLightfv(GL_LIGHT6, GL_DIFFUSE, light_diffuse);
    // Light2 is a classical directional light
//    glEnable(GL_LIGHT2);
//    const GLfloat light_ambient2[4] = {0.2f, 0.2f, 0.2, 1.0};
//    const GLfloat light_diffuse2[4] = {0.4f, 0.4f, 0.4, 1.0};
//    const GLfloat light_specular2[4] = {0.2, 0.2, 0.2, 1.0};

//    glLightfv(GL_LIGHT2, GL_AMBIENT, light_ambient2);
//    glLightfv(GL_LIGHT2, GL_SPECULAR, light_specular2);
//    glLightfv(GL_LIGHT2, GL_DIFFUSE, light_diffuse2);

    restoreStateFromFile();

    camera()->setZNearCoefficient(0.00001f);
    camera()->setZClippingCoefficient(100.0);

    setBackgroundColor(config.backgroundColor);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Opens help window
    help();

    startAnimation();
}

/// load map
void QGLVisualizer::loadMap(std::string configFilename){
    elevationMap.reset(new ElevationMap(configFilename));
    simulator::RenderObjectHeightmap heightfield;
    elevationMap->createHeightField(heightfield);
    elevationMap->normals2surface();
    elevationMap->curvatureSurface();
    initGround(heightfield);
    groundInit = true;
    if (config.visualizeCollisions){
        ObjectsMesh object3dsMap;
        elevationMap->getMeshModel(object3dsMap);
        object3dsMap.filenames.push_back("terrain.map");
//        collisionChecker->initializeTerrain(object3dsMap);
        //        addMesh(object3dsMap);
    }
}

/// capture the key
void QGLVisualizer::keyPressEvent(QKeyEvent* key){
    // Defines the Alt+R shortcut.
    //    if ((key->key() == Qt::Key_R) && (key->modifiers() == Qt::AltModifier)) {
    if ((key->key() == Qt::Key_A)) {
//        if (selectedLeg!=robotMat->getLegsNo()-1)
            selectedLeg++;
    }
    if ((key->key() == Qt::Key_Z)) {
        if (selectedLeg>0)
            selectedLeg--;
    }
    if ((key->key() == Qt::Key_E)) {
        refAnglesKinem[selectedLeg*3+0]+=1.0*M_PI/180.0;
        std::cout << "refAnglesKinem[selectedLeg*3+0] " << refAnglesKinem[selectedLeg*3+0] << "\n";
        updateStateKinemModel = true;
    }
    if ((key->key() == Qt::Key_D)) {
        refAnglesKinem[selectedLeg*3+0]-=1.0*M_PI/180.0;
        std::cout << "refAnglesKinem[selectedLeg*3+0] " << refAnglesKinem[selectedLeg*3+0] << "\n";
        updateStateKinemModel = true;
    }
    if ((key->key() == Qt::Key_R)) {
        refAnglesKinem[selectedLeg*3+1]+=1.0*M_PI/180.0;
        std::cout << "refAnglesKinem[selectedLeg*3+1] " << refAnglesKinem[selectedLeg*3+1] << "\n";
        updateStateKinemModel = true;
    }
    if ((key->key() == Qt::Key_F)) {
        refAnglesKinem[selectedLeg*3+1]-=1.0*M_PI/180.0;
        std::cout << "refAnglesKinem[selectedLeg*3+1] " << refAnglesKinem[selectedLeg*3+1] << "\n";
        updateStateKinemModel = true;
    }
    if ((key->key() == Qt::Key_T)) {
        refAnglesKinem[selectedLeg*3+2]+=1.0*M_PI/180.0;
        std::cout << "refAnglesKinem[selectedLeg*3+2] " << refAnglesKinem[selectedLeg*3+2] << "\n";
        updateStateKinemModel = true;
    }
    if ((key->key() == Qt::Key_G)) {
        refAnglesKinem[selectedLeg*3+2]-=1.0*M_PI/180.0;
        std::cout << "refAnglesKinem[selectedLeg*3+2] " << refAnglesKinem[selectedLeg*3+2] << "\n";
        updateStateKinemModel = true;
    }
    if ((key->key() == Qt::Key_B)) {
        planedPoseIterator = 0;
        showPlannedPose(planedPoseIterator);
    }
    if ((key->key() == Qt::Key_N)) {
        if (planedPoseIterator>0)
            planedPoseIterator--;
        showPlannedPose(planedPoseIterator);
    }
    if ((key->key() == Qt::Key_M)) {
        if (planedPoseIterator<plannedPath.size()-1)
            planedPoseIterator++;
        showPlannedPose(planedPoseIterator);
    }
    if ((key->key() == Qt::Key_V)) {
        if (visualizeClasses)
            visualizeClasses=false;
        else
            visualizeClasses=true;
    }
    if ((key->key() == Qt::Key_C)) {
        if (visualizeCurvature)
            visualizeCurvature=false;
        else
            visualizeCurvature=true;
    }
    if ((key->key() == Qt::Key_P)) {
        if (visualizeFootholdsCost)
            visualizeFootholdsCost=false;
        else
            visualizeFootholdsCost=true;
    }
    if ((key->key() == Qt::Key_I)) {
        config.drawObjects = (config.drawObjects) ? false : true;
    }
    if (key->key() == Qt::Key_O){

    }
    if (key->key() == Qt::Key_J){
        resizeFlag = true;
    }
    if (key->key() == Qt::Key_Right){
        kinemRobotPose(2,3)+=0.05;
        updateStateKinemModel = true;
    }
    if (key->key() == Qt::Key_Left){
        kinemRobotPose(2,3)-=0.05;
        updateStateKinemModel = true;
    }
    if (key->key() == Qt::Key_Up){
        kinemRobotPose(1,3)+=0.05;
        updateStateKinemModel = true;
    }
    if (key->key() == Qt::Key_Down){
        kinemRobotPose(1,3)-=0.05;
        updateStateKinemModel = true;
    }
    if (key->key() == Qt::Key_1){
        walkers::Vec3 rot = walkers::fromRotationMat(kinemRobotPose);
        rot.x()+=0.1;
        kinemRobotPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rot).matrix().block<3,3>(0,0);
        updateStateKinemModel = true;
    }
    if (key->key() == Qt::Key_2){
        walkers::Vec3 rot = walkers::fromRotationMat(kinemRobotPose);
        rot.x()-=0.1;
        kinemRobotPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rot).matrix().block<3,3>(0,0);
        updateStateKinemModel = true;
    }
    if (key->key() == Qt::Key_3){
        walkers::Vec3 rot = walkers::fromRotationMat(kinemRobotPose);
        rot.y()+=0.1;
        kinemRobotPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rot).matrix().block<3,3>(0,0);
        updateStateKinemModel = true;
    }
    if (key->key() == Qt::Key_4){
        walkers::Vec3 rot = walkers::fromRotationMat(kinemRobotPose);
        rot.y()-=0.1;
        kinemRobotPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rot).matrix().block<3,3>(0,0);
        updateStateKinemModel = true;
    }
    else
        QGLViewer::keyPressEvent(key);
}

/// visualize support polygon
void QGLVisualizer::visualizeSupportPolygon(size_t _poseNo, const walkers::Mat34& robotPose, const std::vector<double>& conf){

}
///show planned pose
void QGLVisualizer::showPlannedPose(size_t _poseNo){

}

/// generate help string
std::string QGLVisualizer::help() const{
    std::string text("S i m p l e V i e w e r");
    text += "Use the mouse to move the camera around the object. ";
    text += "You can respectively revolve around, zoom and translate with the three mouse buttons. ";
    text += "Left and middle buttons pressed together rotate around the camera view direction axis<br><br>";
    text += "Pressing <b>Alt</b> and one of the function keys (<b>F1</b>..<b>F12</b>) defines a camera keyFrame. ";
    text += "Simply press the function key again to restore it. Several keyFrames define a ";
    text += "camera path. Paths are saved when you quit the application and restored at next start.<br><br>";
    text += "Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, ";
    text += "<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save a snapshot. ";
    text += "See the <b>Keyboard</b> tab in this window for a complete shortcut list.<br><br>";
    text += "Double clicks automates single click actions: A left button double click aligns the closer axis with the camera (if close enough). ";
    text += "A middle button double click fits the zoom of the camera and the right button re-centers the scene.<br><br>";
    text += "A left button double click while holding right button pressed defines the camera <i>Revolve Around Point</i>. ";
    text += "See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>";
    text += "Press <b>Escape</b> to exit the viewer.";
    return text;
}

/// get class color
std::array<double,3> QGLVisualizer::getClassColor(int classId){
    std::array<double,3> colorClass;
    if (classId==0)
        colorClass = std::array<double,3>{{1.0,0,0}};
    else if (classId==1)
        colorClass = std::array<double,3>{{0.0,1.0,0}};
    else if (classId==2)
        colorClass = std::array<double,3>{{0.0,0,1.0}};
    else if (classId==3)
        colorClass = std::array<double,3>{{1.0,1.0,0}};
    else if (classId==4)
        colorClass = std::array<double,3>{{0.0,1.0,1.0}};
    else if (classId==5)
        colorClass = std::array<double,3>{{1.0,0,1.0}};
    else if (classId==6)
        colorClass = std::array<double,3>{{0.5,0.5,0.5}};
    else {
        colorClass = std::array<double,3>{{0,0,0}};
    }
    return colorClass;
}

/// set current robot pose to the last added trajectory
void QGLVisualizer::setCurrentRobotPose(void){
    currentRobotPose = recorders6D.back().container.begin();
}

///draw registered robot state
void QGLVisualizer::drawRegisteredRobotState(void){
    if (robotStateTraj.container.size()>0&&currentRobotState->first/1000000<realTime.now()){
        if (currentRobotState!=robotStateTraj.container.end()){
            std::cout << "draw registerd robot state\n";
            updateKinematicModel(currentRobotPose->second,currentRobotState->second.currentValues);
            while (currentRobotState->first/1000000<realTime.now()&&currentRobotState!=robotStateTraj.container.end()){
                currentRobotState++;
                currentRobotPose++;
            }
        }
    }
}

void QGLVisualizer::renderObject(RenderObject& object)
{
    double GLmat[16]={object.mat(0,0), object.mat(1,0), object.mat(2,0), object.mat(3,0),
                      object.mat(0,1), object.mat(1,1), object.mat(2,1), object.mat(3,1),
                      object.mat(0,2), object.mat(1,2), object.mat(2,2), object.mat(3,2),
                      object.mat(0,3), object.mat(1,3), object.mat(2,3), object.mat(3,3)};
    if(object.type == RenderObjectType::BOX)
    {
        glPushMatrix();
        glMultMatrixd(GLmat);
        glScaled(object.x, object.y, object.z);
        glutSolidCube(2.0);
        glPopMatrix();
    }
    else if(object.type == RenderObjectType::SPHERE)
    {
        glPushMatrix();
        glMultMatrixd(GLmat);
        glutSolidSphere(object.x, 10, 10);
        glPopMatrix();
    }
    else if(object.type == RenderObjectType::LINE)
    {
        glPushMatrix();
        glLineWidth(3.5);
        glBegin(GL_LINES);
        glVertex3d(object.x, object.y, object.z);
        glVertex3d(object.scaleX, object.scaleY, object.scaleZ);
        glEnd();
        glPopMatrix();
    }
    else if(object.type == RenderObjectType::MODEL3D)
    {
        glPushMatrix();
        glMultMatrixd(GLmat);
        glScaled(object.scaleX,object.scaleY,object.scaleZ);
        //        std::cout << "render object id " << object.id << "\n";
        //        std::cout << "render object mesh id " << object.meshId << "\n";
        displayMeshes.mutex.lock();
        displayMeshes.displays[object.meshId].draw();
        displayMeshes.mutex.unlock();
        glPopMatrix();
    }
    else
    {

    }
}

/// RRT - update random position
void QGLVisualizer::updateRandomPosition(const walkers::Vec3& randPos){
    mtxRandPosRRT.lock();
    randPosRRT = randPos;
    mtxRandPosRRT.unlock();
}

/// add mapNDTOM
void QGLVisualizer::addMapNDTOM(const Octree<mapping::Voxel>& map, const std::unordered_map<std::string, Eigen::Vector3i> indexes){
    std::cout << "add ndt\n";
    displayMapNDTOM.mutex.lock();
    displayMapNDTOM.displays.resize(displayMapNDTOM.displays.size()+1);
    displayMapNDTOM.displays.back().createDisplayList(map, indexes);
    displayMapNDTOM.mutex.unlock();
}

/// update mapNDTOM
void QGLVisualizer::updateMapNDTOM(const Octree<mapping::Voxel>& map, const std::unordered_map<std::string, Eigen::Vector3i> indexes,
                                   size_t mapNo){

    //std::chrono::steady_clock::time_point begin_mutex = std::chrono::steady_clock::now();

    //displayMapNDTOM.mutex.lock(); // do skomentowania
    if (mapNo+1>displayMapNDTOM.displays.size()){

        displayMapNDTOM.mutex.lock(); // do odkomentowania
        displayMapNDTOM.displays.resize(mapNo+1);
        displayMapNDTOM.mutex.unlock(); // do odkomentowania

    }

    //std::chrono::steady_clock::time_point end_mutex = std::chrono::steady_clock::now();

    //timemutex.addMeasurement(end_mutex-begin_mutex);

    //time_mutex_mean=timemutex.calculateMean();
    //time_mutex_SD=timemutex.calculateStandardDeviation();

    //std::chrono::steady_clock::time_point begin_display = std::chrono::steady_clock::now();

    displayMapNDTOM.displays[mapNo].createDisplayList(map, indexes);
    //displayMapNDTOM.mutex.unlock(); // do skomentowania

    //std::chrono::steady_clock::time_point end_display = std::chrono::steady_clock::now();

    //timedisplaylist.addMeasurement(end_display-begin_display);
    //timedisplaylist_mean=timedisplaylist.calculateMean();
    //timedisplaylist_SD=timedisplaylist.calculateStandardDeviation();

}

/// add path
void QGLVisualizer::addPath(const std::vector<planner::PoseSE3>& path, double _lineWidth, double _pointSize,
                            const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    displayPaths.mutex.lock();
    displayPaths.displays.resize(displayPaths.displays.size()+1);
    displayPaths.displays.back().createDisplayList(path, _lineWidth, _pointSize, _lineColor, _pointColor);
    displayPaths.mutex.unlock();
}

/// update path
void QGLVisualizer::updatePath(const std::vector<planner::PoseSE3>& path, size_t pathNo, double _lineWidth,
                               double _pointSize, const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    displayPaths.mutex.lock();
    if (pathNo+1>displayPaths.displays.size())
        displayPaths.displays.resize(pathNo+1);
    displayPaths.displays[pathNo].createDisplayList(path, _lineWidth, _pointSize, _lineColor, _pointColor);
    displayPaths.mutex.unlock();
}

/// add trajectory
void QGLVisualizer::addTrajectory(const Recorder6D& trajectory, double _lineWidth, double _pointSize,
                                  const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    recorders6D.push_back(trajectory);
    displayTrajectories.mutex.lock();
    displayTrajectories.displays.resize(displayTrajectories.displays.size()+1);
    displayTrajectories.displays.back().createDisplayList(trajectory, _lineWidth, _pointSize, _lineColor, _pointColor,
                                                          realTime.now(), false);
    displayTrajectories.mutex.unlock();
}

/// update path
void QGLVisualizer::updateTrajectory(const Recorder6D& trajectory, size_t trajectoryNo, double _lineWidth,
                                     double _pointSize, const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    recorders6D[trajectoryNo] = trajectory;
    displayTrajectories.mutex.lock();
    if (trajectoryNo+1>displayTrajectories.displays.size())
        displayTrajectories.displays.resize(trajectoryNo+1);
    displayTrajectories.displays[trajectoryNo].createDisplayList(trajectory, _lineWidth, _pointSize, _lineColor,
                                                                 _pointColor, realTime.now(), false);
    displayTrajectories.mutex.unlock();
}

/// add rrt trees
void QGLVisualizer::addTree(const planner::RRTree& tree, double _lineWidth, double _pointSize,
                            const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    displayTrees.mutex.lock();
    displayTrees.displays.resize(displayTrees.displays.size()+1);
    displayTrees.displays.back().createDisplayList(tree, _lineWidth, _pointSize, _lineColor, _pointColor);
    displayTrees.mutex.unlock();
}

/// update rrt trees
void QGLVisualizer::updateTree(const planner::RRTree& tree, size_t treeNo, double _lineWidth, double _pointSize,
                               const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    displayTrees.mutex.lock();
    if (treeNo+1>displayTrees.displays.size())
        displayTrees.displays.resize(treeNo+1);
    displayTrees.displays[treeNo].createDisplayList(tree, _lineWidth, _pointSize, _lineColor, _pointColor);
    displayTrees.mutex.unlock();
}

/// add path
void QGLVisualizer::addRobotPath(const std::vector<planner::RobotState3D>& robotPath, double _lineWidthRobot, double _pointSizeRobot,
                                 const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                                 double _lineWidthFeet, double _pointSizeFeet,
                                 const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet){
    displayRobotPaths.mutex.lock();
    displayRobotPaths.displays.resize(displayRobotPaths.displays.size()+1);
    displayRobotPaths.displays.back().createDisplayList(robotPath, _lineWidthRobot, _pointSizeRobot, _lineColorRobot, _pointColorRobot,
                                                        _lineWidthFeet, _pointSizeFeet, _lineColorFeet, _pointColorFeet);
    displayRobotPaths.mutex.unlock();

    /// to visualize robot at the given position
    mtxPlannedPath.lock();
    plannedPath = robotPath;
    mtxPlannedPath.unlock();
}

/// update path
void QGLVisualizer::updateRobotPath(const std::vector<planner::RobotState3D>& robotPath, size_t pathNo, double _lineWidthRobot, double _pointSizeRobot,
                                    const mapping::RGBA& _lineColorRobot, const mapping::RGBA& _pointColorRobot,
                                    double _lineWidthFeet, double _pointSizeFeet,
                                    const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet){
    displayRobotPaths.mutex.lock();
    if (pathNo+1>displayRobotPaths.displays.size())
        displayRobotPaths.displays.resize(pathNo+1);
    displayRobotPaths.displays[pathNo].createDisplayList(robotPath, _lineWidthRobot, _pointSizeRobot, _lineColorRobot, _pointColorRobot,
                                                         _lineWidthFeet, _pointSizeFeet, _lineColorFeet, _pointColorFeet);
    displayRobotPaths.mutex.unlock();

    /// to visualize robot at the given position
    mtxPlannedPath.lock();
    plannedPath = robotPath;
    mtxPlannedPath.unlock();
}

/// add polynomial
void QGLVisualizer::addPolynomial(regression::Regression* _poly, double _centerX, double _centerY, double _offsetZ, double _surfWidth,
                                  double _surfLength, size_t _verticesNo){
    displayPolynomials.displays.resize(displayPolynomials.displays.size()+1);
    displayPolynomials.displays.back().createDisplayList(_centerX, _centerY, _offsetZ,
                                                         _surfWidth, _surfLength, _verticesNo, _poly);
}

/// update polynomial
void QGLVisualizer::updatePolynomial(regression::Regression* _poly, size_t polyNo, double _centerX, double _centerY, double _offsetZ,
                                     double _surfWidth, double _surfLength, size_t _verticesNo){
    if (polyNo+1>displayPolynomials.displays.size())
        displayPolynomials.displays.resize(polyNo+1);
    displayPolynomials.displays[polyNo].createDisplayList(_centerX, _centerY, _offsetZ,
                                                          _surfWidth, _surfLength, _verticesNo, _poly);
}

/// add point cloud
void QGLVisualizer::addCloud(const mapping::PointCloud& _cloud){
    displayPointClouds.mutex.lock();
    displayPointClouds.displays.resize(displayPointClouds.displays.size()+1);
    displayPointClouds.displays.back().pointSize = config.pointSize;
    displayPointClouds.displays.back().createDisplayList(_cloud);
    displayPointClouds.mutex.unlock();
}

/// modify existing point cloud
void QGLVisualizer::updateCloud(const mapping::PointCloud& _cloud, size_t cloudNo){
    displayPointClouds.mutex.lock();
    if (cloudNo+1>displayPointClouds.displays.size())
        displayPointClouds.displays.resize(cloudNo+1);
    displayPointClouds.displays[cloudNo].pointSize = config.pointSize;
    displayPointClouds.displays[cloudNo].createDisplayList(_cloud);
    displayPointClouds.mutex.unlock();
}

/// modify existing point cloud
void QGLVisualizer::updateCloud(const mapping::PointCloud& _cloud, size_t cloudNo, double pointSize){
    displayPointClouds.mutex.lock();
    if (cloudNo+1>displayPointClouds.displays.size())
        displayPointClouds.displays.resize(cloudNo+1);
    displayPointClouds.displays[cloudNo].pointSize = pointSize;
    displayPointClouds.displays[cloudNo].createDisplayList(_cloud);
    displayPointClouds.mutex.unlock();
}

/// modify existing point cloud
void QGLVisualizer::updateCloud(const mapping::PointCloud& _cloud, size_t cloudNo, double pointSize, const mapping::RGBA& color){
    mapping::PointCloud cloud(_cloud);
    for (auto& point : cloud){
        point.color.r = color.r; point.color.g = color.g;
        point.color.b = color.b; point.color.a = color.a;
    }
    displayPointClouds.mutex.lock();
    if (cloudNo+1>displayPointClouds.displays.size())
        displayPointClouds.displays.resize(cloudNo+1);
    displayPointClouds.displays[cloudNo].pointSize = pointSize;
    displayPointClouds.displays[cloudNo].createDisplayList(_cloud);
    displayPointClouds.mutex.unlock();
}

/// add mesh
void QGLVisualizer::addMesh(const ObjectsMesh& _mesh){
    displayMeshes.mutex.lock();
    for (size_t meshNo=0; meshNo<_mesh.filenames.size();meshNo++){
        bool meshExist = false;
        for (const auto& mesh : displayMeshes.displays){
            if (mesh.filename == _mesh.filenames[meshNo])
                meshExist = true;
        }
        if (!meshExist){
            ObjectsMesh singleMesh;
            singleMesh.filenames.push_back(_mesh.filenames[meshNo]);
            singleMesh.scale = _mesh.scale;
            displayMeshes.displays.resize(displayMeshes.displays.size()+1);
            displayMeshes.displays.back().createDisplayList(singleMesh, _mesh.filenames[meshNo]);
        }
    }
    displayMeshes.mutex.unlock();
}

/// update mesh
void QGLVisualizer::updateMesh(const ObjectsMesh& _mesh, size_t meshNo){
    displayMeshes.mutex.lock();
    if (meshNo+1>displayMeshes.displays.size())
        displayMeshes.displays.resize(meshNo+1);
    ObjectsMesh singleMesh;
    singleMesh.filenames.push_back(_mesh.filenames[meshNo]);
    displayMeshes.displays[meshNo].createDisplayList(singleMesh, _mesh.filenames[meshNo]);
    displayMeshes.mutex.unlock();
}

/// get object pose
walkers::Mat34 QGLVisualizer::getObjectPose(const std::string& objectName){
    mtxObjects.lock();
    bool found(false);
    walkers::Mat34 poseObj;
    for (const auto& object : renderObjects){
        if (object.name == objectName){
            found = true;
            poseObj = object.mat;
            break;
        }
    }
    mtxObjects.unlock();
    if (found)
        return poseObj;
    else
        return walkers::Mat34::Identity();
}

/// get distance to surface/object uder pixel
double QGLVisualizer::getDepthUnderPixel(int u, int v){
    //we have to read pixel position in another thread
    pointPixel.rx() = u; pointPixel.ry() = v;
    getPointUnderPixel = true;
    while(getPointUnderPixel){
        usleep(100);
    };
    qglviewer::Vec pix3d = pointUnderPixel;
    std::cout << "pix3d " << pix3d << "\n";
    walkers::Mat34 point(walkers::Mat34::Identity());
    point(0,3) = pix3d.x; point(1,3) = pix3d.y; point(2,3) = pix3d.z;
    point = getCameraPose().inverse()*point;
    return -point(2,3);
}

/// get object position on the image
std::pair<int,int> QGLVisualizer::getObjectPositionUV(const std::string& objectName, double& distance2surface){
    std::pair<int,int> coords(-1,-1);
    distance2surface=-1.0;
    mtxObjects.lock();
    bool found(false);
    walkers::Mat34 poseObj;
    for (const auto& object : renderObjects){
        if (object.name == objectName){
            found = true;
            poseObj = object.mat;
            break;
        }
    }
    mtxObjects.unlock();
    if (found){
        /// coords
        walkers::Mat34 objPoseInCam = getCameraPose().inverse() * poseObj;
        objPoseInCam(0,3) = objPoseInCam(0,3);
        objPoseInCam(1,3) = -objPoseInCam(1,3);
        objPoseInCam(2,3) = -objPoseInCam(2,3);
        walkers::Vec3 center = cameraModel->inverseModel(objPoseInCam(0,3), objPoseInCam(1,3), objPoseInCam(2,3));
        coords.first = (int)center.x(); coords.second = (int)center.y();
        /// depth under pixel
        found = false;
        if (coords.first>=0&&coords.second>=0){
            distance2surface = getDepthUnderPixel(coords.first, coords.second);
        }
        return coords;
    }
    else
        return coords;
}

/// add line
void QGLVisualizer::addLine(const walkers::Line3D& _line){
    displayLines.mutex.lock();
    displayLines.displays.resize(displayLines.displays.size()+1);
    displayLines.displays.back().lineWidth = config.lineWidth;
    displayLines.displays.back().lineColor = config.lineColor;
    displayLines.displays.back().createDisplayList(_line);
    displayLines.mutex.unlock();
}

/// update Lines
void QGLVisualizer::updateLine(const walkers::Line3D& _line, size_t lineNo){
    displayLines.mutex.lock();
    if (lineNo+1>displayLines.displays.size())
        displayLines.displays.resize(lineNo+1);
    displayLines.displays[lineNo].lineWidth = config.lineWidth;
    displayLines.displays[lineNo].lineColor = config.lineColor;
    displayLines.displays[lineNo].createDisplayList(_line);
    displayLines.mutex.unlock();
}

/// update Lines
void QGLVisualizer::updateLine(const walkers::Line3D& _line, size_t lineNo, const mapping::RGBA& color){
    displayLines.mutex.lock();
    if (lineNo+1>displayLines.displays.size())
        displayLines.displays.resize(lineNo+1);
    displayLines.displays[lineNo].lineWidth = config.lineWidth;
    displayLines.displays[lineNo].lineColor = color;
    displayLines.displays[lineNo].createDisplayList(_line);
    displayLines.mutex.unlock();
}

/// add arrow
void QGLVisualizer::addArrow(const walkers::Arrow3D& _arrow, double arrowWidth,
                             const mapping::RGBA& arrowColor){
    displayArrows.mutex.lock();
    displayArrows.displays.resize(displayArrows.displays.size()+1);
    displayArrows.displays.back().arrowWidth = arrowWidth;
    displayArrows.displays.back().arrowColor = arrowColor;
    displayArrows.displays.back().createDisplayList(_arrow);
    displayArrows.mutex.unlock();
}

/// update Arrow
void QGLVisualizer::updateArrow(const walkers::Arrow3D& _arrow, size_t arrowNo){
    displayArrows.mutex.lock();
    if (arrowNo+1>displayArrows.displays.size())
        displayArrows.displays.resize(arrowNo+1);
    displayArrows.displays[arrowNo].createDisplayList(_arrow);
    displayArrows.mutex.unlock();
}

/// update Arrows
void QGLVisualizer::updateArrow(const walkers::Arrow3D& _arrow, size_t arrowNo, double arrowWidth,
                                const mapping::RGBA& arrowColor){
    displayArrows.mutex.lock();
    if (arrowNo+1>displayArrows.displays.size())
        displayArrows.displays.resize(arrowNo+1);
    displayArrows.displays[arrowNo].arrowWidth = arrowWidth;
    displayArrows.displays[arrowNo].arrowColor = arrowColor;
    displayArrows.displays[arrowNo].createDisplayList(_arrow);
    displayArrows.mutex.unlock();
}

/// add BSpline3D
void QGLVisualizer::addBSpline3D(const walkers::BSpline3D& _bspline){
    displayBSplines3D.mutex.lock();
    displayBSplines3D.displays.resize(displayBSplines3D.displays.size()+1);
    displayBSplines3D.displays.back().lineWidth = config.lineWidth;
    displayBSplines3D.displays.back().lineColor = config.lineColor;
    displayBSplines3D.displays.back().createDisplayList(_bspline);
    displayBSplines3D.mutex.unlock();
}

/// update BSpline3D
void QGLVisualizer::updateBSpline3D(const walkers::BSpline3D& _bspline, size_t bsplineNo){
    displayBSplines3D.mutex.lock();
    if (bsplineNo+1>displayBSplines3D.displays.size())
        displayBSplines3D.displays.resize(bsplineNo+1);
    displayBSplines3D.displays[bsplineNo].lineWidth = config.lineWidth;
    displayBSplines3D.displays[bsplineNo].lineColor = config.lineColor;
    displayBSplines3D.displays[bsplineNo].createDisplayList(_bspline);
    displayBSplines3D.mutex.unlock();
}

/// update BSplines3D
void QGLVisualizer::updateBSpline3D(const walkers::BSpline3D& _bspline, size_t bsplineNo, size_t pointsNo, double lineWidth,
                                    const mapping::RGBA& lineColor, const mapping::RGBA& ctrlLineColor, double ctrlPointSize,
                                    bool drawCtrlPoints, bool drawCtrlLines){
    displayBSplines3D.mutex.lock();
    if (bsplineNo+1>displayBSplines3D.displays.size())
        displayBSplines3D.displays.resize(bsplineNo+1);
    displayBSplines3D.displays[bsplineNo].pointsNo = pointsNo;
    displayBSplines3D.displays[bsplineNo].lineWidth = lineWidth;
    displayBSplines3D.displays[bsplineNo].lineColor = lineColor;
    displayBSplines3D.displays[bsplineNo].ctrlLineColor = ctrlLineColor;
    displayBSplines3D.displays[bsplineNo].ctrlPointSize = ctrlPointSize;
    displayBSplines3D.displays[bsplineNo].drawCtrlPoints = drawCtrlPoints;
    displayBSplines3D.displays[bsplineNo].drawCtrlLines = drawCtrlLines;
    displayBSplines3D.displays[bsplineNo].createDisplayList(_bspline);
    displayBSplines3D.mutex.unlock();
}

/// add BSplineSE3
void QGLVisualizer::addBSplineSE3(const walkers::BSplineSE3& _bspline){
    displayBSplinesSE3.mutex.lock();
    displayBSplinesSE3.displays.resize(displayBSplinesSE3.displays.size()+1);
    displayBSplinesSE3.displays.back().lineWidth = config.lineWidth;
    displayBSplinesSE3.displays.back().lineColor = config.lineColor;
    displayBSplinesSE3.displays.back().createDisplayList(_bspline);
    displayBSplinesSE3.mutex.unlock();
}

/// update BSplineSE3
void QGLVisualizer::updateBSplineSE3(const walkers::BSplineSE3& _bspline, size_t bsplineNo){
    displayBSplinesSE3.mutex.lock();
    if (bsplineNo+1>displayBSplinesSE3.displays.size())
        displayBSplinesSE3.displays.resize(bsplineNo+1);
    displayBSplinesSE3.displays[bsplineNo].lineWidth = config.lineWidth;
    displayBSplinesSE3.displays[bsplineNo].lineColor = config.lineColor;
    displayBSplinesSE3.displays[bsplineNo].createDisplayList(_bspline);
    displayBSplinesSE3.mutex.unlock();
}

/// update BSplinesSE3
void QGLVisualizer::updateBSplineSE3(const walkers::BSplineSE3& _bspline, size_t bsplineNo, size_t pointsNo, double lineWidth,
                                    const mapping::RGBA& lineColor, const mapping::RGBA& ctrlLineColor, double ctrlPointSize,
                                    bool drawCtrlPoints, bool drawCtrlLines, bool drawAxis, double framesScale){

    displayBSplinesSE3.mutex.lock();
    if (bsplineNo+1>displayBSplinesSE3.displays.size())
        displayBSplinesSE3.displays.resize(bsplineNo+1);
    displayBSplinesSE3.displays[bsplineNo].pointsNo = pointsNo;
    displayBSplinesSE3.displays[bsplineNo].lineWidth = lineWidth;
    displayBSplinesSE3.displays[bsplineNo].lineColor = lineColor;
    displayBSplinesSE3.displays[bsplineNo].ctrlLineColor = ctrlLineColor;
    displayBSplinesSE3.displays[bsplineNo].ctrlPointSize = ctrlPointSize;
    displayBSplinesSE3.displays[bsplineNo].drawCtrlPoints = drawCtrlPoints;
    displayBSplinesSE3.displays[bsplineNo].drawCtrlLines = drawCtrlLines;
    displayBSplinesSE3.displays[bsplineNo].drawFrames = drawAxis;
    displayBSplinesSE3.displays[bsplineNo].framesScale = framesScale;
    displayBSplinesSE3.displays[bsplineNo].createDisplayList(_bspline);
    displayBSplinesSE3.mutex.unlock();
}

/// add axis
void QGLVisualizer::addAxis(const walkers::Mat34& _axis, double lineWidth, double scale){
    displayAxis.mutex.lock();
    displayAxis.displays.resize(displayAxis.displays.size()+1);
    displayAxis.displays.back().lineWidth = lineWidth;
    displayAxis.displays.back().scale = scale;
    displayAxis.displays.back().createDisplayList(_axis);
    displayAxis.mutex.unlock();
}

/// update axis
void QGLVisualizer::updateAxis(const walkers::Mat34& _axis, size_t axisNo, double lineWidth, double scale){
    displayAxis.mutex.lock();
    if (axisNo+1>displayAxis.displays.size())
        displayAxis.displays.resize(axisNo+1);
    displayAxis.displays[axisNo].lineWidth = lineWidth;
    displayAxis.displays[axisNo].scale = scale;
    displayAxis.displays[axisNo].createDisplayList(_axis);
    displayAxis.mutex.unlock();
}

/// add planes
void QGLVisualizer::addPlane(const std::vector<walkers::Plane3D>& _planes, const mapping::RGBA& color){
    displayPlanes.mutex.lock();
    displayPlanes.displays.resize(displayPlanes.displays.size()+1);
    displayPlanes.displays.back().createDisplayList(_planes, color);
    displayPlanes.mutex.unlock();
}

/// update Planes
void QGLVisualizer::updatePlane(const std::vector<walkers::Plane3D>& _plane, size_t planeNo, const mapping::RGBA& color){
    displayPlanes.mutex.lock();
    if (planeNo+1>displayPlanes.displays.size())
        displayPlanes.displays.resize(planeNo+1);
    displayPlanes.displays[planeNo].createDisplayList(_plane, color);
    displayPlanes.mutex.unlock();
}

/// add triangles
void QGLVisualizer::addTriangle(const std::vector<walkers::Triangle3D>& _triangles, const mapping::RGBA& color){
    displayTriangles.mutex.lock();
    displayTriangles.displays.resize(displayTriangles.displays.size()+1);
    displayTriangles.displays.back().createDisplayList(_triangles, color);
    displayTriangles.mutex.unlock();
}

/// update Triangles
void QGLVisualizer::updateTriangle(const std::vector<walkers::Triangle3D>& _triangle, size_t triangleNo, const mapping::RGBA& color){
    displayTriangles.mutex.lock();
    if (triangleNo+1>displayTriangles.displays.size())
        displayTriangles.displays.resize(triangleNo+1);
    displayTriangles.displays[triangleNo].createDisplayList(_triangle, color);
    displayTriangles.mutex.unlock();
}

/// update ellipsoids
void QGLVisualizer::updateEllipsoids(const mapping::Ellipsoid::Seq& _ellipsoids, size_t ellipsoidNo) {
    displayEllipsoids.mutex.lock();
    if (ellipsoidNo+1>displayEllipsoids.displays.size())
        displayEllipsoids.displays.resize(ellipsoidNo+1);
    displayEllipsoids.displays[ellipsoidNo].createDisplayList(_ellipsoids);
    displayEllipsoids.mutex.unlock();
}

/// add ellipsoids
void QGLVisualizer::addEllipsoids(const mapping::Ellipsoid::Seq& _ellipsoids) {
    displayEllipsoids.mutex.lock();
    displayEllipsoids.displays.resize(displayEllipsoids.displays.size()+1);
    displayEllipsoids.displays.back().createDisplayList(_ellipsoids);
    displayEllipsoids.mutex.unlock();
}

/// update voxels
void QGLVisualizer::updateVoxels(const mapping::VoxelVisu::Seq& _voxels, size_t voxelsNo) {
    displayVoxels.mutex.lock();
    if (voxelsNo+1>displayVoxels.displays.size())
        displayVoxels.displays.resize(voxelsNo+1);
    displayVoxels.displays[voxelsNo].createDisplayList(_voxels);
    displayVoxels.mutex.unlock();
}

/// add voxels
void QGLVisualizer::addVoxels(const mapping::VoxelVisu::Seq& _voxels) {
    displayVoxels.mutex.lock();
    displayVoxels.displays.resize(displayVoxels.displays.size()+1);
    displayVoxels.displays.back().createDisplayList(_voxels);
    displayVoxels.mutex.unlock();
}

/// resize
void QGLVisualizer::resizeWindow(int width, int height) {
    camera()->setScreenWidthAndHeight(width, height);
    resize(width,height);
    resizeGL(width,height);
    //resizeOverlayGL(width,height);
    double hov = 2*atan2(cameraModel->resolution[0],2*cameraModel->focalLength[0]);
    camera()->setHorizontalFieldOfView(hov);
}

/// snap
void QGLVisualizer::snap(int width, int height, const std::string& filenameDepth, const std::string& filenameRGB,
                         const std::string& filenameSegment, const std::string& filenameCloud) {
    std::cout << "snap started\n";
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cv::Mat depthImage(height, width, CV_16UC1, cv::Scalar(0));
    cv::Mat segmentImage(height, width, CV_8UC1, cv::Scalar(0));
    /// point cloud
    grabber::PointCloud cloud;
    setSnapshotQuality(100);
    setSnapshotFormat(QString("png"));
    saveSnapshot(QString(filenameRGB.c_str()), true);
    cv::Mat colorImage = cv::imread(filenameRGB, cv::IMREAD_COLOR);

    walkers::Mat34 camPose = getCameraPose();

    unsigned int dataStencil[width*height];
    glReadPixels(0, 0, width, height, GL_STENCIL_INDEX, GL_UNSIGNED_INT, dataStencil);
    qglviewer::Vec pix3d;
    for (int colNo=0; colNo<width; colNo++){
        for (int rowNo=0; rowNo<height; rowNo++){
            bool found = false;
            QPoint pixel(colNo,rowNo);
            pix3d = camera()->pointUnderPixel(pixel, found);
            //            GLbyte color[4];
            //            GLfloat depth;
            //            GLuint index;

            //            glReadPixels(colNo, rowNo, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, color);
            //            glReadPixels(colNo, rowNo, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
            //            glReadPixels(colNo, height-rowNo-1, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_INT, &index);

            //            printf("Clicked on pixel %d, %d, color %02hhx%02hhx%02hhx%02hhx, depth %f, stencil index %u\n",
            //                   x, y, color[0], color[1], color[2], color[3], depth, index);
            //            segmentImage.at<uint8_t>(rowNo, colNo) = (unsigned char)(index);
            segmentImage.at<uint8_t>(height - rowNo - 1, colNo) =
                    (unsigned char)(dataStencil[rowNo*width+colNo]);

            if (found){
                walkers::Mat34 point(walkers::Mat34::Identity());
                point(0,3) = pix3d.x; point(1,3) = pix3d.y; point(2,3) = pix3d.z;
                point = camPose.inverse()*point;
                double depthVal = -point(2,3);
                if (depthVal>0.2&&depthVal<8.0){
                    depthImage.at<uint16_t>(rowNo, colNo) = (short unsigned int)(5000*depthVal);
                }
                else
                    depthImage.at<uint16_t>(rowNo, colNo) = (short unsigned int)(0);
                grabber::Point3D point3D;
                point3D.x = (float)point(0,3); point3D.y = -(float)point(1,3); point3D.z = -(float)point(2,3);
                const cv::Point3_ <uchar>* p = colorImage.ptr<cv::Point3_<uchar> >(rowNo, colNo);
                point3D.r = p->z; point3D.g = p->y; point3D.b = p->x;
                cloud.push_back(point3D);
            }
        }
    }

    if (filenameCloud!=""){
        pcl::io::savePCDFile(filenameCloud, cloud);
    }
    if (filenameDepth!=""){
        cv::imwrite(filenameDepth, depthImage);
    }
    if (filenameSegment!=""){
        cv::imwrite(filenameSegment, segmentImage);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference for images generation = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    std::cout << "snap finished\n";
    if (poseNo2save == cameraPoses2save.size()){
        busySnap = false;
    }
}

///draw polynomial fitting example
void QGLVisualizer::drawPolynomialFitting(){
    glPushMatrix();
    float reflectColor[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, reflectColor);
    GLfloat emissiveLight[] = { 0.1f, 0.1f, 0.1f, 1.0f };
    glMaterialfv(GL_FRONT, GL_EMISSION, emissiveLight);
    displayPolynomials.draw();
    glPopMatrix();
}

/// get camera pose
walkers::Mat34 QGLVisualizer::getCameraPose(void){
    qglviewer::Vec camPosition = camera()->position();
    qglviewer::Quaternion camRot = camera()->orientation();walkers::Quaternion camRotQuat;
    camRotQuat.x() = camRot[0]; camRotQuat.y() = camRot[1]; camRotQuat.z() = camRot[2]; camRotQuat.w() = camRot[3];
    walkers::Mat34 camPose(walkers::Mat34::Identity());
    camPose = walkers::toRotationMat(camRotQuat);
    camPose(0,3) = camPosition.x; camPose(1,3) = camPosition.y; camPose(2,3) = camPosition.z;
    return camPose;
}

/// set camera pose
void QGLVisualizer::setCameraPose(const walkers::Mat34& cameraPose){
    walkers::Mat34 offsetCam(walkers::Mat34::Identity());
    offsetCam = walkers::toRotationMat(walkers::Vec3(3.14,0.0,0.0));

    walkers::Mat34 camPos = cameraPose*offsetCam;
    qglviewer::Vec position(camPos(0,3),camPos(1,3), camPos(2,3));
    camera()->setPosition(position);
    walkers::Quaternion quatWalker(camPos.rotation());
    qglviewer::Quaternion quat(quatWalker.x(), quatWalker.y(), quatWalker.z(), quatWalker.w());
    camera()->setOrientation(quat);
}

/// generate dataset - camera views
void QGLVisualizer::generateViews(const std::vector<walkers::Mat34>& _cameraPoses,
                                  const std::vector<int>& _parentIds, const std::vector<double>& _pitchs,
                                  const std::vector<double>& _yaws, size_t offsetImageNo){
    busySnap = true;
    poseNo2save = 0;
    cameraPoses2save = _cameraPoses;
    parentIds2save = _parentIds;
    pitchs2save = _pitchs;
    yaws2save = _yaws;
//    std::ofstream fileCamPoses;
    offsetImageNo2save = offsetImageNo;
    //    fileCamPoses.open ("dataset/camPoses.txt");
    //    fileCamPoses.close();
    std::cout << "generate views start\n";
}

/// append camera pose (pitch and yaw with respect to parent_no)
void QGLVisualizer::appendCameraPose(size_t poseNo, int parentNo, double pitch, double yaw, const walkers::Mat34& cameraPose, const std::string& filename){
    std::ofstream fileCamPoses;
    fileCamPoses.open (filename, std::ios::app);

    //pose no
    fileCamPoses << poseNo << " " << parentNo << " " << pitch << " " << yaw << " ";
    // pose SE3
    for (int rowNo=0;rowNo<4;rowNo++){
        for (int colNo=0;colNo<4;colNo++){
            fileCamPoses << cameraPose(rowNo, colNo) << " ";
        }
    }
    fileCamPoses << "\n";
    fileCamPoses.close();
}

/// append objects list
void QGLVisualizer::appendObjectsList(const std::string& filename){
    std::ofstream fileObjects;
    fileObjects.open (filename, std::ios::app);

    //pose no
    //    fileCamPoses << poseNo << " " << parentNo << " " << pitch << " " << yaw << " ";
    // pose SE3
    std::vector<std::string> categories;
    std::vector<std::string> instances;
    std::vector<walkers::Mat34> poses;
    std::vector<size_t> objsIds;
    size_t objectNo = 0;
    mtxObjects.lock();
    for (const auto& renderObject : renderObjects){
        std::filesystem::path pathMesh = renderObject.meshFilename;
        size_t folderNo=0;
        std::string category = "";
        std::string instance = "";
        for (auto it = pathMesh.begin(); it != pathMesh.end(); ++it){
            if (folderNo==10)//depends on the location of the dataset o the disk
                category = *it;
            if (folderNo==11)//depends on the location of the dataset o the disk
                instance = *it;
            folderNo++;
        }
        if (category!="" && instance!=""){
            categories.push_back(category);
            instances.push_back(instance);
            poses.push_back(renderObject.mat);
            objsIds.push_back(objectNo);
//            std::cout << "render object " << renderObject.id << ", " << category <<" " << instance  << "\n";
        }
        objectNo++;
    }
    mtxObjects.unlock();

    walkers::Mat34 camPose = getCameraPose();
    for (size_t objNo=0;objNo<categories.size();objNo++){
        fileObjects << categories[objNo] << " " << instances[objNo] << " " << objsIds[objNo] << " ";
        walkers::Mat34 objPoseInCam = camPose.inverse() * poses[objNo];
        objPoseInCam(0,3) = objPoseInCam(0,3);
        objPoseInCam(1,3) = -objPoseInCam(1,3);
        objPoseInCam(2,3) = -objPoseInCam(2,3);
//        std::cout << "pose in cam\n" << objPoseInCam.matrix() << "\n";
        walkers::Vec3 center = cameraModel->inverseModel(objPoseInCam(0,3), objPoseInCam(1,3), objPoseInCam(2,3));
//        std::cout << "center " << center.x() << " " << center.y() << "\n";
        fileObjects << (int)center.x() << " " << (int)center.y() << " ";
        // export pose of the object in the camera frame
        fileObjects << objPoseInCam(0,0) << " " << objPoseInCam(0,1) << " " << objPoseInCam(0,2) << " " << objPoseInCam(0,3) << " ";
        fileObjects << objPoseInCam(1,0) << " " << objPoseInCam(1,1) << " " << objPoseInCam(1,2) << " " << objPoseInCam(1,3) << " ";
        fileObjects << objPoseInCam(2,0) << " " << objPoseInCam(2,1) << " " << objPoseInCam(2,2) << " " << objPoseInCam(2,3) << " ";
        fileObjects << objPoseInCam(3,0) << " " << objPoseInCam(3,1) << " " << objPoseInCam(3,2) << " " << objPoseInCam(3,3) << "\n";
    }
    fileObjects.close();
}

/// save screenshot
void QGLVisualizer::saveScreenshot(const std::string& filename){
    resizeFlag = true;
    depth2save = "";
    rgb2save = filename;
    segment2save = "";
    cloud2save = "";
}
