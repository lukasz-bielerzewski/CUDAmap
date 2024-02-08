#include "Visualizer/TrajectorySE3.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

/// create display list
void DisplayTrajectorySE3::createDisplayList(const Recorder6D& trajectory, double _lineWidth, double _pointSize,
                                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor, double _currentTime,
                                             bool _drawAllTraj){
    lineColor = _lineColor;
    pointColor = _pointColor;
    lineWidth = _lineWidth;
    pointSize = _pointSize;

    element = trajectory;
    updateListGL = true;

    currentTime = _currentTime;
    drawAllTraj = _drawAllTraj;
}

/// draw axis
void DisplayTrajectorySE3::drawAxis(double _scale){
    glLineWidth((float)(lineWidth+1.0));
    glColor4ub(lineColor.r, lineColor.g, lineColor.b, lineColor.a);
    //x
    glBegin(GL_LINES);
    glVertex3d(0.0,0.0,0.0);
    glVertex3d(1.0*_scale,0.0,0.0);
    glEnd();

    //y
    glBegin(GL_LINES);
    glVertex3d(0,0,0);
    glVertex3d(0,1*_scale,0);
    glEnd();

    //z
    glBegin(GL_LINES);
    glVertex3d(0,0,0);
    glVertex3d(0,0,1*_scale);
    glEnd();
}

/// update display list
void DisplayTrajectorySE3::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
    int slices = 8;
    int stacks = 8;
    glLineWidth((float)lineWidth);
    glBegin(GL_LINE_STRIP);
    walkers::Mat34 lastPose;
    bool lastFound=false;
    for (const auto& trajectorySE3Point : element.container){
        glColor4ub(lineColor.r, lineColor.g, lineColor.b, lineColor.a);
        glVertex3d(trajectorySE3Point.second(0,3), trajectorySE3Point.second(1,3),
                   trajectorySE3Point.second(2,3));
        if (!drawAllTraj&&!lastFound&&trajectorySE3Point.first/1000000>currentTime){
            lastPose = trajectorySE3Point.second;
            lastFound = true;
            break;
        }
    }
    glEnd();
    for (const auto& trajectorySE3Point : element.container){
        glColor4ub(pointColor.r, pointColor.g, pointColor.b, pointColor.a);
        glPushMatrix();
        glTranslated(trajectorySE3Point.second(0,3), trajectorySE3Point.second(1,3),
                     trajectorySE3Point.second(2,3));
        glutSolidSphere(pointSize, slices, stacks);
        glPopMatrix();
        if (!drawAllTraj&&trajectorySE3Point.first/1000000>currentTime){
            break;
        }
    }
    if (!lastFound)
        lastPose = element.container.front().second;

    double GLmat[16]={lastPose(0,0), lastPose(1,0), lastPose(2,0), lastPose(3,0),
                        lastPose(0,1), lastPose(1,1), lastPose(2,1), lastPose(3,1),
                        lastPose(0,2), lastPose(1,2), lastPose(2,2), lastPose(3,2),
                        lastPose(0,3), lastPose(1,3), lastPose(2,3), lastPose(3,3)};
    glPushMatrix();
    glMultMatrixd(GLmat);
    drawAxis(0.1f);
    glPopMatrix();

    glEndList();

//    element.container.clear();
}

/// set current time
void DisplayTrajectorySE3::setCurrentTime(double _currentTime){
    currentTime = _currentTime;
    updateListGL = true;
}
