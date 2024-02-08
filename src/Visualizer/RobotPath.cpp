#include "Visualizer/RobotPath.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

/// create display list
void DisplayRobotPath::createDisplayList(const std::vector<planner::RobotState3D>& robotPath, double _lineWidthBody, double _pointSizeBody,
                                    const mapping::RGBA& _lineColorBody, const mapping::RGBA& _pointColorBody,
                                    double _lineWidthFeet, double _pointSizeFeet,
                                    const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet){
    lineColorBody = _lineColorBody;
    pointColorBody = _pointColorBody;
    lineWidthBody = _lineWidthBody;
    pointSizeBody = _pointSizeBody;
    lineColorFeet = _lineColorFeet;
    pointColorFeet = _pointColorFeet;
    lineWidthFeet = _lineWidthFeet;
    pointSizeFeet = _pointSizeFeet;

    element = robotPath;
    updateListGL = true;
}

/// update display list
void DisplayRobotPath::updateDisplayList(){
    if (element.size()==0)
        return;
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
    int slices = 8;
    int stacks = 8;
    glLineWidth((float)lineWidthBody);
    glBegin(GL_LINE_STRIP);
    for (const auto& pathPoint : element){
        glColor4ub(lineColorBody.r, lineColorBody.g, lineColorBody.b, lineColorBody.a);
        glVertex3d(pathPoint.robotPose.getPosition()(0), pathPoint.robotPose.getPosition()(1), pathPoint.robotPose.getPosition()(2));
    }
    glEnd();
    for (const auto& pathPoint : element){
        glColor4ub(pointColorBody.r, pointColorBody.g, pointColorBody.b, pointColorBody.a);
        glPushMatrix();
        glTranslated(pathPoint.robotPose.getPosition()(0), pathPoint.robotPose.getPosition()(1), pathPoint.robotPose.getPosition()(2));
        glutSolidSphere(pointSizeBody, slices, stacks);
        glPopMatrix();
    }

    for (size_t legNo=0;legNo<element.front().feet.size();legNo++){
        glLineWidth((float)lineWidthFeet);
        glBegin(GL_LINE_STRIP);
        for (const auto& pathPoint : element){
            glColor4ub(lineColorFeet.r, lineColorFeet.g, lineColorFeet.b, lineColorFeet.a);
            if (legNo<pathPoint.feet.size()){
                glVertex3d(pathPoint.feet[legNo].pose(0,3), pathPoint.feet[legNo].pose(1,3), pathPoint.feet[legNo].pose(2,3));
            }
        }
        glEnd();
        for (const auto& pathPoint : element){
            if (legNo<pathPoint.feet.size()){
                glColor4ub(pointColorFeet.r, pointColorFeet.g, pointColorFeet.b, pointColorFeet.a);
                glPushMatrix();
                glTranslated(pathPoint.feet[legNo].pose(0,3), pathPoint.feet[legNo].pose(1,3), pathPoint.feet[legNo].pose(2,3));
                glutSolidSphere(pointSizeFeet, slices, stacks);
                glPopMatrix();
            }
        }
    }
    glEndList();
    element.clear();
}
