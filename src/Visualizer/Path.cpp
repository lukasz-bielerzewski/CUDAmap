#include "Visualizer/Path.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

/// create display list
void DisplayPath::createDisplayList(const std::vector<planner::PoseSE3>& path, double _lineWidth, double _pointSize,
                                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    lineColor = _lineColor;
    pointColor = _pointColor;
    lineWidth = _lineWidth;
    pointSize = _pointSize;

    element = path;
    updateListGL = true;
}

/// update display list
void DisplayPath::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
    int slices = 8;
    int stacks = 8;
    glLineWidth((float)lineWidth);
    glBegin(GL_LINE_STRIP);
    for (const auto& pathPoint : element){
        glColor4ub(lineColor.r, lineColor.g, lineColor.b, lineColor.a);
        glVertex3d(pathPoint.getPosition()(0), pathPoint.getPosition()(1), pathPoint.getPosition()(2));
    }
    glEnd();
    for (const auto& pathPoint : element){
        glColor4ub(pointColor.r, pointColor.g, pointColor.b, pointColor.a);
        glPushMatrix();
        glTranslated(pathPoint.getPosition()(0), pathPoint.getPosition()(1), pathPoint.getPosition()(2));
        glutSolidSphere(pointSize, slices, stacks);
        glPopMatrix();
    }
    glEndList();

    element.clear();
}
