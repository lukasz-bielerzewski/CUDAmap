#include "Visualizer/Tree.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

/// create display list
void DisplayTree::createDisplayList(const planner::RRTree& tree, double _lineWidth, double _pointSize,
                                    const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor){
    lineColor = _lineColor;
    pointColor = _pointColor;
    lineWidth = _lineWidth;
    pointSize = _pointSize;

    element = tree;
    updateListGL = true;
}

/// update display list
void DisplayTree::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
    glLineWidth((float)lineWidth);
    glBegin(GL_LINES);
    glColor4ub(lineColor.r, lineColor.g, lineColor.b, lineColor.a);
    std::set<size_t> children ={0};
    while (children.size()>0){
        size_t nodeNo=*children.begin();
        planner::RRTNode node = element.getNode(nodeNo);
        planner::RRTNode parentNode;
        if (node.parent>0)
            parentNode = element.getNode(node.parent);
        else
            parentNode = element.getNode(0);
        glVertex3d(parentNode.robotPose.body(0,3),parentNode.robotPose.body(1,3),parentNode.robotPose.body(2,3));
        glVertex3d(node.robotPose.body(0,3),node.robotPose.body(1,3),node.robotPose.body(2,3));
        children.erase(children.begin());
        children.insert(node.children.begin(), node.children.end());
    }
    glEnd();
    children = {0};
    while (children.size()>0){
        size_t nodeNo=*children.begin();
        planner::RRTNode node = element.getNode(nodeNo);
        glPushMatrix();
        glTranslated(node.robotPose.body(0,3),node.robotPose.body(1,3),node.robotPose.body(2,3));
        glColor4ub(pointColor.r, pointColor.g, pointColor.b, pointColor.a);
        glutSolidSphere(pointSize, 10, 10);
        glPopMatrix();
        children.erase(children.begin());
        children.insert(node.children.begin(), node.children.end());
    }
    glEndList();

//    element.clear();
}
