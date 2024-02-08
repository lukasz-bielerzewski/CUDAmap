#include "Visualizer/BSplineSE3.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayBSplineSE3::createDisplayList(const walkers::BSplineSE3& bsplineSE3){
    element = bsplineSE3;
    updateListGL = true;
}

/// draw axis
void DisplayBSplineSE3::drawAxis(double _scale){
    glLineWidth((float)(lineWidth+1.0));

    //x
    glColor4ub(255, 0, 0, 255);
    glBegin(GL_LINES);
    glVertex3d(0.0,0.0,0.0);
    glVertex3d(1.0*_scale,0.0,0.0);
    glEnd();

    //y
    glColor4ub(0, 255, 0, 255);
    glBegin(GL_LINES);
    glVertex3d(0,0,0);
    glVertex3d(0,1*_scale,0);
    glEnd();

    //z
    glColor4ub(0, 0, 255, 255);
    glBegin(GL_LINES);
    glVertex3d(0,0,0);
    glVertex3d(0,0,1*_scale);
    glEnd();
}

/// update display list
void DisplayBSplineSE3::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
//    std::vector<walkers::Vec3> samples;
//    element.updateCurve(samples);
//    std::cout << "samples size " << samples.size() << "\n";
    glNewList(listId, GL_COMPILE);
        glLineWidth((GLfloat)lineWidth);
            glColor4ub(lineColor.r, lineColor.g, lineColor.b, lineColor.a);
            glBegin(GL_LINES);
            std::vector<walkers::Mat34> controlPoints = element.getControlPoints();
            std::vector<walkers::Mat34> trajectory(pointsNo);
            if (controlPoints.size()>2){
                walkers::Mat34 p1 = controlPoints[0];
                for (size_t pointNo=0; pointNo<pointsNo;pointNo++){
                    double t = ( double(pointNo) / double(pointsNo) ) * (double(controlPoints.size()) - double(element.order-1) ) + double(element.order)-1;
                    walkers::Mat34 p2 = element.compValueRecursive(t);
                    glVertex3f(float(p1(0,3)), float(p1(1,3)), float(p1(2,3)));
                    glVertex3f(float(p2(0,3)), float(p2(1,3)), float(p2(2,3)));
                    p1=p2;
                    trajectory[pointNo]=p2;
                }
                walkers::Mat34 p2 = controlPoints.back();
                glVertex3f(float(p1(0,3)), float(p1(1,3)), float(p1(2,3)));
                glVertex3f(float(p2(0,3)), float(p2(1,3)), float(p2(2,3)));
            }
            glEnd();

            if (drawFrames){
                for (const auto& pos : trajectory){
                    double GLmat[16]={pos(0,0), pos(1,0), pos(2,0), pos(3,0),
                                      pos(0,1), pos(1,1), pos(2,1), pos(3,1),
                                      pos(0,2), pos(1,2), pos(2,2), pos(3,2),
                                      pos(0,3), pos(1,3), pos(2,3), pos(3,3)};
                    glPushMatrix();
                    glMultMatrixd(GLmat);
                        drawAxis(framesScale);
                    glPopMatrix();
                }
            }

            if (drawCtrlPoints){
                // display control points
                glPointSize(GLfloat(ctrlPointSize));
                glBegin(GL_POINTS);
                for(walkers::Mat34 poseTmp : controlPoints) {
                    glColor3ub(0,255,0);
                    glVertex3d(poseTmp(0,3), poseTmp(1,3), poseTmp(2,3));
                }
                glEnd();
            }

            if (drawCtrlLines){
                glLineWidth((GLfloat)1);
                glColor4ub(ctrlLineColor.r, ctrlLineColor.g, ctrlLineColor.b, ctrlLineColor.a);
                glBegin(GL_LINES);
                if (controlPoints.size()>2){
                    walkers::Mat34 p1 = controlPoints[0];
                    for (size_t pointNo=1; pointNo<controlPoints.size();pointNo++){
                        walkers::Mat34 p2 = controlPoints[pointNo];
                        glVertex3f(float(p1(0,3)), float(p1(1,3)), float(p1(2,3)));
                        glVertex3f(float(p2(0,3)), float(p2(1,3)), float(p2(2,3)));
                        p1=p2;
                    }
                }
                glEnd();
            }

            if (drawFrames){
                for (size_t pointNo=0; pointNo<controlPoints.size();pointNo++){
                    walkers::Mat34 lastPose = controlPoints[pointNo];
                    double GLmat[16]={lastPose(0,0), lastPose(1,0), lastPose(2,0), lastPose(3,0),
                                      lastPose(0,1), lastPose(1,1), lastPose(2,1), lastPose(3,1),
                                      lastPose(0,2), lastPose(1,2), lastPose(2,2), lastPose(3,2),
                                      lastPose(0,3), lastPose(1,3), lastPose(2,3), lastPose(3,3)};
                    glPushMatrix();
                    glMultMatrixd(GLmat);
                    drawAxis(framesScale*2);
                    glPopMatrix();
                }
            }
    glEndList();
}
