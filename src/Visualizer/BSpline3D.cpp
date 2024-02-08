#include "Visualizer/BSpline3D.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayBSpline3D::createDisplayList(const walkers::BSpline3D& bspline3D){
    element = bspline3D;
    updateListGL = true;
}

/// update display list
void DisplayBSpline3D::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
//    std::vector<walkers::Vec3> samples;
//    element.updateCurve(samples);
//    std::cout << "samples size " << samples.size() << "\n";
    glNewList(listId, GL_COMPILE);
        glLineWidth((GLfloat)lineWidth);
            glColor4ub(lineColor.r, lineColor.g, lineColor.b, lineColor.a);
            glBegin(GL_LINES);
            std::vector<walkers::Vec3> controlPoints = element.getControlPoints();
            if (controlPoints.size()>2){
                walkers::Vec3 p1 = controlPoints[0];
                for (size_t pointNo=0; pointNo<pointsNo;pointNo++){
//                    double t = 1.0*double(pointNo)/double(pointsNo-1);
//                    walkers::Vec3 p2 = element.compValue(t);

                    double t = ( double(pointNo) / double(pointsNo) ) * (double(controlPoints.size()) - double(element.order-1) ) + double(element.order)-1;
                    walkers::Vec3 p2 = element.compValueRecursive(t);
//                    walkers::Vec3 p2 = samples[pointNo];
                    glVertex3f(float(p1.x()), float(p1.y()), float(p1.z()));
                    glVertex3f(float(p2.x()), float(p2.y()), float(p2.z()));
                    p1=p2;
                }
                walkers::Vec3 p2 = controlPoints.back();
                glVertex3f(float(p1.x()), float(p1.y()), float(p1.z()));
                glVertex3f(float(p2.x()), float(p2.y()), float(p2.z()));
            }
            glEnd();

            if (drawCtrlPoints){
                // display control points
                glPointSize(GLfloat(ctrlPointSize));
                glBegin(GL_POINTS);
                for(walkers::Vec3 point : controlPoints) {
                    glColor3ub(0,255,0);
                    glVertex3d(point.x(), point.y(), point.z());
                }
                glEnd();
            }

            if (drawCtrlLines){
                glLineWidth((GLfloat)1);
                glColor4ub(ctrlLineColor.r, ctrlLineColor.g, ctrlLineColor.b, ctrlLineColor.a);
                glBegin(GL_LINES);
                if (controlPoints.size()>2){
                    walkers::Vec3 p1 = controlPoints[0];
                    for (size_t pointNo=1; pointNo<controlPoints.size();pointNo++){
                        walkers::Vec3 p2 = controlPoints[pointNo];
                        glVertex3f(float(p1.x()), float(p1.y()), float(p1.z()));
                        glVertex3f(float(p2.x()), float(p2.y()), float(p2.z()));
                        p1=p2;
                    }
                }
                glEnd();
            }
    glEndList();
}
