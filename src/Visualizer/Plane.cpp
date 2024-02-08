#include "Visualizer/Plane.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayPlanes::createDisplayList(const std::vector<walkers::Plane3D>& planes, const mapping::RGBA& _planeColor){
    element = planes;
    planeColor = _planeColor;

    updateListGL = true;
}

/// update display list
void DisplayPlanes::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    double c =1.0;
    glNewList(listId, GL_COMPILE);
    for (const auto& elem : element){
        glColor4ub(planeColor.r, planeColor.g, planeColor.b, planeColor.a);
        Eigen::Matrix3d m;
        m = Eigen::AngleAxisd(0.0, Eigen::Vector3d(elem.A, elem.B, elem.C));
        walkers::Mat34 planeMat;
        planeMat.matrix().block<3,3>(0,0) = m;
        planeMat(0,3) = elem.x0.x(); planeMat(1,3) = elem.x0.y(); planeMat(2,3) = elem.x0.z();
        walkers::Mat34 transl(walkers::Mat34::Identity());
        transl(0,3) = -c; transl(1,3) = c;
        std::vector<walkers::Mat34> quad;
        walkers::Mat34 point = planeMat*transl;
        quad.push_back(point);
//        glBegin(GL_LINES);
//        glVertex3f(float(elem.x0.x()),float(elem.x0.y()),float(elem.x0.z()));
//        glVertex3f(float(point(0,3)), float(point(1,3)), float(point(2,3)));
//        glEnd();
        transl(0,3) = c; transl(1,3) = c;
        point = planeMat*transl;
        quad.push_back(point);
//        glBegin(GL_LINES);
//        glVertex3f(float(elem.x0.x()),float(elem.x0.y()),float(elem.x0.z()));
//        glVertex3f(float(point(0,3)), float(point(1,3)), float(point(2,3)));
//        glEnd();
        transl(0,3) = c; transl(1,3) = -c;
        point = planeMat*transl;
        quad.push_back(point);
//        glBegin(GL_LINES);
//        glVertex3f(float(elem.x0.x()),float(elem.x0.y()),float(elem.x0.z()));
//        glVertex3f(float(point(0,3)), float(point(1,3)), float(point(2,3)));
//        glEnd();
        transl(0,3) = -c; transl(1,3) = -c;
        point = planeMat*transl;
        quad.push_back(point);
//        glBegin(GL_LINES);
//        glVertex3f(float(elem.x0.x()),float(elem.x0.y()),float(elem.x0.z()));
//        glVertex3f(float(point(0,3)), float(point(1,3)), float(point(2,3)));
//        glEnd();

        glBegin(GL_TRIANGLE_STRIP);
        glVertex3d(quad[0](0,3), quad[0](1,3), quad[0](2,3));
        glVertex3d(quad[1](0,3), quad[1](1,3), quad[1](2,3));
        glVertex3d(quad[2](0,3), quad[2](1,3), quad[2](2,3));
        glEnd();

        glBegin(GL_TRIANGLE_STRIP);
        glVertex3d(quad[2](0,3), quad[2](1,3), quad[2](2,3));
        glVertex3d(quad[3](0,3), quad[3](1,3), quad[3](2,3));
        glVertex3d(quad[0](0,3), quad[0](1,3), quad[0](2,3));
        glEnd();
    }
    glEndList();

    element.clear();
}
