#include "Visualizer/Triangle.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayTriangles::createDisplayList(const std::vector<walkers::Triangle3D>& triangles,
                                         const mapping::RGBA& _trianglesColor){
    element = triangles;
    triangleColor = _trianglesColor;
    updateListGL = true;
}

/// update display list
void DisplayTriangles::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
    for (const auto& elem : element){
        glColor4ub(triangleColor.r, triangleColor.g, triangleColor.b, triangleColor.a);
        glBegin(GL_TRIANGLE_STRIP);
        glNormal3d(elem.normal.x(), elem.normal.y(), elem.normal.z());
        glVertex3d(elem.vertices[0].x(), elem.vertices[0].y(), elem.vertices[0].z());
        glNormal3d(elem.normal.x(), elem.normal.y(), elem.normal.z());
        glVertex3d(elem.vertices[1].x(), elem.vertices[1].y(), elem.vertices[1].z());
        glNormal3d(elem.normal.x(), elem.normal.y(), elem.normal.z());
        glVertex3d(elem.vertices[2].x(), elem.vertices[2].y(), elem.vertices[2].z());
        glEnd();
    }
    glEndList();

    element.clear();
}
