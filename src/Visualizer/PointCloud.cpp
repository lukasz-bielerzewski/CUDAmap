#include "Visualizer/PointCloud.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayPointCloud::createDisplayList(const mapping::PointCloud& cloud){
    element = cloud;
    updateListGL = true;
}

/// update display list
void DisplayPointCloud::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
        glPointSize(GLfloat(pointSize));
        glBegin(GL_POINTS);
        for(mapping::Point3D point : element) {
            glColor3ub(point.color.r,point.color.g,point.color.b);
            glVertex3d(point.position.x(), point.position.y(), point.position.z());
        }
        glEnd();
    glEndList();

    element.clear();
}
