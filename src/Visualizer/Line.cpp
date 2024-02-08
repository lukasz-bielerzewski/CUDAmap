#include "Visualizer/Line.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayLine::createDisplayList(const walkers::Line3D& line){
    element = line;
    updateListGL = true;
}

/// update display list
void DisplayLine::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    double c =1000;
    glNewList(listId, GL_COMPILE);
        glLineWidth((GLfloat)lineWidth);
            glColor4ub(lineColor.r, lineColor.g, lineColor.b, lineColor.a);
            glBegin(GL_LINES);
                glVertex3f(float(element.x0.x()+element.A*c), float(element.x0.y()+element.B*c),
                           float(element.x0.z()+element.C*c));
                glVertex3f(float(element.x0.x()-element.A*c), float(element.x0.y()-element.B*c),
                           float(element.x0.z()-element.C*c));
            glEnd();
    glEndList();
}
