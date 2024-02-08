#include "Visualizer/Arrow.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayArrow::createDisplayList(const walkers::Arrow3D& arrow){
    element = arrow;
    updateListGL = true;
}

/// update display list
void DisplayArrow::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);

    glPushMatrix();

    glLineWidth((GLfloat)arrowWidth);

//        double len = sqrt(pow(element.x1.x()-element.x0.x(),2.0)+
//                          pow(element.x1.y()-element.x0.y(),2.0)+
//                          pow(element.x1.z()-element.x0.z(),2.0));
        glBegin(GL_LINES);

        glColor4ub(arrowColor.r, arrowColor.g, arrowColor.b, arrowColor.a);
        glVertex3f((GLfloat)element.x0.x(), (GLfloat)element.x0.y(), (GLfloat)element.x0.z());
        glVertex3f((GLfloat)element.x1.x(), (GLfloat)element.x1.y(), (GLfloat)element.x1.z());

        glEnd();
    glPopMatrix();

    glEndList();
}
