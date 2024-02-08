#include "Visualizer/Axis.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayAxis::createDisplayList(const walkers::Mat34& axisPose){
    element = axisPose;
    updateListGL = true;
}

/// update display list
void DisplayAxis::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);

    GLfloat		mat[16];
    mat[3] = mat[7] = mat[11] = 0;
    mat[15] = 1;
    mat[12] = (float)element(0,3); mat[13] = (float)element(1,3); mat[14] = (float)element(2,3);

    mat[0] = (float)element(0,0); mat[1] = (float)element(1,0); mat[2] = (float)element(2,0);
    mat[4] = (float)element(0,1); mat[5] = (float)element(1,1); mat[6] = (float)element(2,1);
    mat[8] = (float)element(0,2); mat[9] = (float)element(1,2); mat[10] = (float)element(2,2);

    glPushMatrix();
    glMultMatrixf(mat);

    glScaled(scale,scale,scale);
    glLineWidth((GLfloat)lineWidth);

        glBegin(GL_LINES);

        glColor3f (1.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(1.0, 0.0, 0.0);

        glColor3f (0.0, 1.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, 1.0, 0.0);

        glColor3f (0.0, 0.0, 1.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, 1.0);
        glEnd();
    glPopMatrix();

    glEndList();
}
