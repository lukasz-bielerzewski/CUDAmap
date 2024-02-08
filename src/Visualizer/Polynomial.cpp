#include "Visualizer/Polynomial.h"
#include <GL/glut.h>

/// create display list
void DisplayPolynomial::createDisplayList(double _centerX, double _centerY, double _offsetZ, double _surfWidth, double _surfLength,
                                    size_t _verticesNo, regression::Regression* poly){
    centerX = _centerX;
    centerY = _centerY;
    offsetZ = _offsetZ;
    surfWidth = _surfWidth;
    surfLength = _surfLength;
    verticesNo = _verticesNo;

    element = poly;
    updateListGL = true;
}

/// update display list
void DisplayPolynomial::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);

    double incrementX = surfWidth/double(verticesNo-1);
    double incrementY = surfLength/double(verticesNo-1);
    for (unsigned int i = 0; i < verticesNo; i++){
        for (unsigned int j = 0; j < verticesNo; j++) {
            glBegin(GL_TRIANGLE_STRIP);
            size_t rowNo; size_t colNo;
            rowNo=i; colNo=j;
            walkers::Vec3 normal(0.0,0.0,1.0);
            Eigen::MatrixXd input = Eigen::MatrixXd::Zero(1, 2);
            glNormal3d(normal.x(), normal.y(), normal.z());
            input(0,0) = -(surfWidth/2.0)+double(rowNo)*incrementX; input(0,1) = -(surfLength/2.0)+double(colNo)*incrementY;
            glColor3d(0.7+(100*element->computeOutput(input,0)), 0.7+(100*element->computeOutput(input,0)), 0.7);
            glVertex3d(centerX-(surfWidth/2.0)+double(rowNo)*incrementX,centerY-(surfLength/2.0)+double(colNo)*incrementY,element->computeOutput(input,0)+offsetZ);

            rowNo=i; colNo=j+1;
            glNormal3d(normal.x(), normal.y(), normal.z());
            input(0,0) = -(surfWidth/2.0)+double(rowNo)*incrementX; input(0,1) = -(surfLength/2.0)+double(colNo)*incrementY;
            glColor3d(0.7+(100*element->computeOutput(input,0)), 0.7+(100*element->computeOutput(input,0)), 0.7);
            glVertex3d(centerX-(surfWidth/2.0)+double(rowNo)*incrementX,centerY-(surfLength/2.0)+double(colNo)*incrementY,element->computeOutput(input,0)+offsetZ);

            rowNo=i+1; colNo=j;
            glNormal3d(normal.x(), normal.y(), normal.z());
            input(0,0) = -(surfWidth/2.0)+double(rowNo)*incrementX; input(0,1) = -(surfLength/2.0)+double(colNo)*incrementY;
            glColor3d(0.7+(100*element->computeOutput(input,0)), 0.7+(100*element->computeOutput(input,0)), 0.7);
            glVertex3d(centerX-(surfWidth/2.0)+double(rowNo)*incrementX,centerY-(surfLength/2.0)+double(colNo)*incrementY,element->computeOutput(input,0)+offsetZ);

            rowNo=i+1; colNo=j+1;
            glNormal3d(normal.x(), normal.y(), normal.z());
            input(0,0) = -(surfWidth/2.0)+double(rowNo)*incrementX; input(0,1) = -(surfLength/2.0)+double(colNo)*incrementY;
            glColor3d(0.7+(100*element->computeOutput(input,0)), 0.7+(100*element->computeOutput(input,0)), 0.7);
            glVertex3d(centerX-(surfWidth/2.0)+double(rowNo)*incrementX,centerY-(surfLength/2.0)+double(colNo)*incrementY,element->computeOutput(input,0)+offsetZ);
            glEnd();
        }
    }

    glEndList();

//    element = regression::Regression();
}
