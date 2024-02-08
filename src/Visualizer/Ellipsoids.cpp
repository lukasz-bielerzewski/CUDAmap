#include "Visualizer/Ellipsoids.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

/// draw ellipsoid
void DisplayEllipsoids::drawEllipsoid(const walkers::Vec3& pos, const walkers::Mat33& covariance, mapping::RGBA color) const{
    // ---------------------
    //    3D ellipsoid
    // ---------------------
    GLfloat		mat[16];
    Eigen::SelfAdjointEigenSolver<walkers::Mat33> es;
    es.compute(covariance);
    walkers::Mat33 m_eigVec(es.eigenvectors());

    //  A homogeneous transformation matrix, in this order:
    //
    //     0  4  8  12
    //     1  5  9  13
    //     2  6  10 14
    //     3  7  11 15
    //
    mat[3] = mat[7] = mat[11] = 0;
    mat[15] = 1;
    mat[12] = mat[13] = mat[14] = 0;

    mat[0] = (float)m_eigVec(0,0); mat[1] = (float)m_eigVec(1,0); mat[2] = (float)m_eigVec(2,0); mat[12] = (float)pos.x();// New X-axis
    mat[4] = (float)m_eigVec(0,1); mat[5] = (float)m_eigVec(1,1); mat[6] = (float)m_eigVec(2,1);	mat[13] = (float)pos.y();// New X-axis
    mat[8] = (float)m_eigVec(0,2); mat[9] = (float)m_eigVec(1,2); mat[10] = (float)m_eigVec(2,2); mat[14] = (float)pos.z();// New X-axis

    GLUquadricObj	*obj = gluNewQuadric();

    gluQuadricDrawStyle( obj, GLU_FILL);

    glPushMatrix();
    glMultMatrixf( mat );

    glScalef((float)sqrt((float)es.eigenvalues()(0)),(float)sqrt((float)es.eigenvalues()(1)),(float)sqrt((float)es.eigenvalues()(2)));
//    float reflectColor[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    //glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, reflectColor);
//    GLfloat emissiveLight[] = { 0.1f, 0.1f, 0.1f, 1.0f };
    //glMaterialfv(GL_FRONT, GL_EMISSION, emissiveLight);
    glColor4ub(color.r,color.g,color.b, color.a);
    gluSphere( obj, 1,5,5);
    glPopMatrix();

    gluDeleteQuadric(obj);
}

void DisplayEllipsoids::createDisplayList(const mapping::Ellipsoid::Seq& ellipsoids){
    element = ellipsoids;
    updateListGL = true;
}

/// update display list
void DisplayEllipsoids::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
    for( const auto& elli : element) {
        drawEllipsoid(walkers::Vec3(elli.position.x(), elli.position.y(), elli.position.z()), elli.cov, elli.color);
    }
    glEndList();

    element.clear();
}
