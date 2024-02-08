#include "Visualizer/MapNDTOM.h"
#include <GL/glut.h>

void DisplayMapNDTOM::createDisplayList(const Octree<mapping::Voxel>& mapNDTOM,
                                        std::unordered_map<std::string, Eigen::Vector3i> _updatedVoxels){
    element = mapNDTOM;
    updatedVoxels = _updatedVoxels;

    updateListGL = true;
}

void balance_matrix(const walkers::Mat33& A, walkers::Mat33& Aprime, walkers::Mat33& D) {
    // https://arxiv.org/pdf/1401.5766.pdf (Algorithm #3)
    const int p = 2;
    double beta = 2; // Radix base (2?)
    Aprime = A;
    D = Eigen::MatrixXd::Identity(A.rows(), A.cols());
    bool converged = false;
    do {
        converged = true;
        for (int i = 0; i < A.rows(); ++i) {
            double c = Aprime.col(i).lpNorm<p>();
            double r = Aprime.row(i).lpNorm<p>();
            double s = pow(c, p) + pow(r, p);
            double f = 1;
            while (c < r / beta) {
                c *= beta;
                r /= beta;
                f *= beta;
            }
            while (c >= r*beta) {
                c /= beta;
                r *= beta;
                f /= beta;
            }
            if (pow(c, p) + pow(r, p) < 0.95*s) {
                converged = false;
                D(i, i) *= f;
                Aprime.col(i) *= f;
                Aprime.row(i) /= f;
            }
        }
    } while (!converged);
}

/// draw ellipsoid
void DisplayMapNDTOM::drawEllipsoid(const walkers::Vec3& pos, const walkers::Mat33& covariance, mapping::RGBA color) const{
    // ---------------------
    //    3D ellipsoid
    // ---------------------
    GLfloat		mat[16];
    Eigen::EigenSolver<walkers::Mat33> es;

    walkers::Mat33 Aprime; walkers::Mat33 D;
//    balance_matrix(covariance, Aprime, D);
//    es.compute(Aprime);
    es.compute(covariance);
    Eigen::MatrixXcd m_eigVec(es.eigenvectors());
    //  A homogeneous transformation matrix, in this order:
    //
    //     0  4  8  12
    //     1  5  9  13
    //     2  6  10 14
    //     3  7  11 15
    //
    mat[3] = mat[7] = mat[11] = 0; mat[15] = 1;

    mat[0] = (float)m_eigVec(0,0).real(); mat[1] = (float)m_eigVec(1,0).real(); mat[2] = (float)m_eigVec(2,0).real(); mat[12] = (float)pos.x();// New X-axis
    mat[4] = (float)m_eigVec(0,1).real(); mat[5] = (float)m_eigVec(1,1).real(); mat[6] = (float)m_eigVec(2,1).real(); mat[13] = (float)pos.y();// New X-axis
    mat[8] = (float)m_eigVec(0,2).real(); mat[9] = (float)m_eigVec(1,2).real(); mat[10] = (float)m_eigVec(2,2).real(); mat[14] = (float)pos.z();// New X-axis

    GLUquadricObj *obj = gluNewQuadric();

    gluQuadricDrawStyle( obj, GLU_FILL);
    double eigenvalues[3] = {(float)sqrt((float)es.eigenvalues()[0].real()),(float)sqrt((float)es.eigenvalues()[1].real()),
                             (float)sqrt((float)es.eigenvalues()[2].real())};
    bool isNAN = false;
    for (auto& v : eigenvalues){
        if (std::isnan(v))
            isNAN = true;
    }
    if (isNAN){
        eigenvalues[0] = sqrt(covariance(0,0));
        eigenvalues[1] = sqrt(covariance(1,1));
        eigenvalues[2] = sqrt(covariance(2,2));
        mat[0] = 1; mat[1] = 0; mat[2] = 0;
        mat[4] = 0; mat[5] = 1; mat[6] = 0;
        mat[8] = 0; mat[9] = 0; mat[10] = 1;
    }
    glPushMatrix();
    glMultMatrixf( mat );

    glScalef((float)eigenvalues[0], (float)eigenvalues[1], (float)eigenvalues[2]);
    //    float reflectColor[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    //glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, reflectColor);
    //    GLfloat emissiveLight[] = { 0.1f, 0.1f, 0.1f, 1.0f };
    //glMaterialfv(GL_FRONT, GL_EMISSION, emissiveLight);
    glColor4ub(color.r,color.g,color.b, color.a);
    gluSphere( obj, 1,25,25);
    glPopMatrix();
    gluDeleteQuadric(obj);
}

/// update display list
void DisplayMapNDTOM::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
    mapping::Ellipsoid::Seq ellipsoids;
    for( const auto& n : updatedVoxels ) {
        Eigen::Vector3i indexes = n.second;
        mapping::Voxel v = element(indexes.x(), indexes.y(), indexes.z());
        //            if(v.probability > probabilityTreshold) {
        //        if (v.var.determinant()!=1){
            mapping::Ellipsoid eli;
            eli.cov = v.var;    eli.color = v.color;
//                        eli.cov(0,0)=0.0005;
//                        eli.cov(0,1)=0.0;
//                        eli.cov(0,2)=0.0;

//                        eli.cov(1,0)=0.0;
//                        eli.cov(1,1)=0.0005;
//                        eli.cov(1,2)=0.0;

//                        eli.cov(2,0)=0.0;
//                        eli.cov(2,1)=0.0;
//                        eli.cov(2,2)=0.0005;
            eli.position.x() = v.mean(0); eli.position.y() = v.mean(1); eli.position.z() = v.mean(2);
            if (v.probability>0.1)
                ellipsoids.push_back(eli);
        //        }
        //            }
    }
    for( const auto& elli : ellipsoids) {
        drawEllipsoid(walkers::Vec3(elli.position.x(), elli.position.y(), elli.position.z()), elli.cov, elli.color);
    }
    glEndList();

    element = Octree<mapping::Voxel>(8);
}
