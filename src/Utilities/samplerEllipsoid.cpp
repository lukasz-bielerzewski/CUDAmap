/** @file recorder.cpp
*
* @author Dominik Belter
*/

#include "Utilities/samplerEllipsoid.h"
#include <random>

/// get sample
walkers::Vec3 SamplerEllipsoid::getSample(void){
    walkers::Vec3 point3D;

    walkers::Mat33 normTransform;
    Eigen::LLT<walkers::Mat33> cholSolver(ellipsoid.cov);

    static std::mt19937 gen{ std::random_device{}() };
    /// 3*sigma inside the unit sphere
    static std::normal_distribution<double> distNorm(0.0,1.0/3.0);

    // We can only use the cholesky decomposition if
    // the covariance matrix is symmetric, pos-definite.
    // But a covariance matrix might be pos-semi-definite.
    // In that case, we'll go to an EigenSolver
    if (cholSolver.info()==Eigen::Success) {
        // Use cholesky solver
        normTransform = cholSolver.matrixL();
    } else {
        // Use eigen solver
        Eigen::SelfAdjointEigenSolver<walkers::Mat33> eigenSolver(ellipsoid.cov);
        normTransform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }
    walkers::Vec3 sample(distNorm(gen), distNorm(gen), distNorm(gen));
    point3D.vector() =  ellipsoid.position.vector() + normTransform * sample.vector();
    return point3D;
}

/// get ellipsoid
mapping::Ellipsoid SamplerEllipsoid::getEllipsoid(void){
    return ellipsoid;
}

/// create from path
void SamplerEllipsoid::createFromPath(walkers::Mat34 startPose, walkers::Mat34 goalPose, double pathLength){
    double cmin = sqrt(pow(goalPose(0,3)-startPose(0,3),2.0)+pow(goalPose(1,3)-startPose(1,3),2.0));
    Eigen::Vector2d center((startPose(0,3)+goalPose(0,3))/2.0, (startPose(1,3)+goalPose(1,3))/2.0);
    // scalling matrix
    Eigen::Matrix2d S;
    S << pathLength/2.0, 0,
         0, sqrt(pow(pathLength,2.0)-pow(cmin,2.0))/2.0;
    double angle;
    if (goalPose(0,3)-startPose(0,3)==0){
        angle = 0;
    }
    else{
        angle = atan2(goalPose(1,3)-startPose(1,3),goalPose(0,3)-startPose(0,3));
    }
    Eigen::Matrix2d R;
    R << cos(angle), -sin(angle),
         sin(angle), cos(angle);

    Eigen::Matrix2d cov;
    cov = R*S*S*R.inverse();
    ellipsoid.cov << cov(0,0), cov(0,1), 0,
            cov(1,0), cov(1,1), 0,
            0, 0, 0.1;
    ellipsoid.position.x() = center(0);
    ellipsoid.position.y() = center(1);
}
