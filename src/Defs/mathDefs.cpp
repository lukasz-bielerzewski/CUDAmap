/** @file mathDefs.cpp
* @author: Dominik Belter
* Walkers math definitions
*
*/

#include "Defs/mathDefs.h"
#include <numeric>
#include <iostream>

namespace walkers{

    void Triangle3D::print(void) {
        std::cout << "Triangle vertices: \n" << vertices[0].vector().transpose() << "\n"
                  << vertices[1].vector().transpose() << "\n"
                  << vertices[2].vector().transpose() << "\n";
    }

    void Plane3D::print(void) {
        std::cout << "Plane: " << A << "*x + " << B << "*y + " << C << "*z + " << D << "\n";
    }

    double Plane3D::angleBetweenPlanes(const Plane3D& plane){
        return acos(fabs(A*plane.A+B*plane.B+C*plane.C)/(sqrt(pow(A,2.0)+pow(B,2.0)+pow(C,2.0))*sqrt(pow(plane.A,2.0)+pow(plane.B,2.0)+pow(plane.C,2.0))));
    }

    /// extract plane from three random points
    void Plane3D::fromPoints(const std::vector<walkers::Vec3>& points){
        D = points[0].x()*points[1].y()*points[2].z() + points[1].x()*points[2].y()*points[0].z() + points[2].x()*points[0].y()*points[1].z()-
                points[2].x()*points[1].y()*points[0].z() - points[0].x()*points[2].y()*points[1].z() - points[1].x()*points[0].y()*points[2].z();

        if (D==0)
            D = std::numeric_limits<double>::epsilon();
        double d=1.0;
        A = (-d/D)*(points[1].y()*points[2].z() + points[2].y()*points[0].z() + points[0].y()*points[1].z()-
                points[1].y()*points[0].z() - points[2].y()*points[1].z() - points[0].y()*points[2].z());
        B = (-d/D)*(points[0].x()*points[2].z() + points[1].x()*points[0].z() + points[2].x()*points[1].z()-
                points[2].x()*points[0].z() - points[0].x()*points[1].z() - points[1].x()*points[2].z());
        C = (-d/D)*(points[0].x()*points[1].y() + points[1].x()*points[2].y() + points[2].x()*points[0].y()-
                points[2].x()*points[1].y() - points[0].x()*points[2].y() - points[1].x()*points[0].y());

        D -=A*points[0].x()+B*points[0].y()+C*points[0].z()+D;
    }

    void Line3D::print(void) {
        std::cout << "Line: (x-" << x0.x() << ")/" << A << " + " << "(y-" << x0.y() << ")/" << B << " + " << "(z-" << x0.z() << ")/" << C << "\n";
    }

    /// line from Mat34 (rotation mat and point)
    void Line3D::fromPose(const walkers::Mat34& pose){
        this->A = pose(0,2);
        this->B = pose(1,2);
        this->C = pose(2,2);
        x0.x() = pose(0,3);
        x0.y() = pose(1,3);
        x0.z() = pose(2,3);
    }

    void Arrow3D::print(void) {
        std::cout << "Arrow: (" << x0.x() << ", " << x0.y() << ", " << x0.z() << "), (" <<
                     x1.x() << ", " << x1.y() << ", " << x1.z() << "\n";
    }

    /// t should be in range <0,1>
    walkers::Vec3 BSpline3D::compValue(double t){
        throw std::runtime_error("BSplineSE3: comValue method not implemented!\n");
        walkers::Vec3 point;
        if (controlPoints.size()<3)
            return point;
        double s = t*double(controlPoints.size()-1);// s in range 0,...,order-1
        size_t iter = int(s);
        double u = s-double(iter);//normalized time for the segment
        Eigen::MatrixXd points(3,order);
        points = Eigen::MatrixXd::Zero(3,order);
        size_t colNo=0;
        for (size_t pointNo=iter;(pointNo<order)&&(pointNo<controlPoints.size());pointNo++){
            for (size_t dim=0;dim<3;dim++){
                points(dim,colNo) = controlPoints[pointNo].vector()(dim);
            }
            colNo++;
        }
        Eigen::MatrixXd vec_u(order,1);
        for (size_t k=0;k<order;k++){
            vec_u(k)=double(pow(double(u),double(k)));
        }
        point.vector() = points * M * vec_u;
//        point = controlPoints[0];
//        for (size_t pointNo=1; pointNo<controlPoints.size(); pointNo++){
//            double alpha=0.0;
//            if (t<double(pointNo)/double(controlPoints.size()-1))
//                alpha = 0.0;
//            else if (t>=double(pointNo)/double(controlPoints.size()-1) && t<double(pointNo+1)/double(controlPoints.size()-1))
//                alpha = t-double(pointNo)/double(controlPoints.size()-1);
//            else if (t>=double(pointNo+1)/double(controlPoints.size()-1))
//                alpha = 1.0;
//            point.vector()+= alpha*(controlPoints[pointNo].vector()-controlPoints[pointNo-1].vector());
//        }
        return point;
    }

    /// t should be in range 1,...,pointsNo
    walkers::Vec3 BSpline3D::compValueRecursive(double t){
        walkers::Vec3 newPoint(0,0,0);
        bool isWeight(false);
        for(int i=0; i < int(controlPoints.size()); i++) {
            double weight = calculateWeightForPoint(i, int(order), int(controlPoints.size()), t);
            if (weight>0)
                isWeight = true;
            newPoint.vector() += weight * controlPoints.at(i).vector();
        }
        if (!isWeight)
            newPoint = controlPoints.back();
        return newPoint;
    }

    /// t should be in range <0,1>
    walkers::Mat34 BSplineSE3::compValue(double t){
        throw std::runtime_error("BSplineSE3: comValue method not implemented!\n");
        (void) t;
        walkers::Mat34 pose;
        if (controlPoints.size()<3)
            return pose;
        return pose;
    }

//    double BSplineSE3::interValueAbs(Eigen::Vector4d values, double u) {

//        walkers::Mat34 bSplineMatrix;
//        bSplineMatrix.matrix() << 1, -3, 3, -1,  4, 0, -6, 3,   1, 3, 3, -3,    0, 0, 0, 1;
//        double bSplineCoeff = 1.0 / 6;

//        Eigen::Matrix<double, 4, 1> vecU;
//        vecU << 1, u, u*u, u*u*u;

//        return values.transpose() * bSplineCoeff * bSplineMatrix.matrix() * vecU;
//    }

//    double BSplineSE3::interValueCum(Eigen::Vector4d values, double u) {

//        walkers::Mat34 bSplineMatrix;
//        bSplineMatrix.matrix() <<  6, 0, 0, 0,    5, 3, -3, 1,    1, 3, 3, -2,    0, 0, 0, 1;
//        double bSplineCoeff = 1.0 / 6;

//        Eigen::Matrix<double, 4, 1> vecU;
//        vecU << 1, u, u*u, u*u*u;

//        values(3) = values(3) - values(2);
//        values(2) = values(2) - values(1);
//        values(1) = values(1) - values(0);

//        return values.transpose() * bSplineCoeff * bSplineMatrix.matrix() * vecU;
//    }

//    Eigen::Vector3d BSplineSE3::interSO3Cum(Eigen::Matrix<double, 3, 4> & values, double u) {
//        // Preparing bSpline (could be done onve if time is an issue)
//        Eigen::Matrix<double, 4, 4> bSplineMatrix;
//        bSplineMatrix <<  6, 0, 0, 0,    5, 3, -3, 1,    1, 3, 3, -2,    0, 0, 0, 1;
//        double bSplineCoeff = 1.0 / 6;

//        Eigen::Matrix<double, 4, 1> vecU;
//        vecU << 1, u, u*u, u*u*u;

//        Eigen::Matrix<double, 4, 1> b = bSplineCoeff * bSplineMatrix * vecU;

//        // Computing needed diffs in Lie Algebra
//        Eigen::Vector3d val0 = values.col(0);
//        Eigen::Vector3d val1 = values.col(1);
//        Eigen::Vector3d val2 = values.col(2);
//        Eigen::Vector3d val3 = values.col(3);
//        Eigen::Vector3d tmp;
//        Eigen::Matrix3d tmpMat, sum = Eigen::Matrix3d::Identity();

//        // 0
//        sum = sum * walkers::expmap(walkers::Vec3(val0(0),val1(0),val2(0))); // * b(0) == 1

//        // 1
//        tmpMat = walkers::expmap(walkers::Vec3(val0(0),val0(1),val0(2))).transpose()
//                * walkers::expmap(walkers::Vec3(val1(0),val1(1),val1(2)));
//        tmp = b(1) * walkers::logmap(tmpMat);
//        sum = sum * walkers::expmap(walkers::Vec3(tmp(0),tmp(1),tmp(2)));

//        // 2
//        tmpMat = LieAlgebra::exp(val1).transpose() * LieAlgebra::exp(val2);
//        tmp = b(2) * LieAlgebra::log(tmpMat);
//        sum = sum * LieAlgebra::exp(tmp);

//        // 3
//        tmpMat = LieAlgebra::exp(val2).transpose() * LieAlgebra::exp(val3);
//        tmp = b(3) * LieAlgebra::log(tmpMat);
//        sum = sum * LieAlgebra::exp(tmp);

//        // Final estimated rotation
//        return LieAlgebra::log(sum);
//    }

//    Eigen::Matrix4d BSplineSE3::interTSO3Cum(Eigen::Matrix4d & Ta, Eigen::Matrix4d & Tb, Eigen::Matrix4d & Tc, Eigen::Matrix4d & Td, double u) {

//        Eigen::Matrix4d interT = Eigen::Matrix4d::Identity();
//        for (int idx = 0; idx < 3; idx ++) {
//            Eigen::Vector4d vals;
//            vals << Ta(idx, 3), Tb(idx, 3), Tc(idx, 3), Td(idx, 3);
//            interT(idx, 3) = BSpline::interValueCum(vals, u);
//        }

//        Eigen::Matrix<double, 3, 4> test;
//        test.col(0) = LieAlgebra::log(Ta.block<3,3>(0,0));
//        test.col(1) = LieAlgebra::log(Tb.block<3,3>(0,0));
//        test.col(2) = LieAlgebra::log(Tc.block<3,3>(0,0));
//        test.col(3) = LieAlgebra::log(Td.block<3,3>(0,0));

//        Eigen::Vector3d interR = BSpline::interSO3Cum(test, u);
//        interT.block<3,3>(0,0) = LieAlgebra::exp(interR);
//        return interT;
//    }

    /// t should be in range 1,...,pointsNo
    walkers::Mat34 BSplineSE3::compValueRecursive(double t){
        walkers::Mat34 newPose(controlPoints.back());
        walkers::Mat34 prevPose;
        int frameStart = 0;//int(t-0);
        newPose = controlPoints[frameStart];
        prevPose = controlPoints[frameStart];
        // compute position
        walkers::Vec3 newPosition(0,0,0);
        bool isWeight(false);
        for(int i=0; i < int(controlPoints.size()); i++) {
            double weight = calculateWeightForPoint(i, int(order), int(controlPoints.size()), t);
            if (weight>0)
                isWeight = true;
            walkers::Vec3 controlPosition(controlPoints.at(i)(0,3),controlPoints.at(i)(1,3),controlPoints.at(i)(2,3));
            newPosition.vector() += weight * controlPosition.vector();
        }
        if (!isWeight)
            newPosition = walkers::Vec3(controlPoints.back()(0,3),controlPoints.back()(1,3),controlPoints.back()(2,3));

        // compute orientation
        double prevWeight=0;
        isWeight = false;
        int basePoseNo=0;
        for(int i=frameStart; i < int(controlPoints.size()); i++) {
            double weight = calculateWeightForPoint(i, int(order), int(controlPoints.size()), t);
            walkers::Vec3 controlPosition(controlPoints.at(i)(0,3),controlPoints.at(i)(1,3),controlPoints.at(i)(2,3));
            if (weight>0&&prevWeight==0){
                basePoseNo = i;
                newPose = prevPose;
            }
            if (weight>0)
                isWeight = true;
            // compute motion
            walkers::Mat33 motionRot;// = basePose.rotation().inverse()*poseRot;
            // convert motion to 6D vector
            walkers::Vec3 motionRotV(0,0,0);// = walkers::logmap(motionRot);
            if (weight>0){// crucial part -- accumulate motion using logmap
                size_t startP = basePoseNo-1 >=0 ? basePoseNo-1 : 0;
                for (size_t frameNo=startP;frameNo<size_t(i);frameNo++){// compute motion (accumulate rotation)
                    motionRot = controlPoints.at(frameNo).rotation().inverse()*controlPoints.at(frameNo+1).rotation();
                    motionRotV.vector()+=walkers::logmap(motionRot).vector();
                }
                motionRot = walkers::expmap(motionRotV);
            }
            walkers::Vec3 weightedMotionVec;
            weightedMotionVec.vector() = weight * motionRotV.vector();
            walkers::Mat34 scaledMotion;
            scaledMotion = walkers::expmap(weightedMotionVec);
            if (weight>0)
                newPose = newPose*scaledMotion;
            prevPose = controlPoints.at(i);
            prevWeight = weight;
        }
        if (!isWeight)
            newPose = controlPoints.back();
        newPose(0,3) = newPosition.x();
        newPose(1,3) = newPosition.y();
        newPose(2,3) = newPosition.z();

        return newPose;
    }

    /// get trajectory
    std::vector<walkers::Mat34> BSplineSE3::getTrajectory(size_t pointsNo){
        std::vector<walkers::Mat34> trajectory(pointsNo);
        if (controlPoints.size()>2){
            for (size_t pointNo=0; pointNo<pointsNo;pointNo++){
                double t = ( double(pointNo) / double(pointsNo) ) * (double(controlPoints.size()) - double(order-1) ) + double(order)-1;
                walkers::Mat34 p = compValueRecursive(t);
                trajectory[pointNo]=p;
            }
        }
        return trajectory;
    }

    /// balance matrix
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
}
