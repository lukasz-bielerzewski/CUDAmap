#include "Mapping/voxel.h"

/*
 * Voxel methods
 */

using namespace mapping;

Voxel::Voxel() {
    probability = 0.0;
    sampNumber = 0;
    mean = Eigen::Vector3d(0, 0, 0);
    var << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    sampMean = Eigen::Vector3d(0, 0, 0);
    color = RGBA(80, 80, 80, 80);
    points = {};
    numPoints = 0;
    for(int i =0; i<9;i++) {
        for(int j =0; j<9;j++) {
            if(i==j) {
                P_values[9*i+j] = 1;
            }
        }
    }
//    kalmanFilter.loadConfiguration("KalmanVoxel.xml");
}

Voxel::Voxel(int res) {
    (void)res;
    probability = 0.0;
    sampNumber = 0;
    mean = Eigen::Vector3d(0, 0, 0);
    var << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    sampMean = Eigen::Vector3d(0, 0, 0);
    color = RGBA(80, 80, 80, 80);
    points = {};
    numPoints = 0;
    for(int i =0; i<9;i++) {
        for(int j =0; j<9;j++) {
            if(i==j) {
                P_values[9*i+j] = 1;
            }
        }
    }
//    kalmanFilter.loadConfiguration("KalmanVoxel.xml");
}

void Voxel::preinitParameters(double res, Eigen::Vector3d center) {
    this->mean = center;
    var << res/10, 0, 0, 0, res/10, 0, 0, 0, res/10;
}

void Voxel::insertPoint(Point3D point) {
    this->points.push_back(point);
    this->numPoints++;
    updateOccupancy();
}

void Voxel::updateWithSimpleMethod(mapping::updateMethodType updateType, int pointThreshold) {
    if((int)points.size() > pointThreshold) {
        if (updateType == updateMethodType::TYPE_SIMPLE) {
            mean = Eigen::Vector3d(0, 0, 0);
            var << 0, 0, 0, 0, 0, 0, 0, 0, 0;
            color = RGBA(255, 255, 255, 255);
            Eigen::Vector3d meanColor;
            updateSimpleDistribution(mean, var, meanColor);
            updateSimpleColor();
        } else if (updateType == updateMethodType::TYPE_BAYES) {
            updateBayesDistribution();
            updateNaiveColor();
            points.clear();
            //uncertaintyErrors.clear();
        } else if (updateType == updateMethodType::TYPE_KALMAN) {
            updateKalmanDistribution();
            updateNaiveColor();
            points.clear();
            //uncertaintyErrors.clear();
        } else if (updateType == updateMethodType::TYPE_NDTOM)
        {
            updateNDTOM();
            //CUDA_UpdateNDTOM();
            updateNaiveColor();
            points.clear();
        }
    }
    else {
        probability -= 0.001*(int)points.size();
    }
}

void Voxel::updateSimpleDistribution(Eigen::Vector3d& meanPos, walkers::Mat33& variance, Eigen::Vector3d& meanColor) {
    meanPos.setZero();
    meanColor.setZero();
    for(mapping::Point3D &point : points) {
        Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
        meanPos += newPoint;
        Eigen::Vector3d newcolor = Eigen::Vector3d(point.color.r, point.color.g, point.color.b);
        meanColor += newcolor;
    }
    meanPos /= double(points.size());
    meanColor /= double(points.size());
    if(points.size() > 1) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                variance(i,j) = 0.0;
                for(const mapping::Point3D &point : points) {
                    Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
                    variance(i,j) += (meanPos(i) - newPoint(i)) * (meanPos(j) - newPoint(j));
                }
                variance(i,j) /= double(points.size() - 1);
            }
        }
    }
    variance = 4*variance;
}

void Voxel::updateSimpleColor() {
    size_t r,g,b,a;
    r=g=b=a=0;
    for(mapping::Point3D &point : points) {
        r += point.color.r;
        g += point.color.g;
        b += point.color.b;
        a += point.color.a;
    }

    this->color.r = (uint8_t) (r/points.size());
    this->color.g = (uint8_t) (g/points.size());
    this->color.b = (uint8_t) (b/points.size());
    this->color.a = (uint8_t) (a/points.size());
}

walkers::Mat33 Voxel::prostuj(walkers::Mat33 R) {
    if(R(2,2) < 0)
        R = -R;
    if(R(0,0) < 0)
        R.col(0) = -R.col(0);
    if(R.determinant() < 0)
        R.col(1) = -R.col(1);

    return R;
}

std::tuple<walkers::Mat33, Eigen::Vector3d> Voxel::changeOrder(walkers::Mat33 Rot, Eigen::Vector3d S) {
    walkers::Mat33 I = walkers::Mat33::Identity();
    walkers::Mat33 newRot = walkers::Mat33::Zero();
    Eigen::Vector3d newS(0,0,0);
    Eigen::Vector3i index(-1,-1,-1);

    for(int i = 0; i < 3; i++) {
        double angle = 190;
        for(int j = 0; j < 3; j++) {
            double n = I.col(j).cross(Rot.col(i)).norm();
            double d = I.col(j).dot(Rot.col(i));
            double a = std::atan2(n, d) * 180/M_PI;

            if(std::norm(a) < angle || std::norm(a-180) < angle) {

                if (index(0) != j && index(1) != j &&  index(2) != j ) {

                    if (std::norm(a) < std::norm(a-180))
                        angle = std::norm(a);
                    else
                        angle = std::norm(a-180);
                    index(i) = j;
                    newRot.col(j) = Rot.col(i);
                    newS(j) = S(i);
                }

            }
        }
    }

    Eigen::Vector3d tmp = newRot.col(2).cross(newRot.col(0));
    double n = tmp.cross(newRot.col(1)).norm();
    double d = tmp.dot(newRot.col(1));
    double a = std::atan2(n, d) * 180/M_PI;

    if(std::norm(a) > 90)
        newRot.col(1) = -newRot.col(1);

    return  std::make_tuple(newRot, newS);
}

Eigen::Vector3d Voxel::castVector(walkers::Mat33 Rot, Eigen::Vector3d S) {
    Eigen::Vector3d newS;
    newS << 0,0,0;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            if (std::norm(newS(i)) < std::norm(Rot(i,j)*S(j)))
                newS(i) = std::norm(Rot(i,j)*S(j));
        }
    }

    return newS;
}

void Voxel::updateNaiveColor() {
    size_t r,g,b,a;
    int previousSampNumber = int(sampNumber - points.size());
    r = color.r * previousSampNumber;
    g = color.g * previousSampNumber;
    b = color.b * previousSampNumber;
    a = color.a * previousSampNumber;
    for(mapping::Point3D &point : points) {
        r += point.color.r;
        g += point.color.g;
        b += point.color.b;
        a += point.color.a;
    }

    this->color.r = (uint8_t) (r/sampNumber);
    this->color.g = (uint8_t) (g/sampNumber);
    this->color.b = (uint8_t) (b/sampNumber);
    this->color.a = (uint8_t) (a/sampNumber);
}


void Voxel::updateBayesDistribution() {

    if (sampNumber == 0) {
        mean = Eigen::Vector3d(0, 0, 0);
        var << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        color = RGBA(255, 255, 255, 255);
        sampNumber = (int)points.size();
        if (sampNumber == 1) {
            mean = Eigen::Vector3d(points[0].position.x(), points[0].position.y(), points[0].position.z());
            //var = uncertaintyErrors[0];
        } else {
            Eigen::Vector3d meanColor;
            updateSimpleDistribution(mean, var, meanColor);
            color.r = (uint8_t)meanColor(0); color.g = (uint8_t)meanColor(1); color.b = (uint8_t)meanColor(2);
        }
    } else {
        Eigen::Vector3d newMean = Eigen::Vector3d(0, 0, 0);
        for(mapping::Point3D &point : points) {
            Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
            newMean += newPoint;
        }
        newMean = newMean / double(points.size());

        walkers::Mat33 newVar;
        newVar << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        if(points.size() > 1) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    newVar(i,j) = 0.0;
                    for(mapping::Point3D &point : points) {
                        Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
                        newVar(i,j) += (newMean(i) - newPoint(i)) * (newMean(j) - newPoint(j));
                    }
                    newVar(i,j) /= double(points.size() - 1);
                }
            }
        }
        newVar = 4*newVar;

        Eigen::JacobiSVD<walkers::Mat33> svdVar(var, Eigen::ComputeFullU);
        Eigen::JacobiSVD<walkers::Mat33> svdNewVar(newVar, Eigen::ComputeFullU);
        walkers::Mat33 U = svdVar.matrixU();
//        walkers::Mat33 Un = svdNewVar.matrixU();
        Eigen::Vector3d S = svdVar.singularValues();
        Eigen::Vector3d Sn = svdNewVar.singularValues();

        walkers::Mat33 U0 = walkers::Mat33::Identity();//U.inverse()*U;
//        walkers::Mat33 Un0 = U.inverse()*Un;

        walkers::Vec3 u = walkers::logmap(U0);
        Eigen::VectorXd x0(9);
        x0 << mean(0), mean(1), mean(2), S(0), S(1), S(2), u.x(), u.y(), u.z();

        //walkers::Vec3 un = walkers::logmap(Un0);
        Eigen::VectorXd x0n(9);
        x0n << newMean(0), newMean(1), newMean(2), Sn(0), Sn(1), Sn(2), u.x(), u.y(), u.z();

        Eigen::VectorXd xxx(9);

        xxx = x0 * sampNumber / (sampNumber + points.size()) + x0n * points.size() / (sampNumber + points.size());

        Eigen::MatrixXd c = Eigen::MatrixXd::Identity(9, 9);
        sampNumber += (unsigned int)points.size();
        Eigen::VectorXd ux(9);
        ux = xxx;


        Eigen::VectorXd mp(9);
        Eigen::MatrixXd cp = Eigen::MatrixXd::Identity(9, 9);
        Eigen::MatrixXd P_pre(9,9);
        for(int i =0; i<9;i++) {
            for(int j =0; j<9;j++) {
                P_pre(i,j) = P_values[9*i+j];
            }
        }
        cp = (P_pre.inverse() + sampNumber*(c.inverse())).inverse();
        mp = cp*(sampNumber*(c.inverse())*ux + (P_pre.inverse())*x0);
        for(int i =0; i<9;i++) {
            for(int j =0; j<9;j++) {
                P_values[9*i+j] = cp(i,j);
            }
        }

        mean << mp(0), mp(1), mp(2);
        Eigen::Vector3d post_s;
        post_s << mp(3),mp(4),mp(5);
        Eigen::Vector3d post_r;
        post_r << mp(6),mp(7),mp(8);

        walkers::Mat33 postS;
        postS << post_s(0), 0, 0, 0,  post_s(1), 0, 0, 0, post_s(2);
        walkers::Mat33 postR;
        postR = walkers::expmap(walkers::Vec3(post_r(0), post_r(1), post_r(2)));
        postR = U*postR;
        var = postR * postS * postR.transpose();

    }
}

void Voxel::getVoxelState(Eigen::Vector3d& _mean, walkers::Mat33& _var, double& _probability, size_t& _sampNumber, Eigen::Vector3d& _sampMean, RGBA& _color, Eigen::Vector3d& _meanSum, walkers::Mat33& _varSum) const{
    _mean = mean;
    _var=var;
    _probability = probability;
    _sampNumber = sampNumber;
    _sampMean = sampMean;
    _color = color;
    _meanSum = meanSum;
    _varSum = varSum;
}

void Voxel::setVoxelState(const Eigen::Vector3d& _mean, const walkers::Mat33& _var, const double& _probability, const unsigned int& _sampNumber, const Eigen::Vector3d& _sampMean, const RGBA& _color, const Eigen::Vector3d& _meanSum, const walkers::Mat33& _varSum){
    mean = _mean;
    var=_var;
    probability = _probability;
    sampNumber = _sampNumber;
    sampMean = _sampMean;
    color = _color;
    meanSum = _meanSum;
    varSum = _varSum;
}

void Voxel::updateKalmanDistribution() {
    if (sampNumber == 0) {
        mean = Eigen::Vector3d(0, 0, 0);
        var << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        color = RGBA(255, 255, 255, 255);
        sampNumber = (int)points.size();
        if (sampNumber == 1) {
            mean = Eigen::Vector3d(points[0].position.x(), points[0].position.y(), points[0].position.z());
            //var = uncertaintyErrors[0];
        } else {
            Eigen::Vector3d meanColor(0,0,0);
            updateSimpleDistribution(mean, var, meanColor);
            color.r = (uint8_t)meanColor(0); color.g = (uint8_t)meanColor(1); color.b = (uint8_t)meanColor(2);
            walkers::Mat33 eigenvectorsMat;
            walkers::Mat33 eigenValues;
            mapping::computePCA(points, eigenvectorsMat, eigenValues);
            prevRot = eigenvectorsMat;
            Eigen::VectorXd x0(Eigen::VectorXd::Zero(12)); // initial state estimation
            //mean value
            x0(0) = mean(0); x0(1) = mean(1); x0(2) = mean(2);
            //rot
            walkers::Vec3 rotParam = walkers::logmap(eigenvectorsMat);
            x0(3) = rotParam.x(); x0(4) = rotParam.y(); x0(5) = rotParam.z();
            estRot.x() = rotParam.x(); estRot.y() = rotParam.y(); estRot.z() = rotParam.z();
            //width
            estCov.x() = eigenValues(0,0); estCov.y() = eigenValues(1,1); estCov.z() = eigenValues(2,2);
            x0(6) = eigenValues(0,0); x0(7) = eigenValues(1,1); x0(8) = eigenValues(2,2);
            //color
            x0(9) = meanColor(0); x0(10) = meanColor(1); x0(11) = meanColor(2);
//            kalmanFilter.init(0.0,x0);
        }
    } else {
//        mean = Eigen::Vector3d(0, 0, 0);
//        var << 0, 0, 0, 0, 0, 0, 0, 0, 0;
//        color = RGBA(255, 255, 255, 255);
        sampNumber = (int)points.size();
        Eigen::Vector3d meanColor(0,0,0);
//        std::cout << "prev mean " << mean(0) << ", " << mean(1) << ", " << mean(2) << "\n";
//        std::cout << "prev color " << (int)color.r << ", " << (int)color.g << ", " << (int)color.b << "\n";
//        std::cout << "prev cov\n" << var.matrix() << "\n";
//        walkers::Mat33 prevRottmp;
//        walkers::Mat33 eigTmp;
//        computeEigen(var,prevRottmp,eigTmp);
//        std::cout <<"prev rot tmp\n" << prevRottmp << "\n";
//        std::cout <<"prev rot\n" << prevRot << "\n";
//        std::cout <<"prev eig\n" << eigTmp << "\n";
//        walkers::Vec3 rotParamPrev = walkers::logmap(prevRot);
//        std::cout << "param prev " << rotParamPrev.vector() << "\n";

//        updateSimpleDistribution(mean, var, meanColor);
//        color.r = (uint8_t)meanColor(0); color.g = (uint8_t)meanColor(1); color.b = (uint8_t)meanColor(2);
//        computeEigen(var,rotTmp,eigTmp);

        walkers::Mat33 eigenvectorsMat;
        walkers::Mat33 eigenValues;
        mapping::computePCA(points, prevRot, eigenvectorsMat, eigenValues);
//        std::cout <<"next rot\n" << eigenvectorsMat << "\n";
//        std::cout <<"next eig\n" << eigenValues << "\n";
//        walkers::Vec3 rotParamNext = walkers::logmap(eigenvectorsMat);
//        std::cout << "param next " << rotParamNext.vector() << "\n";
        walkers::Mat33 motion = prevRot.inverse()*eigenvectorsMat;
//        std::cout << "motion \n" << motion << "\n";
        walkers::Vec3 motionParam = walkers::logmap(motion);
//        if (motionParam.vector().norm()<1.0){
//            std::cout << "motion param\n" << motionParam.vector() << "\n";
//            std::cout << prevRot << "\n";
//            std::cout << eigenvectorsMat << "\n";
//            getchar();
//    }

//        std::cout << "update mean " << mean(0) << ", " << mean(1) << ", " << mean(2) << "\n";
//        std::cout << "update color " << (int)color.r << ", " << (int)color.g << ", " << (int)color.b << "\n";
//        std::cout << "update cov\n" << var.matrix() << "\n";

        Eigen::VectorXd x(Eigen::VectorXd::Zero(12)); // state measurement
//        x = kalmanFilter.state();
//        std::cout << "prev state: " << x.transpose() << "\n";
        //mean
        x(0) = mean(0); x(1) = mean(1); x(2) = mean(2);
//        walkers::Vec3 rotParam = walkers::logmap(eigenvectorsMat);
        //rot
//        x(3) = rotParam.x(); x(4) = rotParam.y(); x(5) = rotParam.z();
        x(3) += motionParam.x(); x(4) += motionParam.y(); x(5) += motionParam.z();
        estRot.x() = x(3); estRot.y() = x(4); estRot.z() = x(5);
        //width
        estCov.x() = eigenValues(0,0); estCov.y() = eigenValues(1,1); estCov.z() = eigenValues(2,2);
        x(6) = eigenValues(0,0); x(7) = eigenValues(1,1); x(8) = eigenValues(2,2);
        //color
        x(9) = meanColor(0); x(10) = meanColor(1); x(11) = meanColor(2);// something is wrong with the color

        Eigen::MatrixXd Rmat(12,12);
        Rmat.setIdentity();
        Rmat *=1/double(points.size());
//        kalmanFilter.predict();
//        kalmanFilter.update(x, 0.1);
//        std::cout << "upd state: " << x.transpose() << "\n";
//        x = kalmanFilter.state();
//        std::cout << "new state: " << x.transpose() << "\n";
        mean(0) = x(0); mean(1) = x(1); mean(2) = x(2);
        estRot.x() = x(3); estRot.y() = x(4); estRot.z() = x(5);
        walkers::Mat33 rotCov = walkers::expmap(estRot);
        prevRot = rotCov;
        walkers::Mat33 scale(walkers::Mat33::Identity());
        scale(0,0) = x(6); scale(1,1) = x(7); scale(2,2) = x(8);
        var = rotCov * scale * scale * rotCov.inverse();
        color.r = (uint8_t)x(9); color.g = (uint8_t)x(10); color.b = (uint8_t)x(11);
//        std::cout << "new mean " << mean(0) << ", " << mean(1) << ", " << mean(2) << "\n";
//        std::cout << "new color " << (int)color.r << ", " << (int)color.g << ", " << (int)color.b << "\n";
//        std::cout << "new cov\n" << var.matrix() << "\n";
//        getchar();

//        Eigen::Vector3d newMean = Eigen::Vector3d(0, 0, 0);
//        for(mapping::Point3D &point : points) {
//            Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
//            newMean += newPoint;
//        }
//        newMean = newMean / points.size();

//        walkers::Mat33 newVar;
//        newVar << 0, 0, 0, 0, 0, 0, 0, 0, 0;
//        if(points.size() > 1) {
//            for (int i = 0; i < 3; i++) {
//                for (int j = 0; j < 3; j++) {
//                    newVar(i,j) = 0.0;
//                    for(mapping::Point3D &point : points) {
//                        Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
//                        newVar(i,j) += (newMean(i) - newPoint(i)) * (newMean(j) - newPoint(j));
//                    }
//                    newVar(i,j) /= points.size() - 1;
//                }
//            }
//        }
//        newVar = 4*newVar;
//        Eigen::JacobiSVD<walkers::Mat33> svdVar(var, Eigen::ComputeFullU);
//        Eigen::JacobiSVD<walkers::Mat33> svdNewVar(newVar, Eigen::ComputeFullU);
//        walkers::Mat33 U = svdVar.matrixU();
//        walkers::Mat33 Un = svdNewVar.matrixU();
//        Eigen::Vector3d S = svdVar.singularValues();
//        Eigen::Vector3d Sn = svdNewVar.singularValues();

//        walkers::Mat33 U0 = U.inverse()*U;
//        walkers::Mat33 Un0 = U.inverse()*Un;

//        walkers::Vec3 u = walkers::logmap(U0);
//        Eigen::VectorXd x0(9);
//        x0 << mean(0), mean(1), mean(2), S(0), S(1), S(2), u.x(), u.y(), u.z();

//        walkers::Vec3 un = walkers::logmap(Un0);
//        Eigen::VectorXd x0n(9);
//        x0n << newMean(0), newMean(1), newMean(2), Sn(0), Sn(1), Sn(2), un.x(), un.y(), un.z();

//        Eigen::VectorXd xxx(9);

//        xxx = x0 * sampNumber / (sampNumber + points.size()) + x0n * points.size() / (sampNumber + points.size());

//        /// Krok predykcji
//        Eigen::MatrixXd A(9,9);
//        A = Eigen::MatrixXd::Identity(9,9);
//        Eigen::MatrixXd B(9,9);
//        B = Eigen::MatrixXd::Zero(9,9);
//        Eigen::VectorXd us(9);
//        us << 0,0,0,0,0,0,0,0,0;
//        Eigen::MatrixXd C(9,9);
//        C = Eigen::MatrixXd::Identity(9,9);

//        Eigen::VectorXd xp(9);
//        xp = A*x0 + B*us;
//        Eigen::MatrixXd P(9,9);
//        Eigen::MatrixXd P_pre(9,9);
//        for(int i =0; i<9;i++) {
//            for(int j =0; j<9;j++) {
//                P_pre(i,j) = P_values[9*i+j];
//            }
//        }
//        P = A*P_pre*A.transpose();
//        Eigen::VectorXd y(9);
//        y = x0n;

//        Eigen::VectorXd yPred(9);
//        yPred << 0,0,0,0,0,0,0,0,0;

//        Eigen::MatrixXd R(9,9);
//        R = Eigen::MatrixXd::Identity(9,9);
//        Eigen::VectorXd e(9);
//        e = y - C*xp;

//        Eigen::MatrixXd SS(9,9);
//        SS = C*P*C.transpose() + R;
//        Eigen::MatrixXd K(9,9);
//        K = P*C.transpose()*SS.inverse();

//        Eigen::VectorXd postX(9);
//        postX = xp + K*e;
//        P_pre = P - K*SS*K.transpose();
//        for(int i =0; i<9;i++) {
//            for(int j =0; j<9;j++) {
//                P_values[9*i+j] = P_pre(i,j);
//            }
//        }
//        xp = xxx;

//        sampNumber += points.size();

//        mean << xp(0), xp(1), xp(2);
//        Eigen::Vector3d post_s;
//        post_s << xp(3),xp(4),xp(5);
//        Eigen::Vector3d post_r;
//        post_r << xp(6),xp(7),xp(8);

//        walkers::Mat33 postS;
//        postS << post_s(0), 0, 0, 0,  post_s(1), 0, 0, 0, post_s(2);
//        walkers::Mat33 postR;
//        postR = walkers::expmap(walkers::Vec3(post_r(0), post_r(1), post_r(2)));
//        postR = U*postR;
//        var = postR * postS * postR.transpose();
    }
}

void Voxel::updateNDTOM() {
    if (sampNumber == 0) {
        Eigen::Vector3d newMean = Eigen::Vector3d(0, 0, 0);
        for(mapping::Point3D &point : points) {
            Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
            newMean += newPoint;
        }
        meanSum = newMean;
        mean = meanSum / double(points.size());

        walkers::Mat33 newVar;
        newVar << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        if(points.size() > 0) {
            for(mapping::Point3D &point : points) {
                Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
                newVar += (newPoint - mean) * (newPoint - mean).transpose();
            }

        }
        varSum = newVar;
        var =4.0*( varSum / double(points.size() - 1));
        sampNumber = (unsigned int)points.size();
    } else {
        Eigen::Vector3d newMeanSum = Eigen::Vector3d(0, 0, 0);
        for(mapping::Point3D &point : points) {
            Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
            newMeanSum += newPoint;
        }

        Eigen::Vector3d newMean = newMeanSum / double(points.size());

        walkers::Mat33 newVarSum;
        newVarSum << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        if(points.size() > 0) {
            for(mapping::Point3D &point : points) {
                Eigen::Vector3d newPoint = Eigen::Vector3d(point.position.x(), point.position.y(), point.position.z());
                newVarSum += (newPoint - newMean) * (newPoint - newMean).transpose();
            }
        }

        varSum = varSum + newVarSum + (double(sampNumber)/(double(points.size())*(double(points.size()) + double(sampNumber))))*((double(points.size())/double(sampNumber))*meanSum - newMeanSum)*((double(points.size())/double(sampNumber))*meanSum - newMeanSum).transpose();
        meanSum += newMeanSum;
        mean = meanSum / double(sampNumber + points.size());
        var = 4*(varSum / double(sampNumber + points.size() - 1));
        if (fabs(var.determinant())>0.1){
            std::cout << var << "\n";
//            getchar();
        }
        sampNumber += points.size();
        if (mean.x()<0.1 && mean.y()<0.1 && mean.z()<0.1){
            probability = 0.0;
        }
    }
}

void Voxel::updateColor(RGBA _color) {
    if(sampNumber == 1) {
        this->color = _color;
    } else {
        this->color.r = (uint8_t)(((this->color.r*(sampNumber-1)) + _color.r)/sampNumber);
        this->color.g = (uint8_t)(((this->color.g*(sampNumber-1)) + _color.g)/sampNumber);
        this->color.b = (uint8_t)(((this->color.b*(sampNumber-1)) + _color.b)/sampNumber);
        this->color.a = (uint8_t)(((this->color.a*(sampNumber-1)) + _color.a)/sampNumber);
    }
}

void Voxel::updateOccupancy() {
    if(probability<1.0)
        probability+=0.005;
}

void Voxel::updateNullOccupancy() {
    if(probability>0){
//        std::cout << "probability: " << probability << "\n";
        probability-=0.005;
    }
}
