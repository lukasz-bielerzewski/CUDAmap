/** @file mapping_defs.cpp
*
* Mapping definitions
*
*/

#include "Defs/mapping_defs.h"

namespace mapping {

GrabbedImage::GrabbedImage () {

}

GrabbedImage::GrabbedImage(PointCloud _pc, walkers::Vec3 _translation, walkers::Quaternion _orientation, std::vector<walkers::Mat33> _uncertinatyErrors, Eigen::Vector3d _cameraPos) {
    this->pointCloud = _pc;
    this->translation = _translation;
    this->orientation = _orientation;
    this->uncertinatyErrors = _uncertinatyErrors;

    this->cameraPos = _cameraPos;
}

void savePointCloud(const PointCloud& cloud, const std::string& filename){
    std::ofstream file;
    file.open (filename);
    for (const auto& point : cloud)
        file << (int)point.color.r << " " << (int)point.color.g << " " << (int)point.color.b << " " << (int)point.color.a << " " << point.position.x() << " " << point.position.y() << " " << point.position.z() << "\n";
    file.close();
}

PointCloud loadPointCloud(const std::string& filename){
    std::string line;
    std::ifstream myfile(filename);
    PointCloud cloud;
    if (myfile.is_open()){
        while (std::getline(myfile,line)){
            std::istringstream iss(line);
            double x, y, z;
            int r, g, b, a;
            if (!(iss >> r >> g >> b >> a >> x >> y >> z)) {
                break;
            } // error
            else{
                Point3D point(x, y, z,r,g,b,a);
                cloud.push_back(point);
            }
        }
        myfile.close();
    }
    return cloud;
}

PointCloud GrabbedImage::transformedPointCloud() {
    PointCloud newPC;

    Eigen::Matrix4d A;

    A = Eigen::Matrix4d::Identity();
    //        A << 0.241761029007299, -0.156482265388292, 0.957635058606503, 0.078542236056188,
    //                -0.159587283912986, -0.979884124280077, -0.119829052391907, -0.078052590612562,
    //                0.957122512360151, -0.123856382971628, -0.261870373907737, -0.061252007077847,
    //                0, 0, 0, 1;
    newPC.reserve(this->pointCloud.size());
    Eigen::Matrix4d T;
    walkers::Mat33 R = orientation.normalized().toRotationMatrix();
    T << R(0,0), R(0,1), R(0,2), translation.x(),
            R(1,0), R(1,1), R(1,2), translation.y(),
            R(2,0), R(2,1), R(2,2), translation.z(),
            0, 0, 0, 1;
    T = T;
    T = T * A;
    for(mapping::Point3D point : this->pointCloud) {

        Eigen::Matrix4d Tp;
        Tp << 1, 0, 0, point.position.x(),
                0, 1, 0, point.position.y(),
                0, 0, 1, point.position.z(),
                0, 0, 0, 1;
        Eigen::Matrix4d wynik = Eigen::Matrix4d(T * Tp);

        newPC.push_back(mapping::Point3D(
                            wynik(0,3),
                            wynik(1,3),
                            wynik(2,3),
                            point.color.r,
                            point.color.g,
                            point.color.b,
                            point.color.a));

    }
    return newPC;
}

///compute mean and covariance
void computeMeanAndVariance(const PointCloud& points, walkers::Vec3& meanPos, walkers::Mat33& cov){
    //compute mean
    for (const auto& point : points){
        meanPos.vector()+=point.position.vector();
    }
    meanPos.vector()=meanPos.vector()*1.0/(double)points.size();
    // compute covariance
    for (auto & point : points){
        cov+=(point.position.vector()-meanPos.vector())*(point.position.vector()-meanPos.vector()).transpose();
    }
    cov=4.0*cov/double(points.size()-1);
}

/// Compute eigenmatrix and eigen values using PCA
void computePCA(const PointCloud& points, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues){
    //compute mean
    walkers::Vec3 mean(0,0,0);
    walkers::Mat33 cov(walkers::Mat33::Zero());
    computeMeanAndVariance(points, mean, cov);
    computeEigen(cov, eigenvectorsMat, eigenValues);
            //std::cout << "normal " << pointNormal.normal.transpose() << "\n";
    //getchar();
}

/// Compute eigenmatrix and eigen values using PCA
void computePCA(const PointCloud& points, walkers::Vec3& mean, walkers::Mat33& cov, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues){
    //compute mean
    computeMeanAndVariance(points, mean, cov);
    computeEigen(cov, eigenvectorsMat, eigenValues);
            //std::cout << "normal " << pointNormal.normal.transpose() << "\n";
    //getchar();
}

/// Compute eigenmatrix and eigen values using PCA
void computePCA(const PointCloud& points, const walkers::Mat33& prevRot, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues){
    //compute mean
    walkers::Vec3 mean(0,0,0);
    walkers::Mat33 cov(walkers::Mat33::Zero());
    computeMeanAndVariance(points, mean, cov);
    computeEigen(cov, prevRot, eigenvectorsMat, eigenValues);
            //std::cout << "normal " << pointNormal.normal.transpose() << "\n";
    //getchar();
}

/// eigenmatrix and eigenvalues from covariance matrix
void computeEigen(const walkers::Mat33& cov, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues){
    Eigen::EigenSolver<walkers::Mat33> es(cov);
//        std::cout << "cov: \n" << cov << "\n";
//    std::cout << "eigenvalues \n" << es.eigenvalues() << "\n";
//    std::cout << "eigenvector0 " << es.eigenvectors() << "\n";
//    std::cout << std::real(es.eigenvectors()(0,0)) << ", " << std::real(es.eigenvectors()(1,0)) << ", " << std::real(es.eigenvectors()(2,0)) << "\n";

    for (size_t colNo = 0;colNo<3;colNo++){
        for (size_t rowNo = 0;rowNo<3;rowNo++){
            eigenvectorsMat(rowNo,colNo) = std::real(es.eigenvectors()(rowNo,colNo));
        }
    }
    eigenValues = walkers::Mat33::Identity();
    for (size_t dim=0;dim<3;dim++){//some eigenvalues might be small and below 0
        if (std::real(es.eigenvalues()(dim))<0){
            eigenValues(dim,dim) = sqrt(std::real(-es.eigenvalues()(dim)));
            eigenvectorsMat.block<3,1>(0,dim)=-eigenvectorsMat.block<3,1>(0,dim);
        }
        else
            eigenValues(dim,dim) = sqrt(std::real(es.eigenvalues()(dim)));
    }
    if (std::isnan(eigenValues(0,0))||std::isnan(eigenValues(1,1))||std::isnan(eigenValues(2,2))){
        std::cout << "isnan\n" << eigenValues << "\n";
        getchar();
    }

    findMinimalRotation(eigenvectorsMat, eigenValues);
    if (eigenvectorsMat.determinant()<0){
        std::cout << "not right\n";
        getchar();
    }
}

/// eigenmatrix and eigenvalues from covariance matrix
void computeEigen(const walkers::Mat33& cov, const walkers::Mat33& prevRot, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues){
    Eigen::EigenSolver<walkers::Mat33> es(cov);
//        std::cout << "cov: \n" << cov << "\n";
//    std::cout << "eigenvalues \n" << es.eigenvalues() << "\n";
//    std::cout << "eigenvector0 " << es.eigenvectors() << "\n";
//    std::cout << std::real(es.eigenvectors()(0,0)) << ", " << std::real(es.eigenvectors()(1,0)) << ", " << std::real(es.eigenvectors()(2,0)) << "\n";

    for (size_t colNo = 0;colNo<3;colNo++){
        for (size_t rowNo = 0;rowNo<3;rowNo++){
            eigenvectorsMat(rowNo,colNo) = std::real(es.eigenvectors()(rowNo,colNo));
        }
    }
    eigenValues = walkers::Mat33::Identity();
    for (size_t dim=0;dim<3;dim++){//some eigenvalues might be small and below 0
        if (std::real(es.eigenvalues()(dim))<0){
            eigenValues(dim,dim) = sqrt(std::real(-es.eigenvalues()(dim)));
            eigenvectorsMat.block<3,1>(0,dim)=-eigenvectorsMat.block<3,1>(0,dim);
        }
        else
            eigenValues(dim,dim) = sqrt(std::real(es.eigenvalues()(dim)));
    }
    if (std::isnan(eigenValues(0,0))||std::isnan(eigenValues(1,1))||std::isnan(eigenValues(2,2))){
        std::cout << "isnan\n" << eigenValues << "\n";
        getchar();
    }

    findMinimalRotation(prevRot, eigenvectorsMat, eigenValues);
    if (eigenvectorsMat.determinant()<0){
        std::cout << "not right\n";
        std::cout << "prevRot " << prevRot << "\n";
        std::cout << "eigenvectorsMat " << eigenvectorsMat << "\n";
        getchar();
    }
}

/// find minimal rotation
void findMinimalRotation(walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues){
    std::vector<std::pair<int, int>> mapAxis;
    std::vector<double> sign = {1, 1, 1};
    for (int testAxis=0;testAxis<3;testAxis++){
        double minAngle = std::numeric_limits<double>::max();
        int minAxis=0;
        bool reverse = true;
        for (int axisNo=0;axisNo<3;axisNo++){
            bool considerAxis(true);
            for (const auto& is : mapAxis)
                if (axisNo == is.first)
                    considerAxis = false;
            if (considerAxis){
                walkers::Vec3 axisTest(0,0,0);
                axisTest.vector()(testAxis) = 1;
                walkers::Vec3 axis(eigenvectorsMat(0,axisNo), eigenvectorsMat(1,axisNo), eigenvectorsMat(2,axisNo));
                double dotProduct1 = axis.vector().dot(axisTest.vector());
                double angle1 = std::acos(dotProduct1);
                double dotProduct2 = axis.vector().dot(-axisTest.vector());
                double angle2 = std::acos(dotProduct2);
                if (fabs(angle1)<minAngle){
                    minAxis = axisNo;
                    minAngle = std::fabs(angle1);
                    reverse = false;
                }
                if (fabs(angle2)<minAngle){
                    minAxis = axisNo;
                    minAngle = std::fabs(angle2);
                    reverse = true;
                }
            }
        }
        mapAxis.push_back(std::make_pair(minAxis,testAxis));
        if (reverse)
            sign[minAxis] = -1;
    }
    walkers::Mat33 newEigenMatrix;
    walkers::Mat33 newEigenValues(eigenValues);
    for (int axisNo=0;axisNo<3;axisNo++){
        newEigenMatrix.block<3,1>(0,mapAxis[axisNo].first) = sign[mapAxis[axisNo].first]*eigenvectorsMat.block<3,1>(0,mapAxis[axisNo].second);
        newEigenValues(mapAxis[axisNo].first,mapAxis[axisNo].first) = eigenValues(mapAxis[axisNo].second,mapAxis[axisNo].second);
    }
    eigenvectorsMat = newEigenMatrix;
    eigenValues = newEigenValues;
}

/// find minimal rotation
void findMinimalRotation(const walkers::Mat33& prevRot, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues){
    std::vector<std::pair<int, int>> mapAxis;
    std::vector<double> sign = {1, 1, 1};
//    std::cout << "prev rot\n" << prevRot.matrix() << "\n";
//    std::cout << "eigenvectorsMat\n" << eigenvectorsMat << "\n";
    for (int testAxis=0;testAxis<3;testAxis++){
        double minAngle = std::numeric_limits<double>::max();
        int minAxis=0;
        bool reverse = true;
        for (int axisNo=0;axisNo<3;axisNo++){
            bool considerAxis(true);
            for (const auto& is : mapAxis)
                if (axisNo == is.first)
                    considerAxis = false;
            if (considerAxis){
                walkers::Vec3 axisTest(prevRot(0,testAxis),prevRot(1,testAxis),prevRot(2,testAxis));
//                std::cout << "axisTest " << axisTest.vector() << "\n";
                walkers::Vec3 axis(eigenvectorsMat(0,axisNo), eigenvectorsMat(1,axisNo), eigenvectorsMat(2,axisNo));
//                std::cout << "axis " << axis.vector() << "\n";
                double dotProduct1 = axis.vector().dot(axisTest.vector());
                double angle1 = std::acos(dotProduct1);
                double dotProduct2 = axis.vector().dot(-axisTest.vector());
                double angle2 = std::acos(dotProduct2);
//                std::cout << "angle " << angle1 << ", " << angle2 << "\n";
                if (fabs(angle1)<minAngle){
                    minAxis = axisNo;
                    minAngle = std::fabs(angle1);
                    reverse = false;
                }
                if (fabs(angle2)<minAngle){
                    minAxis = axisNo;
                    minAngle = std::fabs(angle2);
                    reverse = true;
                }
//                std::cout << "minAxis " << minAxis << ", " << minAngle << "\n";
            }
        }
        mapAxis.push_back(std::make_pair(minAxis,testAxis));
        if (reverse)
            sign[minAxis] = -1;
    }
    walkers::Mat33 newEigenMatrix;
    walkers::Mat33 newEigenValues(eigenValues);
    for (int axisNo=0;axisNo<3;axisNo++){
        newEigenMatrix.block<3,1>(0,mapAxis[axisNo].first) = sign[mapAxis[axisNo].second]*eigenvectorsMat.block<3,1>(0,mapAxis[axisNo].second);
        newEigenValues(mapAxis[axisNo].first,mapAxis[axisNo].first) = eigenValues(mapAxis[axisNo].second,mapAxis[axisNo].second);
    }
    eigenvectorsMat = newEigenMatrix;
    eigenValues = newEigenValues;
    walkers::Mat33 motion = walkers::Mat33(prevRot.inverse()*eigenvectorsMat);
//        std::cout << "motion \n" << motion << "\n";
    walkers::Vec3 motionParam = walkers::logmap(motion);
    if (motionParam.vector().norm()>1.0){
//        std::cout << "motion param\n" << motionParam.vector() << "\n";
//        std::cout << prevRot << "\n";
//        std::cout << eigenvectorsMat << "\n";
        //correct orientation
        mapAxis.clear();
        for (int axisNo=0;axisNo<3;axisNo++){
            int maxRow=0;
            bool reverse(false);
            double maxVal = std::numeric_limits<double>::min();
            for (int rowNo=0;rowNo<3;rowNo++){
                if (fabs(prevRot(rowNo,axisNo))>maxVal){
                    maxRow = rowNo;
                    maxVal = fabs(prevRot(rowNo,axisNo));
                }
            }
//            std::cout << "maxRov"  << maxRow << "\n";
            double maxAxis2=0;
            maxVal = std::numeric_limits<double>::min();
            for (int colNo=0;colNo<3;colNo++){
                if (fabs(eigenvectorsMat(maxRow,colNo))>maxVal){
                    maxAxis2 = colNo;
                    maxVal = fabs(eigenvectorsMat(maxRow,colNo));
                    if ((prevRot(maxRow,axisNo)>0&&eigenvectorsMat(maxRow,colNo)<0)||(prevRot(maxRow,axisNo)<0&&eigenvectorsMat(maxRow,colNo)>0))
                        reverse = true;
                    else
                        reverse = false;
                }
            }
            mapAxis.push_back(std::make_pair(maxAxis2, axisNo));
            if (reverse)
                sign[axisNo] = -1;
            else
                sign[axisNo] = 1;
//            std::cout << maxAxis2 << " -> " << sign[axisNo]*axisNo << "\n";
        }
        for (int axisNo=0;axisNo<3;axisNo++){
            newEigenMatrix.block<3,1>(0,mapAxis[axisNo].second) = sign[mapAxis[axisNo].first]*eigenvectorsMat.block<3,1>(0,mapAxis[axisNo].first);
            newEigenValues(mapAxis[axisNo].second, mapAxis[axisNo].second) = eigenValues(mapAxis[axisNo].first,mapAxis[axisNo].first);
        }
        eigenvectorsMat = newEigenMatrix;
        eigenValues = newEigenValues;
//        std::cout << "corrected rot\n";
//        std::cout << eigenvectorsMat << "\n";
        motion = prevRot.inverse()*eigenvectorsMat;
        motionParam = walkers::logmap(motion);
//        std::cout << "motion param\n" << motionParam.vector() << "\n";
//        getchar();
    }
//    getchar();
}

void hotColorMap(double (&rgb)[3],double value,double min,double max){
      double max3 = (max-min)/3;
      value -= min;
      if (value == HUGE_VAL)
      {
          rgb[0] = rgb[1] = rgb[2] = 1.0;
      }
      else if (value < 0)
      {
          rgb[0] = rgb[1] = rgb[2] = 0.0;
      }
      else if (value<max3)
      {
          rgb[0] = 1.0*value/max3;
          rgb[1] = 0.0;
          rgb[2] = 0.0;
      }
      else if (value<2*max3)
      {
          rgb[0] = 1.0;
          rgb[1] = 1.0*(value-max3)/max3;
          rgb[2] = 0;
      }
      else if (value<max)
      {
          rgb[0] = 1.0;
          rgb[1] = 1.0;
          rgb[2] = 1.0*(value-2*max3)/max3;
      }
      else
      {
          rgb[0] = rgb[1] = rgb[2] = 1.0;
      }
}

void jetColorMap(double(&rgb)[3],double value,double min,double max){
      double c1 = 0.54;
      double max4 = (max-min)/4;
      value -= min;
      if (value == HUGE_VAL)
      {
          rgb[0] = rgb[1] = rgb[2] = 1.0;
      }
      else if (value < 0)
      {
          rgb[0] = rgb[1] = rgb[2] = 0.0;
      }
      else if (value < max4)
      {
          rgb[0] = 0;
          rgb[1] = 0;
          rgb[2] = c1+((1.0-c1)*value/max4);
      }
      else if (value < 2*max4)
      {
          rgb[0] = 0;
          rgb[1] = 1.0*(value-max4)/max4;
          rgb[2] = 1.0;
      }
      else if (value < 3*max4)
      {
          rgb[0] = 1.0*(value-2*max4)/max4;
          rgb[1] = 1.0;
          rgb[2] = 1.0-rgb[0];
      }
      else if (value < max)
      {
          rgb[0] = 1.0;
          rgb[1] = 1.0-1.0*(value-3*max4)/max4;
          rgb[2] = 0;
      }
      else
      {
          rgb[0] = 1.0;
          rgb[1] = rgb[2] = 0.0;
      }
}

void coldColorMap(double(&rgb)[3],double value,double min,double max){
      double max3 = (max-min)/3;
      value -= min;
      if (value == HUGE_VAL)
      {
          rgb[0] = rgb[1] = rgb[2] = 1.0;
      }
      else if (value < 0)
      {
          rgb[0] = rgb[1] = rgb[2] = 0.0;
      }
      else if (value < max3)
      {
          rgb[0] = 0.0;
          rgb[1] = 0.0;
          rgb[2] = 1.0*value/max3;
      }
      else if (value < 2*max3)
      {
          rgb[0] = 0;
          rgb[1] = 1.0*(value-max3)/max3;
          rgb[2] = 1.0;
      }
      else if (value < max)
      {
          rgb[0] = 1.0*(value-2*max3)/max3;
          rgb[1] = 1.0;
          rgb[2] = 1.0;
      }
      else
      {
          rgb[0] = rgb[1] = rgb[2] = 1.0;
      }
}

void blueColorMap(double(&rgb)[3],double value,double min,double max){
      value -= min;
      if (value == HUGE_VAL)
      {
          rgb[0] = rgb[1] = rgb[2] = 1.0;
      }
      else if (value < 0)
      {
          rgb[0] = rgb[1] = rgb[2] = 0.0;
      }
      else if (value < max)
      {
          rgb[0] = 0;
          rgb[1] = 0;
          rgb[2] = 1.0*value/max;
      }
      else
      {
          rgb[0] = rgb[1] = 0;
          rgb[2] = 1.0;
      }
}

void positiveColorMap(double(&rgb)[3],double value,double min,double max){
      value -= min;
      max -= min;
      value /= max;

      if(value<0)
      {
        rgb[0] = rgb[1] = rgb[2] = 0.0;
        return;
      }
      if(value>1)
      {
        rgb[0] = rgb[1] = rgb[2] = 1.0;
        return;
      }

      rgb[0] = 0.75;
      rgb[1] = 0.0;
      rgb[2] = 0.0;
      rgb[0] += 0.25*value;
      rgb[1] += 1.0 *value;

      if(value > 0.5)
      {
        rgb[2] += 1.0 * 2.0 * (value - 0.5);
      }
}

void negativeColorMap(double(&rgb)[3],double value,double min,double max){
    value -= min;
    max -= min;
    value /= max;

    rgb[0] = 0.0;
    rgb[1] = 0.0;
    rgb[2] = 0.0;
    if(value<0)
    {
        return;
    }
    if(value>1)
    {
        rgb[1] = rgb[2] = 1.0;
        return;
    }

    rgb[1] += 1.0*value;
    if(value > 0.5)
    {
        rgb[2] += 1.0 * 2.0 * (value - 0.5);
    }
}

void colorMap(double(&rgb)[3],double value,double min,double max){
    if(value > 0)
    {
        positiveColorMap(rgb,value,0,max);
    }
    else
    {
        negativeColorMap(rgb,value,min,0);
    }
}

void cyclicColorMap(double(&rgb)[3],double value,double min,double max){
    double max3 = (max-min)/3;
    value -= (max-min)*floor((value-min)/(max-min));
    if (value < max3)
    {
        rgb[0]=1.0-1.0*value/max3;rgb[1]=0.0;rgb[2]=1.0-rgb[0];
    }
    else if (value < 2*max3)
    {
        rgb[0] = 0.0;
        rgb[1] = 1.0*(value-max3)/max3;
        rgb[2] = 1.0-rgb[1];
    }
    else if(value<max)
    {
        rgb[0]=1.0*(value-2*max3)/max3;
        rgb[1]=1.0-rgb[0];
        rgb[2]=0.0;
    }
}

void randColorMap(double(&rgb)[3]){
    srand (static_cast <unsigned> (time(0)));
    rgb[0]=(double)rand()/  RAND_MAX;
    rgb[1]=(double)rand()/  RAND_MAX;
    rgb[2]=(double)rand()/  RAND_MAX;
}

void grayColorMap(double(&rgb)[3],double value,double min,double max){
    max -= min;
    value -= min;
    rgb[0] = rgb[1] = rgb[2] = 1.0*value/max;
}

grabber::PointCloud mappingCloud2grabberCloud(const mapping::PointCloud& cloud){
    grabber::PointCloud cloudOut;
    cloudOut.resize(cloud.size());
    size_t pointNo=0;
    for (auto& point : cloud){
        cloudOut[pointNo].x = float(point.position.x()); cloudOut[pointNo].y = float(point.position.y()); cloudOut[pointNo].z = float(point.position.z());
        cloudOut[pointNo].r = point.color.r; cloudOut[pointNo].g = point.color.g;
        cloudOut[pointNo].b = point.color.b; cloudOut[pointNo].a = point.color.a;
        pointNo++;
    }
    return cloudOut;
}

mapping::PointCloud grabberCloud2mappingCloud(const grabber::PointCloud& cloud){
    mapping::PointCloud cloudOut;
    cloudOut.resize(cloud.size());
    size_t pointNo=0;
    for (auto& point : cloud){
        cloudOut[pointNo].position.x() = point.x; cloudOut[pointNo].position.y() = point.y; cloudOut[pointNo].position.z() = point.z;
        cloudOut[pointNo].color.r = point.r; cloudOut[pointNo].color.g = point.g;
        cloudOut[pointNo].color.b = point.b; cloudOut[pointNo].color.a = point.a;
        pointNo++;
    }
    return cloudOut;
}

/// Calculate normal vector (input: vertices of the triangle, output: normal vector)
walkers::Vec3 calculateNormal(const std::vector<walkers::Vec3>& vertices){
    walkers::Vec3 v1(vertices[0].x() - vertices[1].x(), vertices[0].y() - vertices[1].y(), vertices[0].z() - vertices[1].z());
    walkers::Vec3 v2(vertices[1].x() - vertices[2].x(), vertices[1].y() - vertices[2].y(), vertices[1].z() - vertices[2].z());

    walkers::Vec3 out(v1.y()*v2.z() - v1.z()*v2.y(), v1.z()*v2.x() - v1.x()*v2.z(), v1.x()*v2.y() - v1.y()*v2.x());
    double module = sqrt(pow(out.x(),2.0) + pow(out.y(),2.0) + pow(out.z(),2.0));
    out.x() /= module; out.y() /= module; out.z() /= module;
    return out;
}

void prepareHeightfieldColors(double(&color)[3], int colorScheme, double value, double min, double max)
{
    switch (colorScheme)
    {
        case 0:{
            grayColorMap(color, value, min, max);
            break;
        }
        case 1:{
            hotColorMap(color, value, min, max);
            break;
        }
        case 2:{
            jetColorMap(color, value, min, max);
            break;
        }
        case 3:{
            coldColorMap(color, value, min, max);
            break;
        }
        case 4:{
            blueColorMap(color, value, min, max);
            break;
        }
        case 5:{
            cyclicColorMap(color, value, min, max);
            break;
        }
        case 6:{
            colorMap(color, value, min, max);
            break;
        }
        case 7:{
            randColorMap(color);
            break;
        }
    default:
        break;

    }
}

}

