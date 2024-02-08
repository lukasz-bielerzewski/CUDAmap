/** @file grabber_defs.cpp
*
* Grabber definitions
*
*/

#include "Defs/grabber_defs.h"
#include <tinyxml2.h>

grabber::CameraModel::CameraModel(std::string configFilename){
    setlocale(LC_NUMERIC,"C");
    tinyxml2::XMLDocument config;
    std::string filename = "../../resources/" + configFilename;
    config.LoadFile(filename.c_str());
    if (config.ErrorID()){
        std::cout << "Unable to load Kinect config file.\n";
        std::cout << "Filename " << filename << "\n";
        std::cout << "ErrorID: " << config.ErrorID() << "\n";
    }
    tinyxml2::XMLElement * model = config.FirstChildElement( "Model" );
    model->FirstChildElement( "focalLength" )->QueryDoubleAttribute("fx", &focalLength[0]);
    model->FirstChildElement( "focalLength" )->QueryDoubleAttribute("fy", &focalLength[1]);
    model->FirstChildElement( "focalAxis" )->QueryDoubleAttribute("Cx", &focalAxis[0]);
    model->FirstChildElement( "focalAxis" )->QueryDoubleAttribute("Cy", &focalAxis[1]);
    model->FirstChildElement( "resolution" )->QueryIntAttribute("rx", &resolution[0]);
    model->FirstChildElement( "resolution" )->QueryIntAttribute("ry", &resolution[1]);
    model->FirstChildElement( "depthScale" )->QueryDoubleAttribute("scale", &depthScale);
    model->FirstChildElement( "variance" )->QueryDoubleAttribute("sigmaU", &varU);
    model->FirstChildElement( "variance" )->QueryDoubleAttribute("sigmaV", &varV);
    model->FirstChildElement( "varianceDepth" )->QueryDoubleAttribute("c3", &distVarCoefs[0]);
    model->FirstChildElement( "varianceDepth" )->QueryDoubleAttribute("c2", &distVarCoefs[1]);
    model->FirstChildElement( "varianceDepth" )->QueryDoubleAttribute("c1", &distVarCoefs[2]);
    model->FirstChildElement( "varianceDepth" )->QueryDoubleAttribute("c0", &distVarCoefs[3]);
    tinyxml2::XMLElement * posXML = config.FirstChildElement( "pose" );
    double query[4];
    posXML->QueryDoubleAttribute("qw", &query[0]); posXML->QueryDoubleAttribute("qx", &query[1]); posXML->QueryDoubleAttribute("qy", &query[2]); posXML->QueryDoubleAttribute("qz", &query[3]);
    walkers::Quaternion q(query[0], query[1], query[2], query[3]);
    posXML->QueryDoubleAttribute("x", &query[0]); posXML->QueryDoubleAttribute("y", &query[1]); posXML->QueryDoubleAttribute("z", &query[2]);
    walkers::Vec3 pos(query[0], query[1], query[2]);
    pose = q*pos;

    initPHCPmodel();

    Ruvd << varU, 0, 0,
            0, varV, 0,
            0, 0, 0;
}

///init PHCPmodel
void grabber::CameraModel::initPHCPmodel(void){
    PHCPModel << 1/focalLength[0],0,-focalAxis[0]/focalLength[0],
                  0,1/focalLength[1], -focalAxis[1]/focalLength[1],
                  0,0,1;
}

void grabber::CameraModel::getPoint(unsigned int u, unsigned int v, double depth, Eigen::Vector3d& point3D){
    Eigen::Vector3d point(u, v, 1);
    point3D = depth*PHCPModel*point;
}

walkers::Vec3 grabber::CameraModel::inverseModel(double x, double y, double z) const {
//    walkers::Vec3 point(((focalLength[0]*x)/z)+focalAxis[0]+1e-3, ((focalLength[1]*y)/z)+focalAxis[1]+1e-3, z);
//    if (point.x()<0||point.x()>resolution[0]||point.y()<0||point.y()>resolution[1]||z<0.3||z>6.0){
//        point.x() = -1; point.y() = -1; point.z() = -1;
//    }
//    return point;
    walkers::Vec3 point;
    //point3D = depth*PHCPModel*point;
    walkers::Vec3 point2(x/z, y/z, 1.0);
    point.vector() = PHCPModel.inverse()*point2.vector();
    point.z() = z;
    if (point.x()<0||point.x()>resolution[0]||point.y()<0||point.y()>resolution[1]||z<0.3||z>6.0){
        point.x() = -1; point.y() = -1; point.z() = -1;
    }
    return point;
}

/// pixel from 3D point
walkers::Vec3 grabber::CameraModel::inverseModel(double x, double y, double z, size_t maxCols, size_t maxRows) const {
//    walkers::Vec3 point(((focalLength[0]*x)/z)+(double(maxCols)/2.0)+1e-3, ((focalLength[1]*y)/z)+(double(maxRows)/2.0)+1e-3, z);
//    if (point.x()<0||(size_t)point.x()>=maxRows||point.y()<0||(size_t)point.y()>=maxCols){
//        point.x() = -1; point.y() = -1; point.z() = -1;
//    }
//    return point;
    walkers::Vec3 point;
    //point3D = depth*PHCPModel*point;
    walkers::Vec3 point2(x/z, y/z, 1.0);
    point.vector() = PHCPModel.inverse()*point2.vector();
    point.z() = z;
    if (point.x()<0||point.x()>(double)maxRows||point.y()<0||point.y()>(double)maxCols||z<0.3||z>6.0){
        point.x() = -1; point.y() = -1; point.z() = -1;
    }
    return point;
}

/// Convert disparity image to point cloud
grabber::PointCloud grabber::CameraModel::depth2cloud(const cv::Mat& depthImage, const cv::Mat& colorImage){
    grabber::PointCloud cloud;
    cloud.clear();
    for (int i=0;i<depthImage.rows;i++){
        for (int j=0;j<colorImage.cols;j++){
            const cv::Point3_ <uchar>* p = colorImage.ptr<cv::Point3_<uchar> >(i,j);
            grabber::Point3D point;
            Eigen::Vector3d pointxyz;
            getPoint(j,i, (double)(depthImage.at<uint16_t>(i, j))/depthScale, pointxyz);//
                point.x = (float)pointxyz(0); point.y = (float)pointxyz(1); point.z = (float)pointxyz(2);
                point.r = p->z; point.g = p->y; point.b = p->x;
                cloud.push_back(point);
//                    }
            //getchar();
        }
    }
    return cloud;
}

/// Convert disparity image to point cloud
grabber::PointCloud grabber::CameraModel::depth2cloud(const cv::Mat& depthImage, const cv::Mat& colorImage, double depthFactor, double minDepth, double maxDepth){
    return depth2cloud(depthImage, colorImage, depthFactor, minDepth, maxDepth, std::make_pair(0,depthImage.cols), std::make_pair(0,depthImage.rows));
}

grabber::PointCloud grabber::CameraModel::depth2cloud(const cv::Mat& depthImage, const cv::Mat& colorImage, double depthFactor, double minDepth, double maxDepth, const std::pair<size_t, size_t>& boundingBoxX, const std::pair<size_t, size_t>& boundingBoxY) {
    int offsetX = int(depthImage.cols - 2 * focalAxis[0]);
    int offsetY = int(depthImage.rows - 2 * focalAxis[1]);

    //int numThreads = std::thread::hardware_concurrency();
    int numThreads = 8;

    std::vector<std::thread> threads;
    std::vector<grabber::PointCloud> threadClouds(static_cast<size_t>(numThreads));

    auto processPixels = [&](int startRow, int endRow, grabber::PointCloud& localCloud) {
        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < colorImage.cols; ++j) {
                if (i > int(boundingBoxY.first) && i < int(boundingBoxY.second) && j > int(boundingBoxX.first) && j < int(boundingBoxX.second)) {
                    int coordU = i + offsetY;
                    int coordV = j + offsetX;

                    if (coordU < 0) { coordU = 0; }
                    if (coordU >= depthImage.rows) { coordU = depthImage.rows - 1; }
                    if (coordV < 0) { coordV = 0; }
                    if (coordV >= depthImage.cols) { coordV = depthImage.cols - 1; }

                    const cv::Point3_<uchar>* p = colorImage.ptr<cv::Point3_<uchar>>(coordU, coordV);

                    grabber::Point3D point;
                    Eigen::Vector3d pointxyz;

                    getPoint(j, i, (double)(depthImage.at<uint16_t>(i, j)) * depthFactor, pointxyz);

                    if (pointxyz(2) > minDepth && pointxyz(2) < maxDepth) {
                        point.x = (float)pointxyz(0);
                        point.y = (float)pointxyz(1);
                        point.z = (float)pointxyz(2);
                        point.r = p->z;
                        point.g = p->y;
                        point.b = p->x;

                        if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
                            std::cout << "nan\n";
                            getchar();
                        }

                        localCloud.push_back(point);
                    }
                }
            }
        }
    };

    for (int t = 0; t < numThreads; ++t) {
        int startRow = (t == 0) ? 0 : t * (depthImage.rows / numThreads) + 1;
        int endRow = (t + 1) * (depthImage.rows / numThreads);

        threads.emplace_back(processPixels, startRow, endRow, std::ref(threadClouds[t]));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Concatenate clouds from all threads
    grabber::PointCloud cloud;
    for (const auto& localCloud : threadClouds) {
        cloud.insert(cloud.end(), localCloud.begin(), localCloud.end());
    }

    return cloud;
}


/// Convert disparity image to point cloud
grabber::PointCloud grabber::CameraModel::depth2cloud(const cv::Mat& depthImage){
    Eigen::Vector3d point;
    grabber::PointCloud cloud;
    cloud.clear();
    for (int i=0;i<depthImage.rows;i++){
        for (int j=0;j<depthImage.cols;j++){
            if (depthImage.at<uint16_t>(i,j)>800&&depthImage.at<uint16_t>(i,j)<8500){
                double depthM = double(depthImage.at<uint16_t>(i,j))*(1/depthScale);
                getPoint(j,i,depthM,point);
                grabber::Point3D pointPCL;
                pointPCL.x = (float)point(0); pointPCL.y = (float)point(1); pointPCL.z = (float)point(2);
                pointPCL.r = 255; pointPCL.g = 255; pointPCL.b = 0; pointPCL.a = 0;
                cloud.push_back(pointPCL);
            }
        }
    }
    return cloud;
}

/// Convert disparity image to point cloud
grabber::PointCloudOrg grabber::CameraModel::depth2cloudOrg(const cv::Mat& depthImage, const cv::Mat& colorImage){
    grabber::PointCloudOrg cloud;
    cloud.clear();
    cloud.resize(depthImage.rows);
    for (int i=0;i<depthImage.rows;i++){
        cloud[i].resize(depthImage.cols);
        for (int j=0;j<colorImage.cols;j++){
            const cv::Point3_ <uchar>* p = colorImage.ptr<cv::Point3_<uchar> >(i,j);
            grabber::Point3D point;
            Eigen::Vector3d pointxyz;
            getPoint(j,i, (double)depthImage.at<uint16_t>(i, j)/depthScale, pointxyz);//
            if (pointxyz(2)>0.5&&pointxyz(2)<6.5){
                point.x = (float)pointxyz(0); point.y = (float)pointxyz(1); point.z = (float)pointxyz(2);
                point.r = p->z; point.g = p->y; point.b = p->x;
                cloud[i][j] = point;
            }
        }
    }
    return cloud;
}

/// compute covariance
void grabber::CameraModel::computeCov(int u, int v, double depth, walkers::Mat33& cov){
    //double dispDer = config.k3 * 1/(config.k2*pow(cos((disparity/config.k2) + config.k1),2.0));
    walkers::Mat33 J;
    J << 0.0017*depth, 0, (0.0017*double(u)-0.549),
         0, 0.0017*depth, (0.0017*double(v)-0.443),
         0, 0, 1;
    Ruvd(2,2) = (distVarCoefs[0]*pow(depth,3.0) + distVarCoefs[1]*pow(depth,2.0) + distVarCoefs[2]*depth + distVarCoefs[3])/3.0;
    cov=J*Ruvd*J.transpose();
}
