/** @file procRGBD.cpp
 * @author Dominik Belter
 * image processing
 *
 */

#include "ImageProcessing/procRGBD.h"
#include "opencv2/calib3d/calib3d_c.h"
#include <chrono>
#include <random>

/// detect features and compute descriptors
void KeyFrame::detectorDescriptor(){
    //-- Step 1: Detect the keypoints using SURF Detector
#ifdef CV_BUILD_NONFREE
    int minHessian = 400;
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SURF::create(minHessian);
#endif
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    detector->detect(rgbImage, keypoints);

    //-- Step 2: Calculate descriptors (feature vectors)
#ifdef CV_BUILD_NONFREE
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SURF::create();
#endif
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::AKAZE::create();
    extractor->compute(rgbImage, keypoints, descriptors);
}

/// constructor
KeyFrame::KeyFrame(cv::Mat& _rgbImage, int _frameNo){
    rgbImage = _rgbImage;
    frameNo = _frameNo;
}

///match two frames
void KeyFrame::match(const KeyFrame& frame, double& inliersRatio, double& detHomography){
    //-- Step 3: Matching descriptor vectors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector< cv::DMatch > matches;
    matcher.match(descriptors, frame.descriptors, matches );
    double max_dist = 0; double min_dist = 100;

//    std::cout << "matches size " << matches.size() << "\n";
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors.rows; i++ ) {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< cv::DMatch > good_matches;

    for( int i = 0; i < descriptors.rows; i++ ){
        if( matches[i].distance < 3*min_dist ) {
            good_matches.push_back( matches[i]);
        }
    }
//    std::cout << "good_matches size " << good_matches.size() << "\n";

    cv::Mat imgMatches;
    cv::drawMatches(rgbImage, keypoints, frame.rgbImage, frame.keypoints,
                 good_matches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
//    cv::imshow( "Good Matches & Object detection", imgMatches );

    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for(size_t i = 0; i < good_matches.size(); i++ ){
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints[ good_matches[i].queryIdx ].pt );
      scene.push_back( frame.keypoints[ good_matches[i].trainIdx ].pt );
    }
    cv::Mat mask;
    double ransacReprojThreshold=3;
    cv::Mat H = cv::findHomography( obj, scene, CV_RANSAC, ransacReprojThreshold, mask);
    size_t inliersNo(0); size_t outliersNo(0);
    for (int maskRow = 0; maskRow<mask.rows;maskRow++){
//                    std::cout << "POINTS: object(" << obj.at(maskRow).x << "," << obj.at(maskRow).y << ") - scene(" << scene.at(maskRow).x << "," << scene.at(maskRow).y << ")" << std::endl;
//                    std::cout << "POINTS (via match-set): object(" << keyframes[currentKeyframe].keypoints.at(good_matches.at(maskRow).queryIdx).pt.x << "," << keyframes[currentKeyframe].keypoints.at(good_matches.at(maskRow).queryIdx).pt.y << ") - scene(" << frame.keypoints.at(good_matches.at(maskRow).trainIdx).pt.x << "," << frame.keypoints.at(good_matches.at(maskRow).trainIdx).pt.y << ")" << std::endl;
        if((unsigned int)mask.at<uchar>(maskRow)){
            inliersNo++;
        }
        else{
            outliersNo++;
        }
    }
//    std::cout << "inliersNo " << inliersNo << "\n";
//    std::cout << "outliersNo " << outliersNo << "\n";
    inliersRatio = double(inliersNo)/double(good_matches.size());
    if (H.rows>0&&H.cols>0)
        detHomography = H.at<double>(0, 0) * H.at<double>(1, 1) - H.at<double>(1, 0) * H.at<double>(0, 1);
    else
        detHomography = -1;
    if (inliersRatio>=0.3&&detHomography>0.1){
        cv::imwrite( "image"+std::to_string(frame.frameNo)+".jpg", imgMatches );
    }
}

/// Construction
ImageProcessing::ImageProcessing(void){

}

/// Destructor
ImageProcessing::~ImageProcessing(void){

}

/// compute angle between ellipsoid (major axis) and camera view
double ImageProcessing::angleBetweenEllipsoidAndCamera(const walkers::Mat34& camPose, const walkers::Vec3& meanPos, const walkers::Mat33& eigenvectorsMat, const walkers::Mat33& eigenValues){
    int majorAxisNo = 0;
    if (eigenValues(0,0)>eigenValues(1,1)){
        majorAxisNo = 0;
        if (eigenValues(0,0)>eigenValues(2,2))
            majorAxisNo = 0;
        else
            majorAxisNo = 2;
    }
    else{
        majorAxisNo = 1;
        if (eigenValues(1,1)>eigenValues(2,2))
            majorAxisNo = 1;
        else
            majorAxisNo = 2;
    }
    walkers::Vec3 majorAxis(eigenvectorsMat(0,majorAxisNo), eigenvectorsMat(1,majorAxisNo), eigenvectorsMat(2,majorAxisNo));
    walkers::Vec3 camAxis(meanPos.x()-camPose(0,3), meanPos.y()-camPose(1,3), meanPos.z()-camPose(2,3));
    majorAxis.vector() = majorAxis.vector().normalized();
    camAxis.vector() = camAxis.vector().normalized();
    double dotProduct1 = majorAxis.vector().dot(camAxis.vector());
    double angle1 = std::acos(dotProduct1);
    double dotProduct2 = majorAxis.vector().dot(-camAxis.vector());
    double angle2 = std::acos(dotProduct2);
    if (angle1<angle2) return angle1;
    else return angle2;
}

void ImageProcessing::transformEllipsoids(mapping::Ellipsoid::Seq& ellipsoids, const walkers::Mat34& camPose){
    for (auto& ellipsoid : ellipsoids){
        walkers::Mat34 p(walkers::Mat34::Identity());
        p.matrix().block<3,1>(0,3) = ellipsoid.position.vector();
        walkers::Mat34 p1 = camPose*p;
        ellipsoid.position.vector() = p1.matrix().block<3,1>(0,3);
        ellipsoid.cov = camPose.matrix().block<3,3>(0,0)*ellipsoid.cov*camPose.matrix().block<3,3>(0,0).transpose();
    }
}

mapping::Ellipsoid::Seq ImageProcessing::computeEllipsoids(const grabber::PointCloudOrg& cloud, const walkers::Mat34& camPose, int windowSize){
    mapping::Ellipsoid::Seq ellipsoids;
    for (int col=windowSize/2;col<int(cloud.size()-(windowSize/2));col+=windowSize/2){
        for (int row=windowSize/2;row<int(cloud[col].size()-(windowSize/2));row+=windowSize/2){
            mapping::PointCloud points;
            walkers::Vec3 meanColor(0,0,0);
            double minx=std::numeric_limits<double>::max();
            double maxx=std::numeric_limits<double>::min();
            double miny=std::numeric_limits<double>::max();
            double maxy=std::numeric_limits<double>::min();
            double minz=std::numeric_limits<double>::max();
            double maxz=std::numeric_limits<double>::min();
            for (int i=-windowSize/2;i<1+windowSize/2;i++){
                for (int j=-windowSize/2;j<1+windowSize/2;j++){
                    if (!std::isnan(double(cloud[col+i][row+j].z))){
                        if (cloud[col+i][row+j].z>0){
                            mapping::Point3D point(cloud[col+i][row+j].x, cloud[col+i][row+j].y, cloud[col+i][row+j].z, cloud[col+i][row+j].r, cloud[col+i][row+j].g, cloud[col+i][row+j].b);
                            points.push_back(point);
                            meanColor.x()+=cloud[col+i][row+j].r;
                            meanColor.y()+=cloud[col+i][row+j].g;
                            meanColor.z()+=cloud[col+i][row+j].b;
                            if (cloud[col+i][row+j].x>maxx)
                                maxx = cloud[col+i][row+j].x;
                            if (cloud[col+i][row+j].x<minx)
                                minx = cloud[col+i][row+j].x;
                            if (cloud[col+i][row+j].y>maxy)
                                maxy = cloud[col+i][row+j].y;
                            if (cloud[col+i][row+j].y<miny)
                                miny = cloud[col+i][row+j].y;
                            if (cloud[col+i][row+j].z>maxz)
                                maxz = cloud[col+i][row+j].z;
                            if (cloud[col+i][row+j].z<minz)
                                minz = cloud[col+i][row+j].z;
                        }
                    }
                }
            }
            if (points.size()>70){
                meanColor.vector() = meanColor.vector()/double(points.size());
                walkers::Vec3 meanPos;
                walkers::Mat33 cov;
                    mapping::computeMeanAndVariance(points, meanPos, cov);
                /// Compute eigenmatrix and eigen values using PCA
                walkers::Mat33 eigenvectorsMat; walkers::Mat33 eigenValues;
                mapping::computePCA(points, meanPos, cov, eigenvectorsMat, eigenValues);
                if (angleBetweenEllipsoidAndCamera(camPose, meanPos, eigenvectorsMat, eigenValues)>0.3){
                    if((maxz-minz)<meanPos.z()*0.1&&(maxz-minz)<0.4){
                        std::vector<double> eigVal = {eigenValues(0,0), eigenValues(1,1), eigenValues(2,2)};
                        std::sort(eigVal.begin(), eigVal.end());
    //                std::cout << "eigVal[1] " << eigVal[0] << ", " << eigVal[1] << ", " << eigVal[2] << "\n";
    //                getchar();
//                    if (eigVal[1]/eigVal[2]>0.11){
                        mapping::Ellipsoid ellipsoid;
                        ellipsoid.position = meanPos;
                        ellipsoid.cov = cov;
                        ellipsoid.color.r = (uint8_t)meanColor.x(); ellipsoid.color.g = (uint8_t)meanColor.y(); ellipsoid.color.b = (uint8_t)meanColor.z();
                        ellipsoid.color.a = 255;
                        ellipsoids.push_back(ellipsoid);
                    }
                }
            }
        }
    }
    return ellipsoids;
}

walkers::Vec3 ImageProcessing::sampleFromEllipsoid(const mapping::Ellipsoid& ellipsoid){
    walkers::Vec3 point3D;

    walkers::Mat33 normTransform;
    Eigen::LLT<walkers::Mat33> cholSolver(ellipsoid.cov);

    //auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    static std::mt19937 gen{ std::random_device{}() };
    static std::normal_distribution<> distNorm;

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

void ImageProcessing::filterWithEllipsoids(mapping::PointCloud& pointCloud, const grabber::PointCloudOrg& cloudOrg, const mapping::Ellipsoid::Seq& ellipsoids, const grabber::CameraModel& sensorModel, cv::Mat& colorImage, size_t samplesNo){
    if (ellipsoids.size()==0)
        return;
    pointCloud.clear();
    std::set<std::pair<int,int>> points2remove;
    //initialize rand
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int> distPointNo(0,(int)ellipsoids.size());
    for (size_t pointNo=0; pointNo<samplesNo;pointNo++){
        size_t sample = distPointNo(mt);
        walkers::Vec3 point3D = sampleFromEllipsoid(ellipsoids[sample]);
        walkers::Vec3 imgCoord = sensorModel.inverseModel(point3D.x(), point3D.y(),point3D.z());
        if ((imgCoord.x()>=0)&&(imgCoord.y()>0)){
            const cv::Point3_ <uchar>* p = colorImage.ptr<cv::Point3_<uchar> >(int(imgCoord.y()),int(imgCoord.x()));
            mapping::Point3D newPoint(point3D.x(), point3D.y(), point3D.z(), p->z, p->y, p->x);
            points2remove.insert(std::make_pair<int,int>(int(imgCoord.y()),int(imgCoord.x())));
            pointCloud.push_back(newPoint);
        }
    }
    for (size_t colNo=0;colNo<cloudOrg.size();colNo++){
        for (size_t rowNo=0;rowNo<cloudOrg[colNo].size();rowNo++){
            if (points2remove.find(std::make_pair(colNo,rowNo)) == points2remove.end()){
                mapping::Point3D point;
                point.position = walkers::Vec3(cloudOrg[colNo][rowNo].x, cloudOrg[colNo][rowNo].y, cloudOrg[colNo][rowNo].z);
                point.color.r = cloudOrg[colNo][rowNo].r; point.color.g = cloudOrg[colNo][rowNo].g; point.color.b = cloudOrg[colNo][rowNo].b; point.color.a = cloudOrg[colNo][rowNo].a;
                pointCloud.push_back(point);
            }
        }
    }
}
