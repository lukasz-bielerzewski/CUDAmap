/** @file procRGBD.h
 * @author Dominik Belter
 * Image processing
 *
 */

#ifndef _PROCRGBD_H_
#define _PROCRGBD_H_

#include "Defs/defs.h"
#include "Defs/mapping_defs.h"
#include "Defs/grabber_defs.h"

class KeyFrame{
public:
    /// set of Keyframes
    typedef std::vector<KeyFrame> Seq;
    /// keypoints
    std::vector<cv::KeyPoint> keypoints;
    /// descriptors
    cv::Mat descriptors;
    /// descriptors
    cv::Mat rgbImage;
    /// frame No
    int frameNo;
    /// camera Pose
    walkers::Mat34 cameraPose;

    /// constructor
    KeyFrame(cv::Mat& rgbImage, int _frameNo);

    /// detect features and compute descriptors
    void detectorDescriptor();
    ///match two frames
    void match(const KeyFrame& frame, double& inliersRatio, double& detHomography);
};

/// image processing
class ImageProcessing{
    public:
        /// Pointer
        typedef std::unique_ptr<ImageProcessing> Ptr;

        /// Construction
        ImageProcessing(void);

        /// Destructor
        ~ImageProcessing(void);

        static mapping::Ellipsoid::Seq computeEllipsoids(const grabber::PointCloudOrg& cloud, const walkers::Mat34& camPose, int windowSize);

        static void filterWithEllipsoids(mapping::PointCloud& pointCloud, const grabber::PointCloudOrg& cloudOrg, const mapping::Ellipsoid::Seq& ellipsoids, const grabber::CameraModel& sensorModel, cv::Mat& colorImage, size_t samplesNo);

    private:
        /// sample from ellipsoid
        static walkers::Vec3 sampleFromEllipsoid(const mapping::Ellipsoid& ellipsoid);
        /// transform ellipsoid
        void transformEllipsoids(mapping::Ellipsoid::Seq& ellipsoids, const walkers::Mat34& camPose);
        /// compute angle between ellipsoid (major axis) and camera view
        static double angleBetweenEllipsoidAndCamera(const walkers::Mat34& camPose, const walkers::Vec3& meanPos, const walkers::Mat33& eigenvectorsMat, const walkers::Mat33& eigenValues);
};

#endif // _PROCRGBD_H_
