/** @file mapping_defs.h
*
* Mapping definitions
*
*/

#ifndef MAPPING_DEFS_H_INCLUDED
#define MAPPING_DEFS_H_INCLUDED

#include "defs.h"
#include "Defs/grabber_defs.h"

/// planner name space
namespace mapping {

    /// RGBA color space
    class RGBA {
    public:
        /// Color representation
        union {
            struct {
                std::uint8_t r, g, b, a;
            };
            std::uint8_t rgba_color[4];
        };
        /// Default constructor
        inline RGBA() {
            r = 255; g = 255; b = 255; a = 255;
        }
        /// Default constructor
        inline RGBA(int _r, int _g, int _b, int _a = 255) {
            r = (std::uint8_t)_r;
            g = (std::uint8_t)_g;
            b = (std::uint8_t)_b;
            a = (std::uint8_t)_a;
        }
    };

    /// 3D point representation
    class Point3D {
    public:
        /// Position
        walkers::Vec3 position;
        /// Colour
        RGBA color;

        /// Default constructor
        inline Point3D() {
        }


        inline Point3D(double x, double y, double z) {
            position = walkers::Vec3 (x, y, z);
        }

        inline Point3D(double x, double y, double z, int _r, int _g, int _b, int _a = 255) {
            position = walkers::Vec3 (x, y, z);
            color = RGBA(_r, _g, _b, _a);
        }
    };

    /// 3D point cloud representation
    typedef std::vector<Point3D> PointCloud;

    void savePointCloud(const PointCloud& cloud, const std::string& filename);

    PointCloud loadPointCloud(const std::string& filename);

    class GrabbedImage {
    public:
        PointCloud pointCloud;
        walkers::Vec3 translation;
        walkers::Quaternion orientation;
        std::vector<walkers::Mat33> uncertinatyErrors;
        Eigen::Vector3d cameraPos;


        GrabbedImage();
        GrabbedImage(PointCloud _pc, walkers::Vec3 _translation, walkers::Quaternion _orientation, std::vector<walkers::Mat33> _uncertinatyErrors, Eigen::Vector3d _cameraPos);
        PointCloud transformedPointCloud();
    };

    enum class updateMethodType {
        /// For simple method
        TYPE_SIMPLE,
        /// For bayesian update
        TYPE_BAYES,
        /// For kalman filter implementation
        TYPE_KALMAN,
        /// NDT-OM implementation,
        TYPE_NDTOM,
    };

    /// Ellipsoid representation
    class Ellipsoid {
        public:
            /// sequence of ellipsoids
            typedef std::vector<Ellipsoid> Seq;

            /// position
            walkers::Vec3 position;

            /// covariance matrix
            walkers::Mat33 cov;

            /// color RGBA
            RGBA color;

            /// Default constructor
            inline Ellipsoid(){
            }
    };

    /// Voxel representation
    class VoxelVisu {
        public:
            /// sequence of voxels
            typedef std::vector<VoxelVisu> Seq;

            /// position
            walkers::Mat34 pose;

            /// width
            double width;

            /// color RGBA
            RGBA color;

            /// Default constructor
            inline VoxelVisu(){
            }
    };

    /// Compute eigenmatrix and eigen values using PCA
    void computePCA(const PointCloud& points, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues);

    /// Compute eigenmatrix and eigen values using PCA
    void computePCA(const PointCloud& points, const walkers::Mat33& prevRot, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues);

    /// Compute eigenmatrix and eigen values using PCA
    void computePCA(const PointCloud& points, walkers::Vec3& mean, walkers::Mat33& cov, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues);

    ///compute mean and covariance
    void computeMeanAndVariance(const PointCloud& points, walkers::Vec3& meanPos, walkers::Mat33& cov);

    /// eigenmatrix and eigenvalues from covariance matrix
    void computeEigen(const walkers::Mat33& cov, const walkers::Mat33& prevRot, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues);

    /// eigenmatrix and eigenvalues from covariance matrix
    void computeEigen(const walkers::Mat33& cov, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues);

    /// find minimal rotation
    void findMinimalRotation(walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues);

    /// find minimal rotation
    void findMinimalRotation(const walkers::Mat33& prevRot, walkers::Mat33& eigenvectorsMat, walkers::Mat33& eigenValues);

    void prepareHeightfieldColors(double (&color)[3], int colorScheme, double value, double min, double max);

    void hotColorMap(double(&rgb)[3],double value,double min,double max);

    void jetColorMap(double(&rgb)[3],double value,double min,double max);

    void coldColorMap(double(&rgb)[3],double value,double min,double max);

    void blueColorMap(double(&rgb)[3],double value,double min,double max);

    void positiveColorMap(double(&rgb)[3],double value,double min,double max);

    void negativeColorMap(double(&rgb)[3],double value,double min,double max);

    void colorMap(double(&rgb)[3],double value,double min,double max);

    void cyclicColorMap(double(&rgb)[3],double value,double min,double max);

    void randColorMap(double(&rgb)[3]);

    void grayColorMap(double(&rgb)[3],double value,double min,double max);

    grabber::PointCloud mappingCloud2grabberCloud(const mapping::PointCloud& cloud);

    mapping::PointCloud grabberCloud2mappingCloud(const grabber::PointCloud& cloud);

    /// Calculate normal vector (input: vertices of the triangle, output: normal vector)
    walkers::Vec3 calculateNormal(const std::vector<walkers::Vec3>& vertices);
}

#endif // MAPPING_DEFS_H_INCLUDED
