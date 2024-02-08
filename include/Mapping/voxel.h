#ifndef VOXEL_H
#define VOXEL_H

#include "Defs/defs.h"
#include "Defs/eigen3.h"
#include "Defs/mapping_defs.h"
#include <tuple>

namespace mapping {

    class Voxel {
    public:
        // [ mean_x mean_y mean_z]
        Eigen::Vector3d mean;
        /*
         * --                    --
         * | var_xx var_xy var_xz |
         * | var_yx var_yy var_yz |
         * | var_zx var_yz var_zz |
         * --                    --
         */
        walkers::Mat33 var;
        walkers::Vec3 estRot;
        walkers::Vec3 estCov;
        double probability;
        size_t sampNumber;
        Eigen::Vector3d sampMean;
        RGBA color;
        int numPoints;
        /// Method type

        //For simple method
        mapping::PointCloud points;
        //std::vector<Mat33> uncertaintyErrors;

        double P_values [81] = { };

        Eigen::Vector3d meanSum;
        walkers::Mat33 varSum;

        ///default constructor
        Voxel();
        /// constructor
        Voxel(int res);

        void preinitParameters(double res, Eigen::Vector3d center);

        void insertPoint(Point3D point);

        void updateWithSimpleMethod(mapping::updateMethodType updateType, int pointThreshold);
        void updateSimpleDistribution(Eigen::Vector3d& meanPos, walkers::Mat33& variance, Eigen::Vector3d& meanColor);
        void updateSimpleColor();
        void getVoxelState(Eigen::Vector3d& _mean, walkers::Mat33& _var, double& _probability, size_t& _sampNumber, Eigen::Vector3d& _sampMean, RGBA& _color, Eigen::Vector3d& _meanSum, walkers::Mat33& _varSum) const;
        void setVoxelState(const Eigen::Vector3d& _mean, const walkers::Mat33& _var, const double& _probability, const unsigned int& _sampNumber, const Eigen::Vector3d& _sampMean, const RGBA& _color, const Eigen::Vector3d& _meanSum, const walkers::Mat33& _varSum);

        walkers::Mat33 prostuj(walkers::Mat33 R);
        std::tuple<walkers::Mat33, Eigen::Vector3d> changeOrder(walkers::Mat33 Rot, Eigen::Vector3d S);
        Eigen::Vector3d castVector(walkers::Mat33 Rot, Eigen::Vector3d S);

        void updateNaiveColor();

        void updateBayesDistribution();

        void updateKalmanDistribution();

        void updateNDTOM();
        void CUDA_UpdateNDTOM();

        void updateColor(RGBA color);

        void updateOccupancy();
        void updateNullOccupancy();

    private:
//        walkers::KalmanFilter kalmanFilter;
        /// prev rotation matrix
        walkers::Mat33 prevRot;
    };




}

#endif // VOXEL_H
