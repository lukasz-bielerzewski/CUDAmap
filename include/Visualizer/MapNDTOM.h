/** @file MapNDTOM.h
 *
 * 3D MapNDTOM
 *
 */

#ifndef MAPNDTOM_H_INCLUDED
#define MAPNDTOM_H_INCLUDED

#include "Defs/defs.h"
#include "Defs/mapping_defs.h"
#include "Mapping/gaussmap.h"
#include "Visualizer/DisplayObject.h"


class DisplayMapNDTOM : public DisplayObject{
public:
    /// MapNDTOM container
    Octree<mapping::Voxel> element;

    /// MapNDTOM color
    mapping::RGBA MapNDTOMColor;
    /// updated voxels
    std::unordered_map<std::string, Eigen::Vector3i> updatedVoxels;

    /// construction
    DisplayMapNDTOM() : element(8){
        type = DisplayObjectType::TYPE_MAPNDTOM;
    }

    /// create display list
    void createDisplayList(const Octree<mapping::Voxel>& mapNDTOM,
                           std::unordered_map<std::string, Eigen::Vector3i> updatedVoxels);

    /// update display list
    void updateDisplayList();

private:
    /// draw ellipsoid
    void drawEllipsoid(const walkers::Vec3& pos, const walkers::Mat33& covariance, mapping::RGBA color) const;
};

#endif // MAPNDTOM_H_INCLUDED
