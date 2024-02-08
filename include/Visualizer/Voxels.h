/** @file Voxels.h
 *
 * voxels
 *
 */

#ifndef VOXELS_H_INCLUDED
#define VOXELS_H_INCLUDED

#include "Defs/defs.h"
#include "Visualizer/DisplayObject.h"
#include "Defs/mapping_defs.h"


class DisplayVoxels : public DisplayObject{
public:
    mapping::VoxelVisu::Seq element;


    /// construction
    DisplayVoxels(){
        type = DisplayObjectType::TYPE_POINT_CLOUD;
    }

    /// create display list
    void createDisplayList(const mapping::VoxelVisu::Seq& voxels);

    /// update display list
    void updateDisplayList();
};

#endif // VOXELS_H_INCLUDED
