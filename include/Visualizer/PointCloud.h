/** @file PointCloud.h
 *
 * point cloud
 *
 */

#ifndef POINT_CLOUD_H_INCLUDED
#define POINT_CLOUD_H_INCLUDED

#include "Defs/defs.h"
#include "Visualizer/DisplayObject.h"
#include "Defs/mapping_defs.h"


class DisplayPointCloud : public DisplayObject{
public:
    mapping::PointCloud element;

    /// point size
    double pointSize;

    /// construction
    DisplayPointCloud() : pointSize(3.0){
        type = DisplayObjectType::TYPE_POINT_CLOUD;
    }

    /// create display list
    void createDisplayList(const mapping::PointCloud& cloud);

    /// update display list
    void updateDisplayList();
};

#endif // POINT_CLOUD_H_INCLUDED
