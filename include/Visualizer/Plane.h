/** @file Plane.h
 *
 * 3D Plane
 *
 */

#ifndef PLANE_H_INCLUDED
#define PLANE_H_INCLUDED

#include "Defs/mathDefs.h"
#include "Defs/mapping_defs.h"
#include "Visualizer/DisplayObject.h"


class DisplayPlanes : public DisplayObject{
public:
    /// Plane container
    std::vector<walkers::Plane3D> element;

    /// Plane color
    mapping::RGBA planeColor;

    /// construction
    DisplayPlanes() : planeColor(mapping::RGBA(255, 0, 0, 255)){
        type = DisplayObjectType::TYPE_PLANE;
    }

    /// create display list
    void createDisplayList(const std::vector<walkers::Plane3D>& Plane, const mapping::RGBA& _planeColor);

    /// update display list
    void updateDisplayList();
};

#endif // PLANE_H_INCLUDED
