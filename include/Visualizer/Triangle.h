/** @file Triangle.h
 *
 * 3D Triangle
 *
 */

#ifndef TRIANGLE_H_INCLUDED
#define TRIANGLE_H_INCLUDED

#include "Defs/mathDefs.h"
#include "Defs/mapping_defs.h"
#include "Visualizer/DisplayObject.h"


class DisplayTriangles : public DisplayObject{
public:
    /// Triangle container
    std::vector<walkers::Triangle3D> element;

    /// Triangle color
    mapping::RGBA triangleColor;

    /// construction
    DisplayTriangles() : triangleColor(mapping::RGBA(255, 0, 0, 255)){
        type = DisplayObjectType::TYPE_TRIANGLE;
    }

    /// create display list
    void createDisplayList(const std::vector<walkers::Triangle3D>& triangles, const mapping::RGBA& _trianglesColor);

    /// update display list
    void updateDisplayList();
};

#endif // PLANE_H_INCLUDED
