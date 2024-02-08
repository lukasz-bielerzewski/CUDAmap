/** @file BSpline3D.h
 *
 * 3D Bspline
 *
 */

#ifndef DISPLAY_BSPLINE3D_H_INCLUDED
#define DISPLAY_BSPLINE3D_H_INCLUDED

#include "Defs/mathDefs.h"
#include "Defs/mapping_defs.h"
#include "Visualizer/DisplayObject.h"

class DisplayBSpline3D : public DisplayObject{
public:
    /// Bspline container
    walkers::BSpline3D element;

    /// point size
    double lineWidth;
    /// line color
    mapping::RGBA lineColor;
    /// control line color
    mapping::RGBA ctrlLineColor;
    /// control point size
    double ctrlPointSize;
    /// points No
    size_t pointsNo;
    /// draw ctrl points
    bool drawCtrlPoints;
    /// draw ctrl lines
    bool drawCtrlLines;

    /// construction
    DisplayBSpline3D() : lineWidth(3.0), lineColor(mapping::RGBA(255, 0, 0, 255)),
                         ctrlLineColor(mapping::RGBA(0, 255, 0, 255)), ctrlPointSize(7), pointsNo(20),
                         drawCtrlPoints(true), drawCtrlLines(true){
        type = DisplayObjectType::TYPE_BSPLINE3D;
    }

    /// create display list
    void createDisplayList(const walkers::BSpline3D& line);

    /// update display list
    void updateDisplayList();
};

#endif // DISPLAY_BSPLINE3D_H_INCLUDED
