/** @file BSplineSE3.h
 *
 * SE3 bspline
 *
 */

#ifndef DISPLAY_BSPLINESE3_H_INCLUDED
#define DISPLAY_BSPLINESE3_H_INCLUDED

#include "Defs/mathDefs.h"
#include "Defs/mapping_defs.h"
#include "Visualizer/DisplayObject.h"

class DisplayBSplineSE3 : public DisplayObject{
public:
    /// BSpline container
    walkers::BSplineSE3 element;

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
    /// frames scale
    double framesScale;
    /// draw ctrl points
    bool drawCtrlPoints;
    /// draw ctrl lines
    bool drawCtrlLines;
    /// draw frames
    bool drawFrames;

    /// construction
    DisplayBSplineSE3() : lineWidth(3.0), lineColor(mapping::RGBA(255, 0, 0, 255)),
                         ctrlLineColor(mapping::RGBA(0, 255, 0, 255)), ctrlPointSize(7), pointsNo(20),
                         framesScale(0.05), drawCtrlPoints(true), drawCtrlLines(true), drawFrames(true){
        type = DisplayObjectType::TYPE_BSPLINESE3;
    }

    /// create display list
    void createDisplayList(const walkers::BSplineSE3& line);

    /// update display list
    void updateDisplayList();

private:
    /// draw axis
    void drawAxis(double _scale);
};

#endif // DISPLAY_BSPLINESE3_H_INCLUDED
