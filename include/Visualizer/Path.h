/** @file Path.h
 *
 * 3D path
 *
 */

#ifndef PATH_H_INCLUDED
#define PATH_H_INCLUDED

#include "Defs/defs.h"
#include "Visualizer/DisplayObject.h"
#include "Defs/planner_defs.h"
#include "Defs/mapping_defs.h"

class DisplayPath : public DisplayObject{
public:
    std::vector<planner::PoseSE3> element;

    /// line width
    double lineWidth;
    /// point size
    double pointSize;
    /// line color
    mapping::RGBA lineColor;
    /// point color
    mapping::RGBA pointColor;

    /// construction
    DisplayPath() : lineWidth(3.0), pointSize(0.002),
        lineColor(mapping::RGBA(255, 0, 0, 255)),
        pointColor(mapping::RGBA(255, 0, 0, 255))
    {
        type = DisplayObjectType::TYPE_PATH;
    }

    /// create display list
    void createDisplayList(const std::vector<planner::PoseSE3>& path, double _lineWidth, double _pointSize,
                           const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);

    /// update display list
    void updateDisplayList();
};

#endif // LINE_H_PATH
