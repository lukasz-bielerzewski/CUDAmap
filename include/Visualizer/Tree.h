/** @file Tree.h
 *
 * 3D tree
 *
 */

#ifndef TREE_H_INCLUDED
#define TREE_H_INCLUDED

#include "Defs/defs.h"
#include "Visualizer/DisplayObject.h"
#include "Defs/planner_defs.h"
#include "Defs/mapping_defs.h"

class DisplayTree : public DisplayObject{
public:
    planner::RRTree element;

    /// line width
    double lineWidth;
    /// point size
    double pointSize;
    /// line color
    mapping::RGBA lineColor;
    /// point color
    mapping::RGBA pointColor;

    /// construction
    DisplayTree() : lineWidth(3.0), pointSize(0.002),
        lineColor(mapping::RGBA(255, 0, 0, 255)),
        pointColor(mapping::RGBA(255, 0, 0, 255))
    {
        type = DisplayObjectType::TYPE_TREE;
    }

    /// update display list
    void updateDisplayList();

    /// create display list
    void createDisplayList(const planner::RRTree& tree, double _lineWidth, double _pointSize,
                           const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor);
};

#endif // TREE_H_PATH
