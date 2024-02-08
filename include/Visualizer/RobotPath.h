/** @file RobotPath.h
 *
 * 3D robot path
 *
 */

#ifndef ROBOTPATH_H_INCLUDED
#define ROBOTPATH_H_INCLUDED

#include "Defs/defs.h"
#include "Visualizer/DisplayObject.h"
#include "Defs/planner_defs.h"
#include "Defs/mapping_defs.h"

class DisplayRobotPath : public DisplayObject{
public:
    std::vector<planner::RobotState3D> element;

    /// line width for body
    double lineWidthBody;
    /// point size for body
    double pointSizeBody;
    /// line width for feet
    double lineWidthFeet;
    /// point size for feet
    double pointSizeFeet;
    /// line color body
    mapping::RGBA lineColorBody;
    /// point color body
    mapping::RGBA pointColorBody;
    /// line color feet
    mapping::RGBA lineColorFeet;
    /// point color feet
    mapping::RGBA pointColorFeet;

    /// construction
    DisplayRobotPath() : lineWidthBody(3.0), pointSizeBody(0.002),
        lineWidthFeet(3.0), pointSizeFeet(0.002),
        lineColorBody(mapping::RGBA(255, 0, 0, 255)),
        pointColorBody(mapping::RGBA(255, 0, 0, 255))
    {
        type = DisplayObjectType::TYPE_PATH;
    }

    /// create display list
    void createDisplayList(const std::vector<planner::RobotState3D>& robotPath, double _lineWidthBody, double _pointSizeBody,
                           const mapping::RGBA& _lineColorBody, const mapping::RGBA& _pointColorBody,
                           double _lineWidthFeet, double _pointSizeFeet,
                           const mapping::RGBA& _lineColorFeet, const mapping::RGBA& _pointColorFeet);

    /// update display list
    void updateDisplayList();
};

#endif // LINE_H_PATH
