/** @file Line.h
 *
 * 3D line
 *
 */

#ifndef LINE_H_INCLUDED
#define LINE_H_INCLUDED

#include "Defs/mathDefs.h"
#include "Defs/mapping_defs.h"
#include "Visualizer/DisplayObject.h"


class DisplayLine : public DisplayObject{
public:
    /// Line container
    walkers::Line3D element;

    /// point size
    double lineWidth;
    /// line color
    mapping::RGBA lineColor;

    /// construction
    DisplayLine() : lineWidth(3.0), lineColor(mapping::RGBA(255, 0, 0, 255)){
        type = DisplayObjectType::TYPE_LINE;
    }

    /// create display list
    void createDisplayList(const walkers::Line3D& line);

    /// update display list
    void updateDisplayList();
};

#endif // LINE_H_INCLUDED
