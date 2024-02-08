/** @file Arrow.h
 *
 * 3D Arrow
 *
 */

#ifndef ARROW_H_INCLUDED
#define ARROW_H_INCLUDED

#include "Defs/mathDefs.h"
#include "Defs/mapping_defs.h"
#include "Visualizer/DisplayObject.h"


class DisplayArrow : public DisplayObject{
public:
    /// Arrow container
    walkers::Arrow3D element;

    /// point size
    double arrowWidth;

    /// Arrow color
    mapping::RGBA arrowColor;

    /// construction
    DisplayArrow() : arrowWidth(3.0), arrowColor(mapping::RGBA(255, 0, 0, 255)){
        type = DisplayObjectType::TYPE_ARROW;
    }

    /// create display list
    void createDisplayList(const walkers::Arrow3D& arrow);

    /// update display list
    void updateDisplayList();
};

#endif // ARROW_H_INCLUDED
