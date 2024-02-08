/** @file Axis.h
 *
 * 3D Axis
 *
 */

#ifndef AXIS_H_INCLUDED
#define AXIS_H_INCLUDED

#include "Defs/defs.h"
#include "Defs/mapping_defs.h"
#include "Visualizer/DisplayObject.h"

class DisplayAxis : public DisplayObject{
public:
    /// Axis container
    walkers::Mat34 element;

    /// point size
    double lineWidth;
    /// scale
    double scale;

    /// construction
    DisplayAxis() : lineWidth(3.0), scale(1.0){
        type = DisplayObjectType::TYPE_AXIS;
    }

    /// create display list
    void createDisplayList(const walkers::Mat34& axisPose);

    /// update display list
    void updateDisplayList();
};

#endif // AXIS_H_INCLUDED
