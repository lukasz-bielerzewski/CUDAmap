/** @file TrajectorySE3.h
 *
 * 3D TrajectorySE3
 *
 */

#ifndef TrajectorySE3_H_INCLUDED
#define TrajectorySE3_H_INCLUDED

#include "Defs/defs.h"
#include "Visualizer/DisplayObject.h"
#include "Utilities/recorder.h"
#include "Defs/mapping_defs.h"

class DisplayTrajectorySE3 : public DisplayObject{
public:
    Recorder6D element;

    /// line width
    double lineWidth;
    /// point size
    double pointSize;
    /// line color
    mapping::RGBA lineColor;
    /// point color
    mapping::RGBA pointColor;
    /// current time
    double currentTime;
    /// draw whole trajectory
    bool drawAllTraj;

    /// construction
    DisplayTrajectorySE3() : element(0,"",""), lineWidth(3.0), pointSize(0.002),
        lineColor(mapping::RGBA(255, 0, 0, 255)),
        pointColor(mapping::RGBA(255, 0, 0, 255))
    {
        type = DisplayObjectType::TYPE_TRAJECTORYSE3;
    }

    /// create display list
    void createDisplayList(const Recorder6D& trajectory, double _lineWidth, double _pointSize,
                           const mapping::RGBA& _lineColor, const mapping::RGBA& _pointColor, double _currentTime,
                           bool _drawAllTraj);

    /// update display list
    void updateDisplayList();

    /// set current time
    void setCurrentTime(double _currentTime);

private:
    /// draw axis
    void drawAxis(double _scale);
};

#endif // LINE_H_TrajectorySE3
