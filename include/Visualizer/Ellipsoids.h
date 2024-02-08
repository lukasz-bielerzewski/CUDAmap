/** @file Ellipsoids.h
 *
 * 3D line
 *
 */

#ifndef ELLIPSOIDS_H_INCLUDED
#define ELLIPSOIDS_H_INCLUDED

#include "Defs/defs.h"
#include "Defs/mapping_defs.h"
#include "Visualizer/DisplayObject.h"


class DisplayEllipsoids : public DisplayObject{
public:
    mapping::Ellipsoid::Seq element;

    /// construction
    DisplayEllipsoids(){
        type = DisplayObjectType::TYPE_ELLIPSOID;
    }

    /// create display list
    void createDisplayList(const mapping::Ellipsoid::Seq& ellipsoids);

    /// update display list
    void updateDisplayList();

private:
    /// draw ellipsoid
    void drawEllipsoid(const walkers::Vec3& pos, const walkers::Mat33& covariance, mapping::RGBA color) const;
};

#endif // ELLIPSOIDS_H_INCLUDED
