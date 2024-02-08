/**
 * Project Walkers
 * @author Dominik Belter
 */

#include "Defs/defs.h"
#include "Defs/mapping_defs.h"

#ifndef _SAMPLER_ELLIPSOID_H
#define _SAMPLER_ELLIPSOID_H

class SamplerEllipsoid{
public:
    SamplerEllipsoid(void){
    }
    SamplerEllipsoid(const mapping::Ellipsoid& _ellipsoid) : ellipsoid(_ellipsoid){
    }
    /// get sample
    walkers::Vec3 getSample(void);
    /// create from path
    void createFromPath(walkers::Mat34 startPose, walkers::Mat34 goalPose, double pathLength);
    /// get ellipsoid
    mapping::Ellipsoid getEllipsoid(void);

private:
    /// ellipsoid
    mapping::Ellipsoid ellipsoid;
};

#endif //_SAMPLER_ELLIPSOID_H
