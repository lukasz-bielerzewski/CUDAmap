/** @file defs.h
*
* Simulator definitions
*
*/

#ifndef SIMULATOR_DEFS_H_INCLUDED
#define SIMULATOR_DEFS_H_INCLUDED

#include "defs.h"

/// putslam name space
namespace simulator {

    class RogidBody {
    public:
        /// mass
        double mass;

        /// size
        walkers::Vec3 size;

        /// center
        walkers::Vec3 center;

        /// orientation
        walkers::Quaternion orientation;
    };
    //exception class goes here

    enum RenderObjectType{
        PLANE, BOX, SPHERE, CAPSULE, HEIGHTFIELD, LINE, MODEL3D
    };

    class RenderObject {
        public:
            /// set of RenderObjects
            typedef std::vector<RenderObject> Seq;
            /// construction
            RenderObject(void) : type(RenderObjectType::BOX), x(0.0), y(0.0), z(0.0),
                                 scaleX(1.0), scaleY(1.0), scaleZ(1.0), id(0), color{0.4,0.4,0.4,1.0}, mass(0.0){
            }

            /// position and rotation
            walkers::Mat34 mat;

            /// type
            RenderObjectType type;

            /// size
            double x,y,z;
            /// scale factors
            double scaleX,scaleY,scaleZ;

            /// identifier
            size_t id;

            /// mesh id
            size_t meshId;

            /// mesh filename
            std::string meshFilename;

            /// mesh filename
            std::string name;

            /// color
            std::vector<double> color;

            /// mass
            double mass;
    };

    class RenderObjectHeightmap: public RenderObject {
        public:
            std::vector<std::vector<double>> heightmap;
            std::vector<std::vector<std::vector<double>>> heightmapColors;
    };

}

#endif // SIMULATOR__H_INCLUDED
