/** @file DisplayObject.h
 *
 * Display Objct
 *
 */

#ifndef DISPLAY_OBJECT_H_INCLUDED
#define DISPLAY_OBJECT_H_INCLUDED

#include "Defs/defs.h"
#include <QGLViewer/qglviewer.h>
#include <mutex>

/// Display object type
enum DisplayObjectType {
    /// Mesh
    TYPE_MESH,
    /// Ellipsoid
    TYPE_ELLIPSOID,
    /// Point cloud
    TYPE_POINT_CLOUD,
    /// Line
    TYPE_LINE,
    /// B-SPline3D
    TYPE_BSPLINE3D,
    /// B-SPlineSE3
    TYPE_BSPLINESE3,
    /// Elevation Mesh
    TYPE_ELEVATION_MESH,
    /// Path
    TYPE_PATH,
    /// Tree
    TYPE_TREE,
    /// Polynomial
    TYPE_POLYNOMIAL,
    /// Voxel
    TYPE_VOXEL,
    /// Plane
    TYPE_PLANE,
    /// Map NDTOM
    TYPE_MAPNDTOM,
    /// Trajectory SE3
    TYPE_TRAJECTORYSE3,
    /// Axis
    TYPE_AXIS,
    /// Arrow
    TYPE_ARROW,
    /// Triangles
    TYPE_TRIANGLE
};

class DisplayObject{ // be carefull, the object is non copyable
public:
    /// set of Robot States
    typedef std::vector<DisplayObject> Seq;
    /// name
    std::string name;
    /// type
    DisplayObjectType type;
    /// mutex
    std::mutex mutex;// be carefull, the object is non copyable
    /// list identifier
    GLuint listId;
    /// init object (display list)
    bool initObject;
    /// pose of the object
    walkers::Mat34 pose;
    /// update display list
    bool updateListGL;
    /// scale
    walkers::Vec3 scale;
    /// is visible
    bool isVisible = true;

    inline DisplayObject() : listId(std::numeric_limits<unsigned int>::max()), initObject(false),
        pose(walkers::Mat34::Identity()), updateListGL(false), scale(walkers::Vec3(1.0,1.0,1.0)){}

    // Move initialization
    DisplayObject(DisplayObject&& other) {
        std::lock_guard<std::mutex> lock(other.mutex);
        name = std::move(other.name);
        type = std::move(other.type);
        listId = std::move(other.listId);
        initObject = std::move(other.initObject);
        pose = std::move(other.pose);
        scale = walkers::Vec3(1.0,1.0,1.0);
    }

    // Copy initialization
    DisplayObject(const DisplayObject& other) {
        name = other.name;
        type = other.type;
        listId = other.listId;
        initObject = other.initObject;
        pose = other.pose;
        scale = walkers::Vec3(1.0,1.0,1.0);
    }

    /// update display list
    virtual void updateDisplayList() = 0;

    /// draw
    void draw(void){
        if (isVisible){
//            std::cout << "lock do\n";
            //mutex.lock(); //do usuniecia
//            std::cout << "locked do\n";
            if (updateListGL){
                mutex.lock(); // do odkomentowania
                updateDisplayList();
                updateListGL = false;
                mutex.unlock(); // do odkomentowania
            }
            if (listId<1e6){
                glPushMatrix();
                double GLmat[16]={pose(0,0), pose(1,0), pose(2,0), pose(3,0),
                                  pose(0,1), pose(1,1), pose(2,1), pose(3,1),
                                  pose(0,2), pose(1,2), pose(2,2), pose(3,2),
                                  pose(0,3), pose(1,3), pose(2,3), pose(3,3)};
                glMultMatrixd(GLmat);
                glScaled(scale.x(),scale.y(),scale.z());
                glCallList(listId);
                glPopMatrix();
            }
            else{
                updateListGL = true;
            }
//            std::cout << "unlock do\n";
            //mutex.unlock(); //do usuniecia
//            std::cout << "unlocked do\n";
        }
    }
};

#endif // DISPLAY_OBJECT_H_INCLUDED
