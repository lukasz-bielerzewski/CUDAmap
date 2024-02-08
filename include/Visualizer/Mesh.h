/** @file Mesh.h
 *
 * 3D mesh
 *
 */

#ifndef MESH_H_INCLUDED
#define MESH_H_INCLUDED

#include "Defs/defs.h"
#include "Utilities/objectsMesh.h"
#include "Visualizer/DisplayObject.h"


class DisplayMesh : public DisplayObject{
public:
    /// Line container
    ObjectsMesh element;

    /// unique identifier of the mesh
    std::string filename;

    /// construction
    DisplayMesh() {
        type = DisplayObjectType::TYPE_MESH;
    }

    /// create display list
    void createDisplayList(const ObjectsMesh& mesh, const std::string& _filename);

    /// update display list
    void updateDisplayList();
};

#endif // MESH_H_INCLUDED
