/** @file ElevationMesh.h
 *
 * elevation map -- mesh
 *
 */

#ifndef ELEVATION_MESH_H_INCLUDED
#define ELEVATION_MESH_H_INCLUDED

#include "Defs/defs.h"
#include "Visualizer/DisplayObject.h"
#include "Mapping/elevationMap.h"
#include "Defs/simulator_defs.h"

class DisplayElevationMesh : public DisplayObject{
public:
    ElevationMap element;
    simulator::RenderObjectHeightmap groundObj;
    bool showClasses;
    bool showCurvature;
    bool showFootholds;

    /// construction
    DisplayElevationMesh(){
        type = DisplayObjectType::TYPE_ELEVATION_MESH;
    }

    /// create display list
    void createDisplayList(const ElevationMap& elevationMap, simulator::RenderObjectHeightmap& _groundObj,
                           bool _showClasses, bool _showCurvature, bool _showFootholds);

    /// update display list
    void updateDisplayList();

private:
    /// render heightmap
    void renderHeightmap(const ElevationMap& elevationMap, const simulator::RenderObjectHeightmap& object,
                         bool showClasses, bool _showCurvature, bool _showFootholds);

    /// draw single vertex of a tringe mesh
    void drawVertex(const ElevationMap& elevationMap, const simulator::RenderObjectHeightmap& object,
                    int rowNo, int colNo, bool showClasses, bool _showCurvature, bool _showFootholds);
};

#endif // ELEVATION_MESH_H_INCLUDED
