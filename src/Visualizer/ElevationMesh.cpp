#include "Visualizer/ElevationMesh.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

using namespace simulator;

/// render heightmap
void DisplayElevationMesh::renderHeightmap(const ElevationMap& elevationMap, const RenderObjectHeightmap& object,
                                           bool _showClasses, bool _showCurvature, bool _showFootholds)
{
    double GLmat[16]={object.mat(0,0), object.mat(1,0), object.mat(2,0), object.mat(3,0),
                        object.mat(0,1), object.mat(1,1), object.mat(2,1), object.mat(3,1),
                        object.mat(0,2), object.mat(1,2), object.mat(2,2), object.mat(3,2),
                        object.mat(0,3), object.mat(1,3), object.mat(2,3), object.mat(3,3)};

    if (object.type == RenderObjectType::PLANE)
    {
        glPushMatrix();
        glMultMatrixd(GLmat);
        glColor4d(object.color[0],object.color[1],object.color[2],object.color[3]);
        glScaled(object.x, object.y, object.z);
        glutSolidCube(2.0);
        glPopMatrix();
    }
    else if(object.type == RenderObjectType::HEIGHTFIELD) {
        glPushMatrix();
        float reflectColor[] = { 0.8f, 0.8f, 0.8f, 1.0f };
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, reflectColor);
        GLfloat emissiveLight[] = { 0.1f, 0.1f, 0.1f, 1.0f };
        glMaterialfv(GL_FRONT, GL_EMISSION, emissiveLight);

        for (unsigned int i = 0; i < object.heightmap.size() - 2; i++){
            for (unsigned int j = 0; j < object.heightmap[i].size() - 2; j++) {
                glBegin(GL_TRIANGLE_STRIP);
                size_t colNo; size_t rowNo;
                rowNo=j; colNo=i;
                drawVertex(elevationMap, object,
                           (int)rowNo, (int)colNo, _showClasses, _showCurvature, _showFootholds);
                rowNo=j; colNo=i+1;
                drawVertex(elevationMap, object,
                           (int)rowNo, (int)colNo, _showClasses, _showCurvature, _showFootholds);
                rowNo=j+1; colNo=i;
                drawVertex(elevationMap, object,
                           (int)rowNo, (int)colNo, _showClasses, _showCurvature, _showFootholds);
                rowNo=j+1; colNo=i+1;
                drawVertex(elevationMap, object,
                           (int)rowNo, (int)colNo, _showClasses, _showCurvature, _showFootholds);
                glEnd();
            }
        }
        glPopMatrix();
    }
}

/// draw single vertex of a tringe mesh
void DisplayElevationMesh::drawVertex(const ElevationMap& elevationMap, const RenderObjectHeightmap& object,
                                      int rowNo, int colNo, bool _showClasses, bool _showCurvature, bool _showFootholds)
{
    std::array<double,3> colorClass;
    walkers::Vec3 normal = elevationMap.getNormal((int)rowNo,(int)colNo);
    glNormal3d(normal.x(), normal.y(), normal.z());
    if (_showClasses){
        colorClass = elevationMap.getClassColor((int)rowNo,(int)colNo);
        glColor3d(colorClass[0],colorClass[1],colorClass[2]);
    }
    else if (_showCurvature){
        colorClass = elevationMap.getCurvatureColorClass((int)rowNo,(int)colNo);
        glColor3d(colorClass[0],colorClass[1],colorClass[2]);
    }
    else if (_showFootholds){
        double footholdCost=0;
        size_t footNo=0;
        walkers::Mat34 robotPose = walkers::Mat34::Identity();
        robotPose = walkers::toRotationMat(walkers::Vec3(0,0,0.0));
        if (_showFootholds){
        }
        double _color[3] = {0.4, 0.4, 0.4};

        glColor3d(_color[0],_color[1],_color[2]);
    }
    else
        glColor3d(object.heightmapColors[rowNo][colNo][0], object.heightmapColors[rowNo][colNo][1], object.heightmapColors[rowNo][colNo][2]);
    glVertex3d(elevationMap.toRealX(colNo), elevationMap.toRealY(rowNo), elevationMap.get((int)rowNo,(int)colNo));
}


void DisplayElevationMesh::createDisplayList(const ElevationMap& elevationMap, simulator::RenderObjectHeightmap& _groundObj,
                                             bool _showClasses, bool _showCurvature, bool _showFootholds){
    element = elevationMap;
    groundObj = _groundObj;
    showClasses = _showClasses;
    showCurvature = _showCurvature;
    showFootholds = _showFootholds;
    updateListGL = true;
}

/// update display list
void DisplayElevationMesh::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
        renderHeightmap(element, groundObj, showClasses, showCurvature, showFootholds);
    glEndList();
    element = ElevationMap();
    groundObj = simulator::RenderObjectHeightmap();
}
