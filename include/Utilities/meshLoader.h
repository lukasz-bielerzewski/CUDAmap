/* 3ds loader build using:
 * ---------------- www.spacesimulator.net --------------
 *   ---- Space simulators and 3d engine tutorials ----
 *
 *  Author: Damiano Vitulli <info@spacesimulator.net>
 *
 * File header: meshLoader.h
 *  
 */

#ifndef _MESHLOADER_H_
#define _MESHLOADER_H_

#include <Defs/defs.h>
#include <assimp/scene.h>           // Output data structure

// Our vertex type
class VertexType {
public:
    float x,y,z;
    VertexType() : x(0), y(0), z(0){
    }
};

// The mapcoord type, 2 texture coordinates for each vertex
class MapcoordType {
public:
    float u,v;
    MapcoordType() : u(0), v(0){
    }
};

// The polygon (triangle), 3 numbers that aim 3 vertices
class PolygonType {
public:
    /// indexes of the vertices
    size_t a,b,c;
    /// texture coords
    std::array<MapcoordType,3> textureCoords;
    /// texture index
    int textureIdx;
    PolygonType(): a(0),b(0),c(0), textureIdx(-1){
    }
};

class Vertex {
public:
    VertexType		vertex;
    VertexType		normal;
    MapcoordType	mapcoord;
    Vertex(){
    }

    bool isNormalSet();
};

class Material{
public:
    enum Type
    {
        DoubleT,
        FloatT,
        IntT,
        StringT
    };
    std::string name;
    Type type;
    std::vector<double> shading;
    std::vector<double> colorAmbient;
    std::vector<double> colorDiffuse;
    std::vector<double> colorSpecular;
    std::vector<double> colorEmissive;
    std::vector<double> colorTransparent;
    std::vector<double> materialShininess;
    std::vector<double> materialOpacity;
    std::vector<double> materialRefract;
    std::string textureFile;
    std::vector<double> textureUVCsrc;
};

class ObjTypeLoader{
public:
    std::string name;		// Name of the object

    std::vector<std::string> textureFilenames;
    std::vector<Material> materials;

    std::vector<Vertex> vertices;
    std::array<std::pair<double,double>,3> minMaxXYZ;
    std::vector<PolygonType> polygons;
};

/// convert assimp mat to mat34
walkers::Mat34 toMat34(aiMatrix4x4 mat);
/// get pose of the node from the tree
walkers::Mat34 computePose(aiNode* currNode);
/// transform mesh
void transformMesh(ObjTypeLoader* pObject, const walkers::Mat34& transform);
/**********************************************************
 *
 * FUNCTION Load3DS (obj_type_ptr, char *)
 *
 * This function loads a mesh from a 3ds file.
 * Please note that we are loading only the vertices, polygons and mapping lists.
 * If you need to load meshes with advanced features as for example:
 * multi objects, materials, lights and so on, you must insert other chunk parsers.
 *
 *********************************************************/
char Load3DS (ObjTypeLoader* pObject, const char *pFilename);
char loadAssimp(ObjTypeLoader* pObject, const std::string& pFile);
/// load mesh files
char Load(ObjTypeLoader* pObject, const char *pFilename);
void findMinMax(ObjTypeLoader* pObject);
#endif // _LOADER3DS_H_
