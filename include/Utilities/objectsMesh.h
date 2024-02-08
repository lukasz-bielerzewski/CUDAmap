#ifndef H_OBJECTS3DS
#define H_OBJECTS3DS

#include "meshLoader.h"

class ObjectsMesh {
public:
    ObjectsMesh();
    ~ObjectsMesh();
	
    /// openGL initialization
    void MeshInit(int objQty);
    /// load object
    size_t ObjLoad(const std::string& objectFilename);

    /// compute normal vector
    void calcNormal(const std::vector<walkers::Vec3>& v, double* out);
    /// reduce vector to unit
    void ReduceToUnit(double vector[3]);
    /// load texture
    void loadTextures(int objQty);

    std::vector<ObjTypeLoader> objects;
    std::vector<std::string> filenames;

    /// texture name to idx
    std::map<std::string,int> textureName2idx;
    /// textures ids
    std::vector<unsigned int> textureIds;
    std::array<double,3> scale = {1.0,1.0,1.0};
};
#endif
