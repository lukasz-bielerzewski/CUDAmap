#include "Utilities/objectsMesh.h"

#include "Defs/stb.h"
#ifdef BUILD_VISUALIZER
#include <GL/glut.h>
#endif
#include <random>
#include <iostream>

ObjectsMesh::ObjectsMesh() {
    this->filenames.clear();
    this->objects.clear();
}

ObjectsMesh::~ObjectsMesh() {
    //#ifdef DEBUG
    //std::cout << "Objects3Ds destructor\n";
    //#endif
}

void ObjectsMesh::loadTextures(int objQty){
#ifdef BUILD_VISUALIZER
    textureIds.resize(objects[objQty].textureFilenames.size());
    for (size_t textureNo=0;textureNo<objects[objQty].textureFilenames.size();textureNo++){
        int width, height, nrChannels;
        std::string textureFilename = objects[objQty].textureFilenames[textureNo];
//        std::cout << "load texture " << textureFilename << "\n";
        unsigned char* data = stbi_load(textureFilename.c_str(), &width, &height, &nrChannels, 0);
        if (data == NULL) {
            std::cout << "Error in loading the image: " << textureFilename << "\n";
            exit(1);
        }
        glGenTextures(1, &textureIds[textureNo]);
        textureName2idx[objects[objQty].textureFilenames[textureNo]] = textureIds[textureNo];
//        std::cout << "textureIds[textureNo] " << textureIds[textureNo] << "\n";
        glBindTexture(GL_TEXTURE_2D, textureIds[textureNo]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glBindTexture(GL_TEXTURE_2D, textureIds[textureNo]);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        }
        else
        {
            std::cout << "Failed to load texture" << std::endl;
        }
    }
#endif
}

void ObjectsMesh::MeshInit(int objQty) {
#ifdef BUILD_VISUALIZER
    int textureIdx = -1;
    if ((objQty<(int)objects.size())&&(objects[objQty].polygons[0].textureIdx>-1)&&(textureIds.size()>0) &&
            (objects[objQty].polygons.size()>0)){
        std::string textureName = objects[objQty].textureFilenames[objects[objQty].polygons[0].textureIdx];
        textureIdx = textureName2idx[textureName];
    }
    bool isTexture = (textureIdx>-1) ? true : false;
    if (isTexture){
//        glDisable(GL_TEXTURE_2D);
//        std::vector<double> ambient = objects[objQty].materials[textureIdx].colorAmbient;
//        GLfloat ambientColor[] = {(GLfloat)ambient[0], (GLfloat)ambient[1], (GLfloat)ambient[2], (GLfloat)ambient[3]};
//        glMaterialfv(GL_FRONT, GL_AMBIENT, ambientColor);

//        std::vector<double> diffuse = objects[objQty].materials[textureIdx].colorDiffuse;
//        GLfloat diffuseColor[] = {(GLfloat)diffuse[0], (GLfloat)diffuse[1], (GLfloat)diffuse[2], (GLfloat)diffuse[3]};
//        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor);

//        std::vector<double> specular = objects[objQty].materials[textureIdx].colorSpecular;
//        GLfloat specular_color[4] = {(GLfloat)specular[0], (GLfloat)specular[1], (GLfloat)specular[2], (GLfloat)specular[3] };
//        glMaterialfv(GL_FRONT, GL_SPECULAR, specular_color);

//        std::vector<double> shininess = objects[objQty].materials[textureIdx].materialShininess;
//        GLfloat shininess_color[4] = {(GLfloat)shininess[0], (GLfloat)shininess[1], (GLfloat)shininess[2], (GLfloat)shininess[3] };
//        glMaterialfv(GL_FRONT, GL_SHININESS, shininess_color);
//                std::vector<double> emission = objects[objQty].materials[textureIdx].colorEmissive;
//                GLfloat emission_color[4] = {(GLfloat)emission[0], (GLfloat)emission[1], (GLfloat)emission[2], (GLfloat)emission[3] };
//                glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emission_color);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureIdx);    // Select Our Texture
    }
    else{
//        glDisable(GL_TEXTURE_2D);
//        GLfloat diffuseColor[] = {0.5, 0.5, 0.5, 0.0};
//        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseColor);

//        float reflectColor[] = { 0.5f, 0.5f, 0.5f, 0.0f };
//        glMaterialfv(GL_FRONT, GL_AMBIENT, reflectColor);

//        GLfloat specular_color[4] = {0.0, 0.0, 0.0, 0.0};
//        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular_color);

//        GLfloat shininess_color[4] = {0.5, 0.5, 0.5, 1.0};
//        glMaterialfv(GL_FRONT, GL_SHININESS, shininess_color);
    }

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(50, 205);
    std::uniform_int_distribution<> distrib2(-50, 50);
    int color = distrib(gen);

    double normal[3];
    std::vector<walkers::Vec3> vertices(3,walkers::Vec3());
    std::vector<walkers::Vec3> normals(3,walkers::Vec3());
    glColor3ub((GLubyte)(color+distrib2(gen)),(GLubyte)(color+distrib2(gen)),(GLubyte)(color+distrib2(gen)));
    glBegin(GL_TRIANGLES);
    for (size_t j=0;j<objects[objQty].polygons.size();j++) {
        if (isTexture&&objects[objQty].polygons[j].textureIdx>-1){
            std::string textureName = objects[objQty].textureFilenames[objects[objQty].polygons[j].textureIdx];

            int newTextureIdx = textureName2idx[textureName];
            if (newTextureIdx!=textureIdx){
                textureIdx = newTextureIdx;
                glEnd();

//                glDisable(GL_TEXTURE_2D);
//                std::vector<double> ambient = objects[objQty].materials[objects[objQty].polygons[j].textureIdx].colorAmbient;
//                GLfloat ambientColor[] = {(GLfloat)ambient[0], (GLfloat)ambient[1], (GLfloat)ambient[2], (GLfloat)ambient[3]};
//                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambientColor);

//                std::vector<double> diffuse = objects[objQty].materials[objects[objQty].polygons[j].textureIdx].colorDiffuse;
//                GLfloat diffuseColor[] = {(GLfloat)diffuse[0], (GLfloat)diffuse[1], (GLfloat)diffuse[2], (GLfloat)diffuse[3]};
//                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseColor);

//                std::vector<double> specular = objects[objQty].materials[objects[objQty].polygons[j].textureIdx].colorSpecular;
//                GLfloat specular_color[4] = {(GLfloat)specular[0], (GLfloat)specular[1], (GLfloat)specular[2], (GLfloat)specular[3] };
//                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular_color);

//                std::vector<double> emission = objects[objQty].materials[objects[objQty].polygons[j].textureIdx].colorEmissive;
//                GLfloat emission_color[4] = {(GLfloat)emission[0], (GLfloat)emission[1], (GLfloat)emission[2], (GLfloat)emission[3] };
//                glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emission_color);

                glEnable(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D, textureIdx);    // Select Our Texture
                glBegin(GL_TRIANGLES);
            }
        }
        if (objects[objQty].polygons[j].a<objects[objQty].vertices.size()&&
                objects[objQty].polygons[j].b<objects[objQty].vertices.size()&&
                objects[objQty].polygons[j].c<objects[objQty].vertices.size()){
            vertices[0].x()=(objects[objQty].vertices[ objects[objQty].polygons[j].a ].vertex.x);
            vertices[0].y()=(objects[objQty].vertices[ objects[objQty].polygons[j].a ].vertex.y);
            vertices[0].z()=(objects[objQty].vertices[ objects[objQty].polygons[j].a ].vertex.z);
            normals[0].x()=(objects[objQty].vertices[ objects[objQty].polygons[j].a ].normal.x);
            normals[0].y()=(objects[objQty].vertices[ objects[objQty].polygons[j].a ].normal.y);
            normals[0].z()=(objects[objQty].vertices[ objects[objQty].polygons[j].a ].normal.z);

            vertices[1].x()=(objects[objQty].vertices[ objects[objQty].polygons[j].b ].vertex.x);
            vertices[1].y()=(objects[objQty].vertices[ objects[objQty].polygons[j].b ].vertex.y);
            vertices[1].z()=(objects[objQty].vertices[ objects[objQty].polygons[j].b ].vertex.z);
            normals[1].x()=(objects[objQty].vertices[ objects[objQty].polygons[j].b ].normal.x);
            normals[1].y()=(objects[objQty].vertices[ objects[objQty].polygons[j].b ].normal.y);
            normals[1].z()=(objects[objQty].vertices[ objects[objQty].polygons[j].b ].normal.z);

            vertices[2].x()=(objects[objQty].vertices[ objects[objQty].polygons[j].c ].vertex.x);
            vertices[2].y()=(objects[objQty].vertices[ objects[objQty].polygons[j].c ].vertex.y);
            vertices[2].z()=(objects[objQty].vertices[ objects[objQty].polygons[j].c ].vertex.z);
            normals[2].x()=(objects[objQty].vertices[ objects[objQty].polygons[j].c ].normal.x);
            normals[2].y()=(objects[objQty].vertices[ objects[objQty].polygons[j].c ].normal.y);
            normals[2].z()=(objects[objQty].vertices[ objects[objQty].polygons[j].c ].normal.z);
        }

        if (objects[objQty].vertices[ objects[objQty].polygons[j].a ].isNormalSet()&&
                objects[objQty].vertices[ objects[objQty].polygons[j].b ].isNormalSet()&&
                objects[objQty].vertices[ objects[objQty].polygons[j].c ].isNormalSet()){
            glNormal3d (normals[0].x(),normals[0].y(),normals[0].z());
            if (isTexture){
                glTexCoord2f(objects[objQty].polygons[j].textureCoords[0].u, objects[objQty].polygons[j].textureCoords[0].v);
            }
            glVertex3d(vertices[0].x(), vertices[0].y(), vertices[0].z());
            glNormal3d (normals[1].x(),normals[1].y(),normals[1].z());
            if (isTexture){
                glTexCoord2f(objects[objQty].polygons[j].textureCoords[1].u, objects[objQty].polygons[j].textureCoords[1].v);
            }
            glVertex3d(vertices[1].x(), vertices[1].y(), vertices[1].z());
            glNormal3d (normals[2].x(),normals[2].y(),normals[2].z());
            if (isTexture){
                glTexCoord2f(objects[objQty].polygons[j].textureCoords[2].u, objects[objQty].polygons[j].textureCoords[2].v);
            }
            glVertex3d(vertices[2].x(), vertices[2].y(), vertices[2].z());
        }
        else{
            calcNormal(vertices, normal);
            glNormal3d (normal[0],normal[1],normal[2]);
            glVertex3d(vertices[0].x(), vertices[0].y(), vertices[0].z());
            glVertex3d(vertices[1].x(), vertices[1].y(), vertices[1].z());
            glVertex3d(vertices[2].x(), vertices[2].y(), vertices[2].z());
        }
    }
    glEnd();
    if (isTexture){
        glDisable(GL_TEXTURE_2D);
    }
#endif
}

size_t ObjectsMesh::ObjLoad(const std::string& objectFilename) {
    for(size_t i=0; i<objects.size(); ++i) {
        if(!filenames[i].compare(objectFilename)) {
            return i+1;
        }
    }
    std::string sources = objectFilename;
    objects.resize(objects.size()+1);
    if (Load(&objects[objects.size()-1],sources.c_str())==0) return(0);
    filenames.push_back(objectFilename.c_str());
    return (objects.size());
}

void ObjectsMesh::calcNormal(const std::vector<walkers::Vec3>& v, double *out) {
    double v1[3],v2[3];
    const int x = 0;
    const int y = 1;
    const int z = 2;

    v1[x] = v[0].x() - v[1].x();
    v1[y] = v[0].y() - v[1].y();
    v1[z] = v[0].z() - v[1].z();

    v2[x] = v[1].x() - v[2].x();
    v2[y] = v[1].y() - v[2].y();
    v2[z] = v[1].z() - v[2].z();

    out[x] = v1[y]*v2[z] - v1[z]*v2[y];
    out[y] = v1[z]*v2[x] - v1[x]*v2[z];
    out[z] = v1[x]*v2[y] - v1[y]*v2[x];

    ReduceToUnit(out);
}

void ObjectsMesh::ReduceToUnit(double vector[3]) {
    double length = sqrt(  (vector[0]*vector[0]) + (vector[1]*vector[1]) + (vector[2]*vector[2]));
    if(length == 0.0)
        length = 1.0;

    vector[0] /= length;
    vector[1] /= length;
    vector[2] /= length;
}
