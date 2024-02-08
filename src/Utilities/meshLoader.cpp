/*
3ds loader build using:
 * ---------------- www.spacesimulator.net --------------
 *   ---- Space simulators and 3d engine tutorials ----
 *
 *  Author: Damiano Vitulli <info@spacesimulator.net>
 *
 * Include File: 3dsloader.cpp
 *
 */
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <list>

/// Assimp
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

#include "Utilities/meshLoader.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <stdexcept>
#include <iostream>

walkers::Mat34 toMat34(aiMatrix4x4 mat){
    walkers::Mat34 matrix(walkers::Mat34::Identity());
    matrix(0,0) = mat.a1; matrix(0,1) = mat.a2; matrix(0,2) = mat.a3; matrix(0,3) = mat.a4;
    matrix(1,0) = mat.b1; matrix(1,1) = mat.b2; matrix(1,2) = mat.b3; matrix(1,3) = mat.b4;
    matrix(2,0) = mat.c1; matrix(2,1) = mat.c2; matrix(2,2) = mat.c3; matrix(2,3) = mat.c4;
    matrix(3,0) = mat.d1; matrix(3,1) = mat.d2; matrix(3,2) = mat.d3; matrix(3,3) = mat.d4;
    return matrix;
}

walkers::Mat34 computePose(aiNode* currNode){
    std::vector<walkers::Mat34> chain;

    aiMatrix4x4 mat = currNode->mTransformation;
    aiNode * node = currNode;
    while (node!=NULL){
        mat = node->mTransformation;
        chain.push_back(toMat34(mat));
        node = node->mParent;
    }
    walkers::Mat34 pose(walkers::Mat34::Identity());
    for (int poseNo=(int)chain.size()-1;poseNo>=0;poseNo--){
        pose=pose*chain[poseNo];
    }
    return pose;
}

/// is normal vector initialized?
bool Vertex::isNormalSet(){
    if (this->normal.x!=0.0||this->normal.y!=0.0||this->normal.z!=0.0)
        return true;
    else
        return false;
}

/// transform mesh
void transformMesh(ObjTypeLoader* pObject, const walkers::Mat34& transform){
    for (auto& node : pObject->vertices){
        walkers::Mat34 vert(walkers::Mat34::Identity());
        vert(0,3) = node.vertex.x;
        vert(1,3) = node.vertex.y;
        vert(2,3) = node.vertex.z;
        vert = transform*vert;
        node.vertex.x = (float)vert(0,3);
        node.vertex.y = (float)vert(1,3);
        node.vertex.z = (float)vert(2,3);
    }
}

void findMinMax(ObjTypeLoader* pObject){
    for (size_t dimNo=0; dimNo<3; dimNo++){
        pObject->minMaxXYZ[dimNo].first = std::numeric_limits<double>::max();
        pObject->minMaxXYZ[dimNo].second = std::numeric_limits<double>::min();
    }

    for (const auto& vert : pObject->vertices){
        for (size_t dimNo=0; dimNo<3; dimNo++){
            if (vert.vertex.x < pObject->minMaxXYZ[0].first)
                pObject->minMaxXYZ[0].first = vert.vertex.x;
            if (vert.vertex.x > pObject->minMaxXYZ[0].second)
                pObject->minMaxXYZ[0].second = vert.vertex.x;

            if (vert.vertex.y < pObject->minMaxXYZ[1].first)
                pObject->minMaxXYZ[1].first = vert.vertex.y;
            if (vert.vertex.y > pObject->minMaxXYZ[1].second)
                pObject->minMaxXYZ[1].second = vert.vertex.y;

            if (vert.vertex.z < pObject->minMaxXYZ[2].first)
                pObject->minMaxXYZ[2].first = vert.vertex.z;
            if (vert.vertex.z > pObject->minMaxXYZ[2].second)
                pObject->minMaxXYZ[2].second = vert.vertex.z;
        }
    }
}

char loadAssimp(ObjTypeLoader* pObject, const std::string& pFile)
{
    // Create an instance of the Importer class
    Assimp::Importer importer;
    // And have it read the given file with some example postprocessing
    // Usually - if speed is not the most important aspect for you - you'll
    // propably to request more postprocessing than we do in this example.
    const aiScene* scene = importer.ReadFile( pFile,
                                              aiProcess_CalcTangentSpace       |
                                              aiProcess_Triangulate            |
                                              aiProcess_JoinIdenticalVertices  |
                                              aiProcess_GenUVCoords            |
                                              aiProcess_SortByPType);

    std::string extension = pFile.substr(pFile.find_last_of(".") + 1);
    // If the import failed, report it
    if( !scene) {
        std::cout << "Assimp error: " << importer.GetErrorString() << "\n";
        return false;
    }
    // Now we can access the file's contents.
    if (scene->HasMeshes()){
        pObject->vertices.clear();
        pObject->polygons.clear();

        std::vector<std::string> textures;
        if (scene->HasMaterials()){
            pObject->materials.resize(scene->mNumMaterials);
            for (size_t materialNo=0; materialNo<scene->mNumMaterials; materialNo++){
                aiMaterial* material = scene->mMaterials[materialNo];//Get the current material
                aiString materialName;//The name of the material found in mesh file
                aiReturn ret;//Code which says whether loading something has been successful of not

                ret = material->Get(AI_MATKEY_NAME, materialName);//Get the material name (pass by reference)
                if (ret != AI_SUCCESS) materialName = "";//Failed to find material name so makes var empty

                pObject->materials[materialNo].name = material->GetName().C_Str();
//                std::cout << "material->GetName()" << material->GetName().C_Str() << "\n";
//                std::cout << material->mNumProperties << "\n";
                for (size_t propNo=0;propNo<material->mNumProperties; propNo++){
                    if (material->mProperties[propNo]->mType== aiPTI_Float)
                        pObject->materials[materialNo].type = Material::Type::FloatT;
                    else if (material->mProperties[propNo]->mType== aiPTI_Double)
                        pObject->materials[materialNo].type = Material::Type::DoubleT;
                    else if (material->mProperties[propNo]->mType== aiPTI_Integer)
                        pObject->materials[materialNo].type = Material::Type::IntT;
                    else if (material->mProperties[propNo]->mType== aiPTI_String)
                        pObject->materials[materialNo].type = Material::Type::StringT;

                    std::vector<double> values(material->mProperties[propNo]->mDataLength,0.0);
                    if (pObject->materials[materialNo].type != Material::Type::StringT){
//                        std::cout << "\n";
                        for (size_t dataNo=0;dataNo<material->mProperties[propNo]->mDataLength;dataNo++){
                            values[dataNo] = material->mProperties[propNo]->mData[dataNo];
//                            std::cout << values[dataNo] << ", ";
                        }
//                        std::cout << "\n";
                        if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$mat.shadingm"){
                            pObject->materials[materialNo].shading = values;
                        }
                        else if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$clr.ambient"){
                            pObject->materials[materialNo].colorAmbient = values;
                        }
                        else if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$clr.diffuse"){
                            pObject->materials[materialNo].colorDiffuse = values;
                        }
                        else if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$clr.specular"){
                            pObject->materials[materialNo].colorSpecular = values;
                        }
                        else if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$clr.emissive"){
                            pObject->materials[materialNo].colorEmissive = values;
                        }
                        else if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$mat.shininess"){
                            pObject->materials[materialNo].materialShininess = values;
                        }
                        else if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$clr.transparent"){
                            pObject->materials[materialNo].colorTransparent = values;
                        }
                        else if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$mat.refracti"){
                            pObject->materials[materialNo].materialRefract = values;
                        }
                        else if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$tex.uvwsrc"){
                            pObject->materials[materialNo].textureUVCsrc = values;
                        }
                    }
                    else{
//                        std::cout << "material->mProperties[propNo]->mDataLength " << material->mProperties[propNo]->mDataLength << "\n";
                        if ((std::string)material->mProperties[propNo]->mKey.C_Str()=="$tex.file"){
//                            for (size_t dataNo=0;dataNo<material->mProperties[propNo]->mDataLength;dataNo++){
//                                std::cout << material->mProperties[propNo]->mData[dataNo];
//                            }
                            pObject->materials[materialNo].textureFile =
                                    material->mProperties[propNo]->mData[0];
//                            std::cout << "pObject->materials[materialNo].textureFile " << pObject->materials[materialNo].textureFile <<"\n";
                        }
                    }
//                    std::cout << "material->mProperties[propNo]->mKey " << material->mProperties[propNo]->mKey.C_Str() << "\n";
                }
//                getchar();
                //Diffuse maps
                int numTextures = material->GetTextureCount(aiTextureType_DIFFUSE);//Amount of diffuse textures
                aiString textureName;//Filename of the texture using the aiString assimp structure

                if (numTextures > 0) {
                    ret = material->Get(AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0), textureName);
                    size_t found = pFile.find_last_of("/\\");
                    std::string prefix = pFile.substr(0,found+1);
                    std::string texturePath;
                    if (extension == "dae"){
                        texturePath = prefix + textureName.C_Str();
                    }
                    else{
                        texturePath = prefix + textureName.C_Str();
                    }
                    textures.push_back(texturePath);
                    pObject->textureFilenames.push_back(texturePath);
                }
            }
        }

        std::list<aiNode*> toVisit;
        toVisit.push_back(scene->mRootNode);
        while (toVisit.size()!=0){
            aiNode* currNode = toVisit.front();
            for (size_t childNo = 0; childNo<currNode->mNumChildren; childNo++){
                toVisit.push_back(currNode->mChildren[childNo]);
            }

            // create mesh
            for (size_t meshNo = 0; meshNo<currNode->mNumMeshes; meshNo++){

                size_t startVertIdx = pObject->vertices.size();
                int meshId = currNode->mMeshes[meshNo];
                size_t numVertices =  scene->mMeshes[meshId]->mNumVertices;
                pObject->vertices.resize(startVertIdx+numVertices);
                walkers::Mat34 poseMesh = computePose(currNode);
                for (size_t vertNo = 0; vertNo<numVertices; vertNo++){
                    walkers::Mat34 vert(walkers::Mat34::Identity());
                    vert(0,3) = scene->mMeshes[meshId]->mVertices[vertNo].x;
                    vert(1,3) = scene->mMeshes[meshId]->mVertices[vertNo].y;
                    vert(2,3) = scene->mMeshes[meshId]->mVertices[vertNo].z;
                    vert = poseMesh*vert;
                    pObject->vertices[startVertIdx+vertNo].vertex.x = (float)vert(0,3);
                    pObject->vertices[startVertIdx+vertNo].vertex.y = (float)vert(1,3);
                    pObject->vertices[startVertIdx+vertNo].vertex.z = (float)vert(2,3);

                    if (scene->mMeshes[meshId]->HasVertexColors((unsigned int)(vertNo))){
                        std::cout << "has vertex colors\n";
                    }
                    if (scene->mMeshes[meshId]->HasNormals()){
                        walkers::Mat34 normal(walkers::Mat34::Identity());
                        normal(0,3) = scene->mMeshes[meshId]->mNormals[vertNo].x;
                        normal(1,3) = scene->mMeshes[meshId]->mNormals[vertNo].y;
                        normal(2,3) = scene->mMeshes[meshId]->mNormals[vertNo].z;
                        normal = poseMesh*normal;

                        pObject->vertices[startVertIdx+vertNo].normal.x = (float)normal(0,3);
                        pObject->vertices[startVertIdx+vertNo].normal.y = (float)normal(1,3);
                        pObject->vertices[startVertIdx+vertNo].normal.z = (float)normal(2,3);
                    }
                }
                size_t startPolyIdx = pObject->polygons.size();
                size_t numPolygons = scene->mMeshes[meshId]->mNumFaces;
                pObject->polygons.resize(startPolyIdx+numPolygons);

//                std::cout << "scene->mMeshes[meshId]->mMaterialIndex " << scene->mMeshes[meshId]->mMaterialIndex << "\n";
//                getchar();
                for (size_t polyNo = 0; polyNo<numPolygons; polyNo++){
                    pObject->polygons[polyNo+startPolyIdx].a = startVertIdx + scene->mMeshes[meshId]->mFaces[polyNo].mIndices[0];
                    pObject->polygons[polyNo+startPolyIdx].b = startVertIdx + scene->mMeshes[meshId]->mFaces[polyNo].mIndices[1];
                    pObject->polygons[polyNo+startPolyIdx].c = startVertIdx + scene->mMeshes[meshId]->mFaces[polyNo].mIndices[2];
                    if(scene->mMeshes[meshId]->HasTextureCoords(0)) {
                        if (textures.size()<=scene->mMeshes[meshId]->mMaterialIndex-1)
                            scene->mMeshes[meshId]->mMaterialIndex=(unsigned int)textures.size();
                        if (scene->mMeshes[meshId]->mMaterialIndex-1<textures.size()){
                            auto const &uv1 = scene->mMeshes[meshId]->mTextureCoords[0][scene->mMeshes[meshId]->mFaces[polyNo].mIndices[0]];
//                            std::cout << "scene->mMeshes[meshId]->mMaterialIndex " << scene->mMeshes[meshId]->mMaterialIndex << "\n";
                            pObject->polygons[polyNo+startPolyIdx].textureIdx = scene->mMeshes[meshId]->mMaterialIndex-1;
                            pObject->polygons[polyNo+startPolyIdx].textureCoords[0].u = uv1.x;
                            pObject->polygons[polyNo+startPolyIdx].textureCoords[0].v = uv1.y;
//                            std::cout << "uv read: " << uv1.x << ", " << uv1.y <<"\n";
                            auto const &uv2 = scene->mMeshes[meshId]->mTextureCoords[0][scene->mMeshes[meshId]->mFaces[polyNo].mIndices[1]];
//                            std::cout << "uv read: " << uv2.x << ", " << uv2.y <<"\n";
                            pObject->polygons[polyNo+startPolyIdx].textureCoords[1].u = uv2.x;
                            pObject->polygons[polyNo+startPolyIdx].textureCoords[1].v = uv2.y;
                            auto const &uv3 = scene->mMeshes[meshId]->mTextureCoords[0][scene->mMeshes[meshId]->mFaces[polyNo].mIndices[2]];
//                            std::cout << "uv read: " << uv3.x << ", " << uv3.y <<"\n";
                            pObject->polygons[polyNo+startPolyIdx].textureCoords[2].u = uv3.x;
                            pObject->polygons[polyNo+startPolyIdx].textureCoords[2].v = uv3.y;
                            //                            if (scene->mMeshes[meshId]->mMaterialIndex==2){
//                            std::cout << "meshId " << meshId << "\n";
//                            std::cout << "scene->num " << scene->mNumMeshes << "\n";
//                            std::cout << "polyNo " << polyNo << "\n";
//                            std::cout << "scene->mMeshes[meshId]->mFaces " << scene->mMeshes[meshId]->mNumFaces << "\n";
                            //                                getchar();
                            //                            }
                        }
                    }
                }
            }
            toVisit.pop_front();
        }
    }
    walkers::Mat34 transform;
    transform = walkers::toRotationMat(walkers::Vec3(M_PI/2.0,0.0,0));
    transformMesh(pObject, transform);
    findMinMax(pObject);
    return true;
}

/// load mesh files
char Load(ObjTypeLoader* pObject, const char *pFilename){
    std::string filename(pFilename);
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    if(extension == "3das") {
        return Load3DS(pObject, pFilename);
    }
    else if ((extension == "3ds")||(extension == "dae")||(extension == "obj")) {
        loadAssimp(pObject, filename);
    }
    else
        return (0); // Returns failure
    return (1); // Returns ok
}

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
char Load3DS(ObjTypeLoader* pObject, const char *pFilename)
{
    int i; //Index variable

    FILE *l_file; //File pointer

    unsigned short l_chunk_id; //Chunk identifier
    unsigned int l_chunk_lenght; //Chunk lenght

    unsigned char l_char; //Char variable
    unsigned short l_qty; //Number of elements in each chunk

    unsigned short l_face_flags; //Flag that stores some face information

    struct stat filestatus;
    stat(pFilename, &filestatus);

    if ((l_file=fopen (pFilename, "rb"))== NULL) return 0; //Open the file


    while (ftell (l_file) < filestatus.st_size) {//Loop to scan the whole file
        //getche(); //Insert this command for debug (to wait for keypress for each chuck reading)

        if (fread (&l_chunk_id, 2, 1, l_file)==0)
            std::cout << "read problem\n"; //Read the chunk header
        //printf("ChunkID: %x\n",l_chunk_id);
        if (fread (&l_chunk_lenght, 4, 1, l_file)==0)
            std::cout << "read problem\n";//Read the lenght of the chunk
        //printf("ChunkLenght: %x\n",l_chunk_lenght);

        switch (l_chunk_id) {
        //----------------- MAIN3DS -----------------
        // Description: Main chunk, contains all the other chunks
        // Chunk ID: 4d4d
        // Chunk Lenght: 0 + sub chunks
        //-------------------------------------------
        case 0x4d4d:
            break;

            //----------------- EDIT3DS -----------------
            // Description: 3D Editor chunk, objects layout info
            // Chunk ID: 3d3d (hex)
            // Chunk Lenght: 0 + sub chunks
            //-------------------------------------------
        case 0x3d3d:
            break;

            //--------------- EDIT_OBJECT ---------------
            // Description: Object block, info for each object
            // Chunk ID: 4000 (hex)
            // Chunk Lenght: len(object name) + sub chunks
            //-------------------------------------------
        case 0x4000:
            i=0;
            do{
                if (fread (&l_char, 1, 1, l_file)==0)
                    std::cout << "read problem\n";
                //pObject->name[i]=l_char;
                const char ch(l_char);
                pObject->name.append(&ch);
                i++;
            }while(l_char != '\0' && i<20);
            break;

            //--------------- OBJ_TRIMESH ---------------
            // Description: Triangular mesh, contains chunks for 3d mesh info
            // Chunk ID: 4100 (hex)
            // Chunk Lenght: 0 + sub chunks
            //-------------------------------------------
        case 0x4100:
            break;

            //--------------- TRI_VERTEXL ---------------
            // Description: Vertices list
            // Chunk ID: 4110 (hex)
            // Chunk Lenght: 1 x unsigned short (number of vertices)
            //             + 3 x float (vertex coordinates) x (number of vertices)
            //             + sub chunks
            //-------------------------------------------
        case 0x4110:
            if (fread (&l_qty, sizeof (unsigned short), 1, l_file)==0)
                std::cout << "read problem\n";
            //pObject->verticesQty = l_qty;
            pObject->vertices.resize(l_qty);
            //printf("Number of vertices: %d\n",l_qty);
            for (i=0; i<l_qty; i++) {
                if (fread (&pObject->vertices[i].vertex.x, sizeof(float), 1, l_file)==0)
                    std::cout << "read problem\n";
                //printf("Vertices list x: %f\n",p_object->vertex[i].x);
                if (fread (&pObject->vertices[i].vertex.y, sizeof(float), 1, l_file)==0)
                    std::cout << "read problem\n";
                //printf("Vertices list y: %f\n",p_object->vertex[i].y);
                if (fread (&pObject->vertices[i].vertex.z, sizeof(float), 1, l_file)==0)
                    std::cout << "read problem\n";
                if (isnan(pObject->vertices[i].vertex.x)||isnan(pObject->vertices[i].vertex.y)||isnan(pObject->vertices[i].vertex.z))
                    throw std::runtime_error("3ds file incorrect (vertex)\n");
                if (!isfinite(pObject->vertices[i].vertex.x)||!isfinite(pObject->vertices[i].vertex.y)||!isfinite(pObject->vertices[i].vertex.z))
                    throw std::runtime_error("3ds file incorrect (vertex coord is not finite)\n");
                //printf("Vertices list z: %f\n",p_object->vertex[i].z);
            }
            break;

            //--------------- TRI_FACEL1 ----------------
            // Description: Polygons (faces) list
            // Chunk ID: 4120 (hex)
            // Chunk Lenght: 1 x unsigned short (number of polygons)
            //             + 3 x unsigned short (polygon points) x (number of polygons)
            //             + sub chunks
            //-------------------------------------------
        case 0x4120:
            if (fread (&l_qty, sizeof (unsigned short), 1, l_file)==0)
                std::cout << "read problem\n";
            //pObject->polygonsQty = l_qty;
            pObject->polygons.resize(l_qty);
            // printf("Number of polygons: %d\n",l_qty);
            for (i=0; i<l_qty; i++) {
                unsigned short readVal;
                if (fread (&readVal, sizeof (unsigned short), 1, l_file)==0)
                    std::cout << "read problem\n";
                pObject->polygons[i].a = readVal;
                //printf("Polygon point a: %d\n",p_object->polygon[i].a);
                if (fread (&readVal, sizeof (unsigned short), 1, l_file)==0)
                    std::cout << "read problem\n";
                pObject->polygons[i].b = readVal;
                //printf("Polygon point b: %d\n",p_object->polygon[i].b);
                if (fread (&readVal, sizeof (unsigned short), 1, l_file)==0)
                    std::cout << "read problem\n";
                pObject->polygons[i].c = readVal;
                //printf("Polygon point c: %d\n",p_object->polygon[i].c);
                if (fread (&l_face_flags, sizeof (unsigned short), 1, l_file)==0)
                    std::cout << "read problem\n";
                //printf("Face flags: %x\n",l_face_flags);
            }
            break;

            //------------- TRI_MAPPINGCOORS ------------
            // Description: Vertices list
            // Chunk ID: 4140 (hex)
            // Chunk Lenght: 1 x unsigned short (number of mapping points)
            //             + 2 x float (mapping coordinates) x (number of mapping points)
            //             + sub chunks
            //-------------------------------------------
        case 0x4140:
            if (fread (&l_qty, sizeof (unsigned short), 1, l_file)==0)
                std::cout << "read problem\n";
            for (i=0; i<l_qty; i++)
            {
                if (fread (&pObject->vertices[i].mapcoord.u, sizeof (float), 1, l_file)==0)
                    std::cout << "read problem\n";
                //printf("Mapping list u: %f\n",p_object->mapcoord[i].u);
                if (fread (&pObject->vertices[i].mapcoord.v, sizeof (float), 1, l_file)==0)
                    std::cout << "read problem\n";
                //printf("Mapping list v: %f\n",p_object->mapcoord[i].v);
            }
            break;

            //----------- Skip unknow chunks ------------
            //We need to skip all the chunks that currently we don't use
            //We use the chunk lenght information to set the file pointer
            //to the same level next chunk
            //-------------------------------------------
        default:
            fseek(l_file, l_chunk_lenght-6, SEEK_CUR);
        }
    }
    fclose (l_file); // Closes the file stream
    return (1); // Returns ok
}
