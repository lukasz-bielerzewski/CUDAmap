#include "Visualizer/Mesh.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <GL/glut.h>

void DisplayMesh::createDisplayList(const ObjectsMesh& mesh, const std::string& _filename){
    element = mesh;
    filename = _filename;
    updateListGL = true;
}

/// update display list
void DisplayMesh::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    element.ObjLoad(element.filenames[0]);
    element.loadTextures(0);
    glNewList(listId, GL_COMPILE);
        glScalef((float)element.scale[0], (float)element.scale[1], (float)element.scale[2]);
        element.MeshInit(0);
    glEndList();

    element = ObjectsMesh();
}
