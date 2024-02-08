#include "Visualizer/Voxels.h"
#include <GL/glut.h>

void DisplayVoxels::createDisplayList(const mapping::VoxelVisu::Seq& voxels){
    element = voxels;
    updateListGL = true;
}

/// update display list
void DisplayVoxels::updateDisplayList(){
    glDeleteLists(listId,1);
    listId = glGenLists(1);
    glNewList(listId, GL_COMPILE);
    for( const auto& vox : element) {
        GLfloat		mat[16];
        mat[3] = mat[7] = mat[11] = 0;
        mat[15] = 1;
        mat[12] = (float)vox.pose(0,3); mat[13] = (float)vox.pose(1,3); mat[14] = (float)vox.pose(2,3);

        mat[0] = (float)vox.pose(0,0); mat[1] = (float)vox.pose(1,0); mat[2] = (float)vox.pose(2,0);
        mat[4] = (float)vox.pose(0,1); mat[5] = (float)vox.pose(1,1); mat[6] = (float)vox.pose(2,1);
        mat[8] = (float)vox.pose(0,2); mat[9] = (float)vox.pose(1,2); mat[10] = (float)vox.pose(2,2);

        glPushMatrix();
        glMultMatrixf( mat );

        glColor4ub(vox.color.r, vox.color.g, vox.color.b, vox.color.a);
        glutSolidCube(vox.width);
        glPopMatrix();
    }
    glEndList();

    element.clear();
}
