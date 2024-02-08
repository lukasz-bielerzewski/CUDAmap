#!/usr/bin/env python3

import os

def coord2xy(row,col,map_width, map_height, raster_x, raster_y):
    coordx = -map_width*raster_x/2.0+row*raster_x
    coordy = -map_height*raster_y/2.0+col*raster_y
    return (coordx, coordy)

class Box:

    width = 1.0
    depth = 1.0
    height = 1.0
    center_x = 0.0
    center_y = 0.0

    def __init__(self, _width, _depth, _height, _center_x, _center_y):
        self.width = _width
        self.depth = _depth
        self.height = _height
        self.center_x = _center_x
        self.center_y = _center_y

    def isCollision(self,x,y):
        if (x<self.center_x+(self.width/2.0) and x>self.center_x-(self.width/2.0) and y<self.center_y+(self.depth/2.0) and y>self.center_y-(self.depth/2.0)) :
            return True
        else :
            return False

map_width = 600
map_height = 600
raster_x = 0.025
raster_y = 0.025

box1 = Box(_width=4.2, _depth = 0.2, _height = 1.0, _center_x = 0.0, _center_y = 0.0)
box2 = Box(_width=0.2, _depth = 4.0, _height = 1.0, _center_x = 2.0, _center_y = -2.0)
box3 = Box(_width=0.2, _depth = 4.0, _height = 1.0, _center_x = -2.0, _center_y = -2.0)
# box4 = Box(_width=1.5, _depth = 0.2, _height = 1.0, _center_x = 1.35, _center_y = -4.0)
# box5 = Box(_width=1.5, _depth = 0.2, _height = 1.0, _center_x = -1.35, _center_y = -4.0)
box4 = Box(_width=1.25, _depth = 0.2, _height = 1.0, _center_x = 1.5, _center_y = -4.0)
box5 = Box(_width=1.25, _depth = 0.2, _height = 1.0, _center_x = -1.5, _center_y = -4.0)

file_path = os.path.abspath(os.path.dirname(__file__))
# output_map_filename = file_path + '/map_bug_trap.dat'
output_map_filename = file_path + '/map_bug_trap_anymal.dat'
file = open(output_map_filename, 'w')

file.write("# sizeX sizeY rasterX rasterY\n")
file.write(str(map_width) + " " + str(map_height) + " " + str(raster_x) + " " + str(raster_y) +"\n")

for col in range(0, map_width):
    for row in range(0, map_height):
        coordx,coordy = coord2xy(row,col,map_width, map_height, raster_x, raster_y)
        if (box1.isCollision(coordx,coordy)) :
            file.write("{:.6f} ".format(box1.height))
        elif (box2.isCollision(coordx,coordy)) :
            file.write("{:.6f} ".format(box2.height))
        elif (box3.isCollision(coordx,coordy)) :
            file.write("{:.6f} ".format(box3.height))
        elif (box4.isCollision(coordx,coordy)) :
            file.write("{:.6f} ".format(box4.height))
        elif (box5.isCollision(coordx,coordy)) :
            file.write("{:.6f} ".format(box5.height))
        else :
            file.write("{:.6f} ".format(0.0))
    file.write("\n")

file.close()