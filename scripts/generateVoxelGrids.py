#!/bin/python3

import os

def listfiles(rootdir):
   for file in os.listdir(rootdir):
      if file.endswith(".npz"):
         print(os.path.join(rootdir, file))
         # os.system('cd ../build/bin')
         bin = './demoNumpyLoad \"' + rootdir + '/\" \"' + file + '\"'
         print(bin)
         os.system(bin)
         # os.system('cd ../../scripts')

def listdirs(rootdir):
   for file in os.listdir(rootdir):
      d = os.path.join(rootdir, file)
      if os.path.isdir(d):
         print(d)
         listfiles(d)
         listdirs(d)


rootdir = '/home/dominik/uczelnia/badania/artykul_objects_recon/results/voxel_grids'
listdirs(rootdir)
