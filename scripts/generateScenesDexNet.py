#!/usr/bin/env python3

import glob, os
import pathlib
from signal import pause
import lxml.etree as ET
import sys
import subprocess
import re
import shlex
from subprocess import Popen, PIPE
import datetime
import numpy as np
import shutil 

ON_POSIX = 'posix' in sys.builtin_module_names

number_of_files = 5000
PATTERN = "dataset/rgb*.png"
models_prefix = "/home/dominik/Sources/walkers/resources/models/shapeNetDatasetDexNetTrain"
#models_prefix = "/home/dominik/Sources/walkers/resources/models/shapeNetDatasetDexNetTest"

print (os.getcwd())
os.chdir(os.path.dirname("../build/bin/"))
print (os.getcwd())

#train
categories = ["shapeNetGraspable_30instances_bottle",
    "shapeNetGraspable_30instances_can",
    "shapeNetGraspable_30instances_helmet", "shapeNetGraspable_30instances_jar",
    "shapeNetGraspable_30instances_laptop", "shapeNetGraspable_30instances_mug"]

#test
#categories = [
#"Bottle",
#    "Can",
#    "Helmet",
#    "Jar",
#    "Laptop", "Mug"]

# todo: set scene graspable in configGlobal.xml

for category in categories :
    tree_config = ET.parse('../../resources/scene_graspable_Dexnet_train.xml')
    root_config = tree_config.getroot()
    
    #configure scene generator
#    models_path_tables = models_prefix + "/" + category + "/" + "tables"
    models_path_objs = models_prefix + "/" + category + "/" + "objects"
    for child_config in root_config:
        if (child_config.tag == "object") :
#            if child_config.attrib["name"] == "table" :
#                child_config.attrib["file"] = models_path_tables
            if child_config.attrib["name"] == "random1" :
                child_config.attrib["file"] = models_path_objs
    tree_config.write('../../resources/scene_graspable_Dexnet_train.xml')
    
    matching_files = 0
    while matching_files < number_of_files :

        #generate scene
        print("matching_files " + str(matching_files))
        filelist = glob.glob(PATTERN)
        matching_files = len(filelist)/1
        os.system(r"./sceneGeneratorDexNet")

    #reproject images
    os.system(r"./reprojectImgs")

    #copy results
    os.system(r"mkdir shapeNetGraspable\_30instances_DexNet_Train")
    os.system(r"mkdir shapeNetGraspable\_30instances_DexNet_Train/" + str(category))
    os.system(r"pwd")
    #os.system(r"mv dataset/\* shapeNetGraspable_30instances_DexNet_Train/" + str(category))
    source_dir = 'dataset'
    target_dir = "shapeNetGraspable_30instances_DexNet_Train/" + str(category)

    file_names = os.listdir(source_dir)

    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
