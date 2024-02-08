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

number_of_files = 2
PATTERN_IN = "dataset/*in.png"
PATTERN_OUT = "dataset/*out.png"

print (os.getcwd())
os.chdir(os.path.dirname("../build/bin/"))
print (os.getcwd())

preferred_curvature_pitchs = [-3.6, -2.4, -1.2, 0.0, 1.2, 2.4, 3.6]
preferred_curvature_rolls = [-3.6, -2.4, -1.2, 0.0, 1.2, 2.4, 3.6]
legs_no = [0, 1]

# todo: set scene graspable in configGlobal.xml

for preferred_pitch in preferred_curvature_pitchs :
    for preferred_roll in preferred_curvature_rolls :
        tree_config = ET.parse('../../resources/Anymal_C/Anymal_C.xml')
        root_robot = tree_config.getroot()
        
        #configure scene generator
        for child_robot in root_robot:
            if (child_robot.tag == "leg") :
                for child2_robot in child_robot:
                    if (child2_robot.tag == "footholdSelection") :
                        child2_robot.attrib["preferredCurvaturePitch"] = str(preferred_pitch)
                        child2_robot.attrib["preferredCurvatureRoll"] = str(preferred_roll)
        tree_config.write('../../resources/Anymal_C/Anymal_C.xml')

        #generate data
        os.system(r"./genFootTrainPisa")

        #copy results
        os.system(r"mkdir dataset\_Pisa")
        os.system(r"mkdir dataset\_Pisa/" + "pisa\_curv\_pitch\_" + str(preferred_pitch) + "\_roll\_" + str(preferred_roll))
        os.system(r"pwd")
        #os.system(r"mv dataset/\* shapeNetGraspable_30instances/" + str(category))
        source_dir = 'datasetPisaTmp'
        target_dir = "dataset_Pisa/" + "pisa_curv_pitch_" + str(preferred_pitch) + "_roll_" + str(preferred_roll)

        file_names = os.listdir(source_dir)

        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), target_dir)
