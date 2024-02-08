#!/usr/bin/env python3

from signal import pause
import lxml.etree as ET
import os
import sys
import subprocess
import re
import shlex
from subprocess import Popen, PIPE
import datetime
import numpy as np
from threading  import Thread
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # python 2.x

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        print(line)
        queue.put(line)
    out.close()

print (os.getcwd())
os.chdir(os.path.dirname("../build/bin/"))
print (os.getcwd())

# robots = ["MessorII", "Anymal"]
robots = ["MessorII"]
# robot_configs = ["MessorII.xml", "Anymal.xml"]
robot_configs = ["MessorII.xml"]

#planners = ["RRTConnect", "RRTStarConnect", "InformedRRTStarConnect", "GuidedRRT", "GuidedRRTStarConnect", "InformedGuidedRRTStarConnect"]
#planner_configs = ["plannerRRTConnect.xml", "plannerRRTStarConnect.xml", "plannerRRTStarConnect.xml", "guidedRRT.xml", "guidedRRTStarConnect.xml", "guidedRRTStarConnect.xml"]
planners = ["InformedGuidedRRTStarConnect"]
planner_configs = ["guidedRRTStarConnect.xml"]

map_path = "maps/"
map_ext = ".dat"
#maps = ["flatBig", "mapaBig1", "map_box", "map_bug_trap"]
maps = ["map_bug_trap"]
init_positions = [(0,-2.3), (0,-2.3), (0,-2.3), (0,-2.3)]
goal_positions = [(0,2.3), (0,2.3), (0,2.3), (0,2.3)]
goal_yaws = [ 0.0, 0.0, 0.0, 0.0]

numTrials = 1
maxPlanningTime = 600 #seconds

tree_config = ET.parse('../../resources/configGlobal.xml')
root_config = tree_config.getroot()

tree_terrain = ET.parse('../../resources/terrain.xml')
root_terrain = tree_terrain.getroot()

results_dict = dict()

for robot_name, robot_config in zip(robots,robot_configs) :
    for planner_name, planner_config in zip(planners,planner_configs) :
        for map_name, init_position, goal_position, goal_yaw in zip(maps,init_positions,goal_positions, goal_yaws) :

            trial_no = 0
            while trial_no<numTrials :
                print("Robot name: " + robot_name + ", planner: " + planner_name + ", map name: " + map_name)
                #print("Init position: (" + str(init_position[0]) + "," + str(init_position[1]) + "), Goal position: (" + 
                #str(goal_position[0]) + "," + str(goal_position[1]) + ")")
                #print("Goal yaw:" + str(goal_yaw))
                for child_config in root_config:
                    # print(child.tag, child.attrib)

                    # change the robot
                    if (child_config.tag == "Robot") :
                        # for child2 in child:
                            # print(child2.tag, child2.attrib, child2.text)
                        for robot in child_config.iter('type'):
                            robot.text = robot_name
                        for config in child_config.iter('config'):
                            config.text = robot_name + "/" + robot_config

                    # change the planner
                    if (child_config.tag == "Planner") :
                        for planner in child_config.iter('type'):
                            planner.text = planner_name
                            if (planner_name == "InformedRRTStarConnect") :
                                planner.text = "RRTStarConnect"
                            if (planner_name == "InformedGuidedRRTStarConnect") :
                                planner.text = "GuidedRRTStarConnect"
                        for config in child_config.iter('config'):
                            config.text = robot_name + "/" + planner_config

                tree_config.write('../../resources/configGlobal.xml')

                for child_terrain in root_terrain:
                    # change the terrain
                    if (child_terrain.tag == "heightfield") :
                        if (robot_name == "Anymal" and map_name == "map_bug_trap") :
                            child_terrain.attrib["filename"] = map_path + "map_bug_trap_anymal" + map_ext
                        else :
                            child_terrain.attrib["filename"] = map_path + map_name + map_ext

                tree_terrain.write('../../resources/terrain.xml')
                print("Start planning...")
                
                #change parameters of the planner
                # and get the config of the RRT*-Connect
                tree_planner = ET.parse("../../resources/" + robot_name + "/" + planner_config)
                root_planner = tree_planner.getroot()
                rrt_star_connect_config = ""
                for child_planner in root_planner: 
                    # change between informed/standard version
                    if (child_planner.tag == "parameters") :
                        child_planner.attrib["maxPlanningTime"] = str(maxPlanningTime)
                        if (planner_name == "RRTStarConnect") :
                            child_planner.attrib["informed"] = "false"
                        if (planner_name == "InformedRRTStarConnect") :
                            child_planner.attrib["informed"] = "true"
                        if (planner_name == "GuidedRRTStarConnect") or (planner_name == "InformedGuidedRRTStarConnect") :
                            rrt_star_connect_config = child_planner.attrib["configRRTStarConnect"]
                tree_planner.write("../../resources/" + robot_name + "/" + planner_config)

                # change the informed standard version of the guided RRT* connect
                if (planner_name == "GuidedRRTStarConnect") or (planner_name == "InformedGuidedRRTStarConnect") :
                    tree_planner = ET.parse("../../resources/" + rrt_star_connect_config)
                    root_planner = tree_planner.getroot()
                    for child_planner in root_planner:
                        # change between informed/standard version
                        if (child_planner.tag == "parameters") :
                            if (planner_name == "GuidedRRTStarConnect") :
                                child_planner.attrib["informed"] = "false"
                            if (planner_name == "InformedGuidedRRTStarConnect") :
                                child_planner.attrib["informed"] = "true"

                now = datetime.datetime.now()
                print ("Current date and time : ")
                print (now.strftime("%Y-%m-%d %H:%M:%S"))

                # process = subprocess.Popen(shlex.split("./planPath"), stderr=sys.stderr, stdout=sys.stdout)
                # process = Popen(shlex.split("./planPath"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, encoding='utf-8', errors='replace')
                process = Popen(shlex.split("./genPlanPathsDataset"), stdout=PIPE, bufsize=1, close_fds=ON_POSIX)
                q = Queue()
                t = Thread(target=enqueue_output, args=(process.stdout, q))
                t.daemon = True # thread dies with the program
                t.start()

                exit_code = process.wait()
                print(exit_code)
                Popen.wait(process)
                # (output, err) = process.communicate()
                # print("err ")
                # print(list(q.queue))
                output = str(q.queue)
                print("output \n\n\n\n\n\n\n\n\n\n")
                print(output)
                print("\n\n\n\n\n\n\n\n\n\n")

                planning_time = np.nan
                path_length2D = np.nan
                path_length3D = np.nan
                if (exit_code==0):
                    print("Timeout")
                    # print(output)
                    #trial_no += 1
                elif (exit_code == 1) :
                    p = re.compile(r'Path planning finished \(t = (.{10})ms\)')
                    trial_no += 1
                elif (exit_code == 2) :
                    print("segfault")
                    trial_no -= 1 #if segfault try again
                # print("pause")
                # print(trial_no)
                # input("Press Enter to continue...")

