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
robots = ["Anymal"]
# robot_configs = ["MessorII.xml", "Anymal.xml"]
robot_configs = ["Anymal.xml"]

#planners = ["RRTConnect", "RRTStarConnect", "InformedRRTStarConnect", "GuidedRRT", "GuidedRRTStarConnect", "InformedGuidedRRTStarConnect"]
#planner_configs = ["plannerRRTConnect.xml", "plannerRRTStarConnect.xml", "plannerRRTStarConnect.xml", "guidedRRT.xml", "guidedRRTStarConnect.xml", "guidedRRTStarConnect.xml"]
planners = ["InformedGuidedRRTStarConnect"]
planner_configs = ["guidedRRTStarConnect.xml"]

map_path = "maps/"
map_ext = ".dat"
# maps = ["flatBig", "mapaBig1", "map_box", "map_bug_trap"]
maps = ["map_bug_trap"]
init_positions = [(0,-2.3), (0,-2.3), (0,-2.3), (0,-2.3)]
goal_positions = [(0,2.3), (0,2.3), (0,2.3), (0,2.3)]
goal_yaws = [ 0.0, 0.0, 0.0, 0.0]

numTrials = 10
maxPlanningTime = 600 #seconds

tree_config = ET.parse('../../resources/configGlobal.xml')
root_config = tree_config.getroot()

tree_terrain = ET.parse('../../resources/terrain.xml')
root_terrain = tree_terrain.getroot()

results_dict = dict()

for robot_name, robot_config in zip(robots,robot_configs) :
    results_dict[robot_name] = dict()
    for planner_name, planner_config in zip(planners,planner_configs) :
        results_dict[robot_name][planner_name] = dict()
        for map_name, init_position, goal_position, goal_yaw in zip(maps,init_positions,goal_positions, goal_yaws) :
            results_dict[robot_name][planner_name][map_name] = dict()
            results_dict[robot_name][planner_name][map_name]["planning_time"] = list()
            results_dict[robot_name][planner_name][map_name]["path2d"] = list()
            results_dict[robot_name][planner_name][map_name]["path3d"] = list()
            results_dict[robot_name][planner_name][map_name]["success"] = list()

            trial_no = 0
            while trial_no<numTrials :
                print("Robot name: " + robot_name + ", planner: " + planner_name + ", map name: " + map_name)
                print("Init position: (" + str(init_position[0]) + "," + str(init_position[1]) + "), Goal position: (" + 
                str(goal_position[0]) + "," + str(goal_position[1]) + ")")
                print("Goal yaw:" + str(goal_yaw))
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

                #change initial position of the robot
                tree_robot = ET.parse("../../resources/" + robot_name + "/" + robot_config)
                root_robot = tree_robot.getroot()

                for child_robot in root_robot:
                    if (child_robot.tag == "robot_body") :
                        for child2_robot in child_robot:
                            if (child2_robot.tag == "collision") :
                                for child3_robot in child2_robot:
                                    if (child3_robot.tag == "origin") :
                                        init_pos_str = child3_robot.attrib["xyz"]
                                        # split x y z
                                        init_pos_list = init_pos_str.split()
                                        init_posf = list(map(float,init_pos_list))
                                        print(init_posf)
                                        # change initial position
                                        init_pos_str = str(init_position[0]) + " " + str(init_position[1]) + " " + str(init_posf[2])
                                        # update xml file
                                        child3_robot.attrib["xyz"] = init_pos_str
                tree_robot.write("../../resources/" + robot_name + "/" + robot_config)

                #change goal position of the robot
                # and get the config of the RRT*-Connect
                tree_planner = ET.parse("../../resources/" + robot_name + "/" + planner_config)
                root_planner = tree_planner.getroot()
                rrt_star_connect_config = ""
                for child_planner in root_planner:
                    if (child_planner.tag == "goalPose") :
                        child_planner.attrib["xy"] = str(goal_position[0]) + " " + str(goal_position[1])
                        child_planner.attrib["yaw"] = str(goal_yaw)    
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
                        if (child_planner.tag == "goalPose") :
                            child_planner.attrib["xy"] = str(goal_position[0]) + " " + str(goal_position[1])
                            child_planner.attrib["yaw"] = str(goal_yaw)
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
                process = Popen(shlex.split("./planPath"), stdout=PIPE, bufsize=1, close_fds=ON_POSIX)
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
                    results_dict[robot_name][planner_name][map_name]["success"].append(0)
                    results_dict[robot_name][planner_name][map_name]["planning_time"].append(np.nan)
                    results_dict[robot_name][planner_name][map_name]["path2d"].append(np.nan)
                    results_dict[robot_name][planner_name][map_name]["path3d"].append(np.nan)
                    trial_no += 1
                elif (exit_code == 1) :
                    p = re.compile(r'Path planning finished \(t = (.{10})ms\)')
                    result = p.search(str(output))
                    if result :
                        planning_time = float(result.groups(0)[0])
                    else :
                        print("Planning time pattern does not match")
                    
                    p = re.compile(r'Planned path length nodes only 2D: (.{10}) m')
                    result = p.search(str(output))
                    if result :
                        path_length2D = float(result.groups(0)[0])
                    else :
                        print("Path 2D pattern does not match")

                    p = re.compile(r'Planned path length 3D: (.{10}) m')
                    result = p.search(str(output))
                    if result :
                        path_length3D = float(result.groups(0)[0])
                    else :
                        print("Path 3D pattern does not match")

                    results_dict[robot_name][planner_name][map_name]["success"].append(1)
                    results_dict[robot_name][planner_name][map_name]["planning_time"].append(planning_time)
                    results_dict[robot_name][planner_name][map_name]["path2d"].append(path_length2D)
                    results_dict[robot_name][planner_name][map_name]["path3d"].append(path_length3D)
                    trial_no += 1
                elif (exit_code == 2) :
                    print("segfault")
                    # trial_no -= 1 #if segfault try again
                # print("pause")
                # print(trial_no)
                # print(len(results_dict[robot_name][planner_name][map_name]["success"]))
                # input("Press Enter to continue...")


for robot_name in robots :
    f = open("planning_results" + robot_name + ".tex", "w")

    for planner_name in planners :
        for map_name in maps :
            f.write("# " + robot_name + ", " + planner_name + ", " + map_name + "\n")
            f.write("#planning\_time [ms] = [")
            for t in results_dict[robot_name][planner_name][map_name]["planning_time"]:
                f.write(str(t) + ", ")
            f.write("]\\n")
            f.write("#path2d [m] = [")
            for d in results_dict[robot_name][planner_name][map_name]["path2d"]:
                f.write(str(d) + ", ")
            f.write("]\n")
            f.write("#path3d [m] = [")
            for d in results_dict[robot_name][planner_name][map_name]["path3d"]:
                f.write(str(d) + ", ")
            f.write("]\n")
            f.write("#success [] = [")
            for s in results_dict[robot_name][planner_name][map_name]["success"]:
                f.write(str(s) + ", ")
            f.write("]\n")

    f.write("\\begin{table*}[t]\n")
    f.write("\\caption{Average planning time $\overline{t}_p$ and obtained path length $\overline{p}$ on the " + robot_name + " robot. The results indicated by '*' are obtained for the single successful experiment}\n")
    f.write("\\newcommand{\mc}[3]{\multicolumn{#1}{#2}{#3}}\n")
    f.write("\\label{tab:results_" + robot_name + "}\n")
    f.write("\\begin{center}\n")
    f.write("\\setlength\\tabcolsep{1.0pt}\n")
    f.write("\\begin{tabular}{l|")
    for map_name in maps :
        f.write("ccccccc|")
    f.write("}\n")
    f.write(" ")
    for map_name in maps :
        f.write("& \\mc{6}{c}{"+map_name.replace("_","\_")+"} ")
    f.write("\\\\\n")

    f.write(" ")
    for map_name in maps :
        f.write("& s~[\\%] & $\overline{t}_p$~[s] & $\sigma_t$~[s] & $\overline{p_{\\rm 2D}}$~[m] & $\sigma_{p_{\\rm 2D}}$~[m] & $\overline{p_{\\rm 3D}}$~[m] & $\sigma_{p_{\\rm 3D}}$~[m]  ")
    f.write("\\\\\n")

    for planner_name in planners :
        f.write(repr(planner_name))
        for map_name in maps : 
            f.write( " & " + str(100*np.sum(results_dict[robot_name][planner_name][map_name]["success"])/numTrials)) # mean time
            mean_time = np.nanmean(results_dict[robot_name][planner_name][map_name]["planning_time"])
            f.write( " & " + f"{mean_time/1000:.1f}") # mean time
            std_time = np.nanstd(results_dict[robot_name][planner_name][map_name]["planning_time"])
            f.write( " & " + f"{std_time/1000:.1f}") # std time
            mean_2d = np.nanmean(results_dict[robot_name][planner_name][map_name]["path2d"])
            f.write( " & " + f"{mean_2d:.3f}") # mean path 2D
            std_2d = np.nanstd(results_dict[robot_name][planner_name][map_name]["path2d"])
            f.write( " & " + f"{std_2d:.3f}") # std path 2D
            mean_3d = np.nanmean(results_dict[robot_name][planner_name][map_name]["path3d"])
            f.write( " & " + f"{mean_3d:.3f}") # mean path 3D
            std_3d = np.nanstd(results_dict[robot_name][planner_name][map_name]["path3d"])
            f.write( " & " + f"{std_3d:.3f}") # std path 3D
        f.write("\\\\\n")

    f.write("\\end{tabular}\n")
    f.write("\\end{center}\n")
    f.write("\\end{table*}\n")
    f.close()
