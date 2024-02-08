/** @file recorder.cpp
*
* @author Dominik Belter
*/

#include "Utilities/recorder.h"
#include <iostream>
#include <fstream>
#include <iomanip>

/// plot graph
void Recorder1D::savePlot(void){
    std::ofstream mfile;
    mfile.open (name + ".m");
    mfile << name << "Value = [";
    for (const auto & element : container)
        mfile << element.second << ", ";
    mfile << "];\n";
    mfile << name << "Time = [";
    for (const auto & element : container)
        mfile << element.first << ", ";
    mfile << "];\n";
    mfile << "plot(" << name << "Time, " << name << "Value, " << plotAtr << ");";
    mfile.close();
}

/// plot graph
void Recorder6D::savePlot(void){
    std::ofstream mfile;
    mfile.open (name);
    mfile << name << "X = [";
    for (const auto & element : container)
        mfile << element.second(0,3) << ", ";
    mfile << "];\n";
    mfile << name << "Y = [";
    for (const auto & element : container)
        mfile << element.second(1,3) << ", ";
    mfile << "];\n";
    mfile << name << "Z = [";
    for (const auto & element : container)
        mfile << element.second(2,3) << ", ";
    mfile << "];\n";

    mfile << name << "Roll = [";
    for (const auto & element : container){
        walkers::Vec3 rpy = walkers::fromRotationMat(element.second);
        mfile << rpy.x() << ", ";
    }
    mfile << "];\n";
    mfile << name << "Pitch = [";
    for (const auto & element : container){
        walkers::Vec3 rpy = walkers::fromRotationMat(element.second);
        mfile << rpy.y() << ", ";
    }
    mfile << "];\n";
    mfile << name << "Yaw = [";
    for (const auto & element : container){
        walkers::Vec3 rpy = walkers::fromRotationMat(element.second);
        mfile << rpy.z() << ", ";
    }
    mfile << "];\n";

    mfile << name << "Time = [";
    for (const auto & element : container)
        mfile << element.first << ", ";
    mfile << "];\n";
    mfile << "plot3(" << name << "X, " << name << "Y, " << name << "Z, " << plotAtr << ");";
    mfile.close();
}

/// plot graph
void RecorderFoot::savePlot(void){
    std::ofstream mfile;
    mfile.open (name + ".m");
    mfile.close();
}

/// plot graph
void RecorderRobotState::savePlot(void){
    std::ofstream mfile;
    mfile.open (name + ".m");
    mfile.close();
}

/// loadMocapData
void Recorder6D::loadMocapData(const std::string& filename, const walkers::Mat34& offset){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        while (std::getline(myfile,line)){
            std::istringstream iss(line);
            if (line.size()>0&&line.substr(0, 1)!="#"){
                double timestamp, x, y, z, qx, qy, qz ,qw;
                if (!(iss >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw)) { break; } // error
                else{
                    walkers::Quaternion quat(qw,qx,qy,qz);
                    walkers::Mat34 pose;
                    pose.matrix().block<3,3>(0,0) = quat.matrix();
                    pose(0,3)=x; pose(1,3)=y; pose(2,3)=z;
                    store(timestamp, offset*pose);
                }
            }
        }
        myfile.close();
    }
}

/// loadMocapData
void Recorder6D::loadTimestamps(const std::string& filename, const walkers::Mat34& offset){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        while (std::getline(myfile,line)){
            std::istringstream iss(line);
            if (line.size()>0&&line.substr(0, 1)!="#"){
                double timestamp;
                if (!(iss >> timestamp)) { break; } // error
                else{
                    store(timestamp, offset);
                }
            }
        }
        myfile.close();
    }
}

/// synchronizeWithCamera
void Recorder6D::synchronizeWithCamera(const std::string& matchedFilename){
    std::string line;
    std::ifstream myfile(matchedFilename);
    std::vector<std::pair<double,walkers::Mat34>> cameraPoses;
    if (myfile.is_open()){
//        int minPoseNo=0;
        while (std::getline(myfile,line)){
            std::istringstream iss(line);
            if (line.size()>0&&line.substr(0, 1)!="#"){
                double firstTime, secondTime;
                if (!(iss >> firstTime >> secondTime)) { break; } // error
                else{
                    double errorMin = std::numeric_limits<double>::max();
                    double curTime; walkers::Mat34 curPose;
                    for (const auto& val : container){
                        double err = fabs(secondTime-val.first);
                        if (err<errorMin){
                            errorMin = err;
                            curTime = val.first;
                            curPose = val.second;
                        }
                    }
                    cameraPoses.push_back(std::make_pair(curTime,curPose));
                }
            }
        }
        myfile.close();
    }
    container.clear();
    for (const auto& pose : cameraPoses){
        store(pose.first, pose.second);
    }
}

/// load orders Data
void Recorder6D::loadOrdersData(const std::string& filename, const walkers::Mat34& offset){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        while (std::getline(myfile,line)){
            std::istringstream iss(line);
            if (line.size()>0&&line.substr(0, 1)!="#"){
                double timestamp, x, y, z, rotx, roty, rotz, speed;
                std::string orderType;
                if (!(iss >> orderType >> timestamp >> x >> y >> z >> rotx >> roty >> rotz >> speed)) { break; } // error
                else{
                    walkers::Vec3 rot(rotx, roty, rotz);
                    walkers::Mat34 pose;
                    pose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rot);
                    pose(0,3)=x; pose(1,3)=y; pose(2,3)=z;
                    store(timestamp, offset*pose);
                }
            }
        }
        myfile.close();
    }
}

/// load SLAM Data
void Recorder6D::loadSLAMdata(const std::string& filename, const walkers::Mat34& offset){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        while (std::getline(myfile,line)){
            std::istringstream iss(line);
            if (line.size()>0&&line.substr(0, 1)!="#"){
                double timestamp, x, y, z, qx, qy, qz, qw;
                if (!(iss >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw)) { break; } // error
                else{
                    walkers::Quaternion rot(qw, qx, qy, qz);
                    walkers::Mat34 pose;
                    pose.matrix().block<3,3>(0,0) = rot.matrix();
                    pose(0,3)=x; pose(1,3)=y; pose(2,3)=z;
                    store(timestamp, offset*pose);
                }
            }
        }
        myfile.close();
    }
}

/// loadMocapData
void Recorder6D::loadAHRSData(const std::string& filename, const walkers::Mat34& offset){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        while (std::getline(myfile,line)){
            if (line.size()>0&&line.substr(0, 1)!="#"){
                std::stringstream ss(line);
                std::vector<double> dataIn;
                double in;
                while (ss >> in){
                    dataIn.push_back(in);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
                double timestamp(dataIn[1]), qx(dataIn[12]), qy(dataIn[13]), qz(dataIn[14]) ,qw(dataIn[11]);
//                double accelx(dataIn[2]), accely(dataIn[3]), accelz(dataIn[4]), velx(dataIn[5]), vely(dataIn[6]), velz(dataIn[7]), magx(dataIn[8]), magy(dataIn[9]), magz(dataIn[10]);
//                int iterNo(int(dataIn[0]));
                walkers::Quaternion quat(qw,qx,qy,qz);
                walkers::Mat34 pose(walkers::Mat34::Identity());
                pose.matrix().block<3,3>(0,0) = quat.matrix();
                store(timestamp, offset*pose);
            }
        }
        myfile.close();
    }
}

/// load reference motion
void Recorder6D::loadRefMotion(const std::string& filename, const walkers::Mat34& offset){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        int prevOrder=-1;
        walkers::Mat34 prevMotion(walkers::Mat34::Identity());
        walkers::Mat34 prevPlatformMotion(walkers::Mat34::Identity());
        walkers::Mat34 globalPose(walkers::Mat34::Identity());
        int lineNo=0;
        while (std::getline(myfile,line)){
            if (line.size()>0&&line.substr(0, 1)!="#"){
                std::stringstream ss(line);
                std::vector<double> dataIn;
                double in;
                while (ss >> in){
                    dataIn.push_back(in);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
                int orderId((int)dataIn[0]);
                double timestamp(dataIn[1]);
                double x(0),y(0),z(0),qx(0),qy(0),qz(0),qw(1);
                if (orderId==1||orderId==3){
                    x=dataIn[3]; y=dataIn[4]; z=dataIn[5]; qx=dataIn[6]; qy=dataIn[7]; qz=dataIn[8]; qw=dataIn[9];
                }
                else if (orderId==2){
                    int idX(0);
                    if (dataIn[3]==2&&dataIn[11]==2){
                        (dataIn[6]<dataIn[14]) ? idX=1 : idX=0;
                        prevPlatformMotion=walkers::Mat34::Identity();
                    }
                    else if (dataIn[3]==2)
                        idX=0;
                    else if (dataIn[11]==2)
                        idX=1;
                    x=dataIn[4+idX*8]; y=dataIn[5+idX*8]; z=dataIn[6+idX*8]; qx=dataIn[7+idX*8]; qy=dataIn[8+idX*8]; qz=dataIn[9+idX*8]; qw=dataIn[10+idX*8];
                }
                walkers::Quaternion quat(qw,qx,qy,qz);
                walkers::Mat34 pose(walkers::Mat34::Identity());
                pose.matrix().block<3,3>(0,0) = quat.matrix();
                pose(0,3) = x; pose(1,3) = y; pose(2,3) = z;
                if (prevOrder==-1)
                    store(0, prevMotion);
                if (orderId==0&&prevOrder==3)
                    store(timestamp, prevMotion);
                if (orderId==0&&prevOrder==1){
                    globalPose = globalPose*prevMotion;
                    prevPlatformMotion = prevMotion;
                    store(timestamp, globalPose);
                }
                if (orderId==0&&prevOrder==2){
                    walkers::Mat34 motion(prevMotion);
                    motion(0,3) = prevMotion(0,3) - prevPlatformMotion(0,3);
                    motion(1,3) = prevMotion(1,3) - prevPlatformMotion(1,3);
                    motion(2,3) = prevMotion(2,3) - prevPlatformMotion(2,3);
                    walkers::Vec3 prevPlatformRot = walkers::fromRotationMat(prevPlatformMotion);
                    walkers::Vec3 prevRot = walkers::fromRotationMat(prevMotion);
                    walkers::Vec3 rot(prevRot.x()-prevPlatformRot.x(),prevRot.y()-prevPlatformRot.y(), prevRot.z()-prevPlatformRot.z());
                    motion.matrix().block<3,3>(0,0) = walkers::toRotationMat(rot);
                    globalPose = globalPose*motion;
                    prevPlatformMotion = prevMotion;
                    store(timestamp, globalPose);
                }
                prevMotion = offset*pose;
                prevOrder = orderId;
                lineNo++;
            }
        }
        myfile.close();
    }
}

/// save GT traj
void Recorder6D::saveGTtraj(std::string filename, walkers::Mat34 offset){
    std::ofstream file;
    file.open (filename);
    size_t poseNo=0;
    for (auto& pose : container){
        walkers::Mat34 pp = pose.second *offset;
        walkers::Quaternion quat(pose.second.rotation());
//        file << poseNo << " " << pp(0,3) << " " << pp(1,3) << " " << pp(2,3) << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "\n";
//        file << pose.first/1e9 << " " << pp(0,3) << " " << pp(1,3) << " " << pp(2,3) << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "\n";
        file << std::fixed << std::setprecision(6) << std::setfill('0') << 0.05*double(poseNo) << " " << pp(0,3) << " " << pp(1,3) << " " << pp(2,3) << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << "\n";
        poseNo++;
    }
    file.close();
}

/// load reference motion
void Recorder6D::loadRefMotion(const std::string& filename, const walkers::Mat34& offset, const Recorder6D& ahrsData, const walkers::Vec3& offsetIMU){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        int prevOrder=-1;
        walkers::Mat34 prevMotion(walkers::Mat34::Identity());
        walkers::Mat34 prevAHRSMotion(walkers::Mat34::Identity());
        walkers::Mat34 prevPlatformMotion(walkers::Mat34::Identity());
        walkers::Mat34 globalPose(walkers::Mat34::Identity());
        int lineNo=0;
        auto ahrsIt = ahrsData.container.begin();
        walkers::Vec3 prevRPY(walkers::fromRotationMat(ahrsIt->second));
        while (std::getline(myfile,line)){
            if (line.size()>0&&line.substr(0, 1)!="#"){
                std::stringstream ss(line);
                std::vector<double> dataIn;
                double in;
                while (ss >> in){
                    dataIn.push_back(in);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
                int orderId((int)dataIn[0]);
                double timestamp(dataIn[1]);
                while (ahrsIt->first<timestamp&&ahrsIt!=ahrsData.container.end())//find actual ahrs data
                    ahrsIt++;
                double x(0),y(0),z(0),qx(0),qy(0),qz(0),qw(1);
                if (orderId==1||orderId==3){
                    x=dataIn[3]; y=dataIn[4]; z=dataIn[5]; qx=dataIn[6]; qy=dataIn[7]; qz=dataIn[8]; qw=dataIn[9];
                    ///correction from ahrs
                    walkers::Vec3 rpy = walkers::fromRotationMat(ahrsIt->second);
                    walkers::Vec3 rpyOrders = walkers::fromRotationMat(globalPose);
                    prevRPY=rpy;
                    rpy.z()=rpyOrders.z();
                    rpy.x()-=offsetIMU.x(); rpy.y()-=offsetIMU.y();
                    globalPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rpy);
                }
                else if (orderId==2){
                    int idX(0);
                    if (dataIn[3]==2&&dataIn[11]==2){
                        (dataIn[6]<dataIn[14]) ? idX=1 : idX=0;
                        prevPlatformMotion=walkers::Mat34::Identity();
                    }
                    else if (dataIn[3]==2)
                        idX=0;
                    else if (dataIn[11]==2)
                        idX=1;
                    x=dataIn[4+idX*8]; y=dataIn[5+idX*8]; z=dataIn[6+idX*8]; qx=dataIn[7+idX*8]; qy=dataIn[8+idX*8]; qz=dataIn[9+idX*8]; qw=dataIn[10+idX*8];
                    ///correction from ahrs
                    walkers::Vec3 rpy = walkers::fromRotationMat(ahrsIt->second);
                    walkers::Vec3 rpyOrders = walkers::fromRotationMat(globalPose);
                    prevRPY=rpy;
                    rpy.z()=rpyOrders.z();
                    rpy.x()-=offsetIMU.x(); rpy.y()-=offsetIMU.y();
                    globalPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rpy);
                }
                walkers::Quaternion quat(qw,qx,qy,qz);
                walkers::Mat34 pose(walkers::Mat34::Identity());
                pose.matrix().block<3,3>(0,0) = quat.matrix();
                pose(0,3) = x; pose(1,3) = y; pose(2,3) = z;
                if (prevOrder==-1)
                    store(0, prevMotion);
                if (orderId==0&&prevOrder==3)
                    store(timestamp, prevMotion);
                if (orderId==0&&prevOrder==1){
//                    walkers::Mat34 ahrsPose = ahrsIt->second;
//                    walkers::Vec3 rpy = walkers::fromRotationMat(ahrsPose);
//                    rpy.z()=0;
//                    ahrsPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rpy);

//                    walkers::Mat34 motion(ahrsPose);
//                    motion(0,3) = ahrsPose(0,3) - prevAHRSMotion(0,3);
//                    motion(1,3) = ahrsPose(1,3) - prevAHRSMotion(1,3);
//                    motion(2,3) = ahrsPose(2,3) - prevAHRSMotion(2,3);
//                    walkers::Vec3 prevPlatformRot = walkers::fromRotationMat(prevAHRSMotion);
                    walkers::Vec3 rot = walkers::fromRotationMat(prevMotion);
//                    walkers::Vec3 rot(prevRot.x()-prevPlatformRot.x(),prevRot.y()-prevPlatformRot.y(), prevRot.z()-prevPlatformRot.z());
//                    motion.matrix().block<3,3>(0,0) = walkers::toRotationMat(rot);

                    ///correction from ahrs
                    walkers::Vec3 rpy = walkers::fromRotationMat(ahrsIt->second);
                    rpy.vector() = rpy.vector() - prevRPY.vector();
                    if (fabs(rpy.z())>5){
                        if (rpy.z()>0)
                            rpy.z()-=2*M_PI;
                        else if (rpy.z()<0)
                            rpy.z()+=2*M_PI;
                    }
                    walkers::Vec3 meanRot; meanRot.vector() = 0.35*rpy.vector()+0.65*rot.vector();
                    prevMotion.matrix().block<3,3>(0,0) = walkers::toRotationMat(meanRot);
//                    walkers::Vec3 rpyOrders = walkers::fromRotationMat(globalPose);
//                    rpy.z()=rpyOrders.z();
//                    globalPose.matrix().block<3,3>(0,0) = walkers::toRotationMat(rpy);

                    globalPose = globalPose*prevMotion;
                    prevPlatformMotion = prevMotion;
                    prevAHRSMotion = ahrsIt->second;
                    store(timestamp, globalPose);
                }
                if (orderId==0&&prevOrder==2){
                    walkers::Mat34 motion(prevMotion);
                    motion(0,3) = prevMotion(0,3) - prevPlatformMotion(0,3);
                    motion(1,3) = prevMotion(1,3) - prevPlatformMotion(1,3);
                    motion(2,3) = prevMotion(2,3) - prevPlatformMotion(2,3);
                    walkers::Vec3 prevPlatformRot = walkers::fromRotationMat(prevPlatformMotion);
                    walkers::Vec3 prevRot = walkers::fromRotationMat(prevMotion);
                    walkers::Vec3 rot(prevRot.x()-prevPlatformRot.x(),prevRot.y()-prevPlatformRot.y(), prevRot.z()-prevPlatformRot.z());
                    motion.matrix().block<3,3>(0,0) = walkers::toRotationMat(rot);
                    ///correction from ahrs
                    walkers::Vec3 rpy = walkers::fromRotationMat(ahrsIt->second);
                    rpy.vector() = rpy.vector() - prevRPY.vector();
                    if (fabs(rpy.z())>5){
                        if (rpy.z()>0)
                            rpy.z()-=2*M_PI;
                        else if (rpy.z()<0)
                            rpy.z()+=2*M_PI;
                    }
                    walkers::Vec3 meanRot; meanRot.vector() = 0.35*rpy.vector()+0.65*rot.vector();
                    motion.matrix().block<3,3>(0,0) = walkers::toRotationMat(meanRot);

                    globalPose = globalPose*motion;

                    prevPlatformMotion = prevMotion;
                    store(timestamp, globalPose);
                }
                prevAHRSMotion = ahrsIt->second;
                prevMotion = offset*pose;
                prevOrder = orderId;
                lineNo++;
            }
        }
        myfile.close();
    }
}

/// loadMocapData
void RecorderRobotState::loadMessorRobotState(const std::string& filename){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        while (std::getline(myfile,line)){
            if (line.size()>0&&line.substr(0, 1)!="#"){
                std::stringstream ss(line);
                std::vector<double> dataIn;
                double in;
                while (ss >> in){
                    dataIn.push_back(in);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
//                int iterNo((int)dataIn[0]);
                double timestamp(dataIn[1]);
                walkers::RobotState state(18,6);
                for (size_t jointNo=0; jointNo<state.jointsNo;jointNo++){
                    state.refValues[jointNo] = dataIn[2+jointNo];
                    state.currentValues[jointNo] = dataIn[20+jointNo];
                    state.refSpeed[jointNo] = dataIn[38+jointNo];
                }
                for (size_t legNo=0;legNo<state.legsNo;legNo++)
                    state.contactSensors[legNo] = dataIn[56+legNo];
                store(timestamp, state);
            }
        }
        myfile.close();
        currentElement = container.begin();
    }
}

/// load planned path
void RecorderRobotState3D::loadRobotState(const std::string& filename){
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        size_t iterNo=0;
        while (std::getline(myfile,line)){
            if (line.size()>0&&line.substr(0, 1)!="#"){
                std::stringstream ss(line);
                std::vector<double> dataIn;
                double in;
                while (ss >> in){
                    dataIn.push_back(in);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
                planner::RobotState3D state;
                state.feet.resize(dataIn.size()-7/8);
                state.robotPose.setPosition(Eigen::Vector3d(dataIn[0],dataIn[1],dataIn[2]));
                state.robotPose.setRotation(walkers::Quaternion(dataIn[3], dataIn[4], dataIn[5], dataIn[6]));
                for (size_t legNo=0; legNo<state.feet.size();legNo++){
                    planner::Foot foot;
                    walkers::Mat34 footPose;
                    int isFoothold((int)dataIn[7+legNo*8]);
                    if (isFoothold)
                        foot.isFoothold = true;
                    else
                        foot.isFoothold = false;
                    footPose(0,3) = dataIn[8+legNo*8]; footPose(1,3) = dataIn[9+legNo*8]; footPose(2,3) = dataIn[10+legNo*8];
                    walkers::Quaternion quat;
                    quat.w() = dataIn[11+legNo*8]; quat.x() = dataIn[12+legNo*8]; quat.y() = dataIn[13+legNo*8]; quat.z() = dataIn[14+legNo*8];
                    footPose.matrix().block<3,3>(0,0) = quat.matrix();
                    foot.pose = footPose;
                    state.feet[legNo] = foot;
                }
                store((double)iterNo, state);
                iterNo++;
            }
        }
        myfile.close();
        currentElement = container.begin();
    }
}

/// get next element
bool RecorderRobotState3D::getNextElement(std::pair<double,planner::RobotState3D>& element){
    if (container.size()==0)
        return false;
    if (currentElement!=container.end()){
        element = *currentElement;
        currentElement++;
        return true;
    }
    else{
        currentElement = container.begin();
        return false;
    }
}

/// plot graph
void RecorderRobotState3D::savePlot(void){
    std::ofstream mfile;
    mfile.open (name + ".m");
    mfile.close();
}

/// loadOldPlannerData
void Recorder6D::loadOldPlannerData(const std::string& filename){
    std::vector<std::vector<double>> pos; pos.resize(3);
    std::vector<std::vector<double>> rot; rot.resize(3);
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        for (size_t coord = 0; coord<pos.size();coord++){
            std::getline(myfile,line);//x
            if (line.size()>0&&line.substr(0, 1)!="#"){
                std::stringstream ss(line);
                double in;
                while (ss >> in){
                    pos[coord].push_back(in);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }
        }
        for (size_t coord = 0; coord<rot.size();coord++){
            std::getline(myfile,line);//x
            if (line.size()>0&&line.substr(0, 1)!="#"){
                std::stringstream ss(line);
                double in;
                while (ss >> in){
                    rot[coord].push_back(in);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }
        }
        myfile.close();
    }
    for (size_t poseNo=0;poseNo<pos[0].size();poseNo++){
        walkers::Mat34 motion; walkers::Vec3 rotation;
        motion(0,3)=pos[0][poseNo]; motion(1,3)=pos[1][poseNo]; motion(2,3)=pos[2][poseNo];
        rotation.x()=rot[0][poseNo]; rotation.y()=rot[1][poseNo]; rotation.z()=rot[2][poseNo];
        motion.matrix().block<3,3>(0,0) = walkers::toRotationMat(rotation);
        store(0, motion);
    }
}

/// loadOldPlannerData
void RecorderFoot::loadOldPlannerData(const std::string& filename){
    std::vector<std::vector<double>> pos; pos.resize(3);
    std::vector<int> isFoothold;
    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open()){
        for (size_t coord = 0; coord<pos.size();coord++){
            std::getline(myfile,line);//x
            if (line.size()>0&&line.substr(0, 1)!="#"){
                std::stringstream ss(line);
                double in;
                while (ss >> in){
                    pos[coord].push_back(in);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }
        }
        std::getline(myfile,line);//isFoothold
        if (line.size()>0&&line.substr(0, 1)!="#"){
            std::stringstream ss(line);
            double in;
            while (ss >> in){
                isFoothold.push_back((int)in);
                if (ss.peek() == ',')
                    ss.ignore();
            }
        }
        myfile.close();
    }
    bool prevFoothold = true;
    for (size_t poseNo=0;poseNo<pos[0].size();poseNo++){
        planner::Foot foot;
        foot.pose(0,3)=pos[0][poseNo]; foot.pose(1,3)=pos[1][poseNo]; foot.pose(2,3)=pos[2][poseNo];
        (isFoothold[poseNo]) ? foot.isFoothold=true : foot.isFoothold=false;
        if (foot.isFoothold&&prevFoothold)
            foot.pose(2,3)-=0.04;
        store(0, foot);
    }
}

/// get next element
bool Recorder6D::getNextElement(std::pair<double,walkers::Mat34>& element){
    if (container.size()==0)
        return false;
    if (currentElement!=container.end()){
        element = *currentElement;
//        element = *container.begin();
        currentElement++;
        return true;
    }
    else{
        currentElement = container.begin();
        return false;
    }
    //return container.getNextElement(element);
}

/// get next element
bool RecorderRobotState::getNextElement(std::pair<double,walkers::RobotState>& element){
    if (container.size()==0)
        return false;
    if (currentElement!=container.end()){
        element = *currentElement;
        currentElement++;
        return true;
    }
    else{
        currentElement = container.begin();
        return false;
    }
}
