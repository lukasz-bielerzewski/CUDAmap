/**
 * Project Walkers
 * @author Dominik Belter
 */

#include "Defs/defs.h"
#include "Defs/planner_defs.h"

#ifndef _RECORDER_H
#define _RECORDER_H

template <class T> class Recorder{
public:
    Recorder(std::string _plotAtr, int _delay, std::string _name) : plotAtr(_plotAtr), name(_name),
        delay(_delay), currDelay(0){
        maxRecordTime = std::numeric_limits<double>::max();
        isRecording = true;
    }
    /// set max recording time
    void setMaxRecordTime(double _maxRecordTime);
    /// save plot
    virtual void savePlot(void) = 0;
    /// store element
    void store(double time, T value);

    /// max recording time
    double maxRecordTime;
    /// plot atributes
    std::string plotAtr;
    /// is recording
    bool isRecording;

    /// time + value (1D+2D)
    std::list<std::pair<double,T>> container;
    /// name
    std::string name;
private:
  /// store every delay-th element
  int delay;
  /// current delay
  int currDelay;
};

/// set max recording time
template <typename T>
void Recorder<T>::setMaxRecordTime(double _maxRecordTime){
    maxRecordTime = _maxRecordTime;
}

template <typename T>
void Recorder<T>::store (double time, T value){
    if (isRecording&&time<maxRecordTime){
        if (currDelay-1==delay){
            container.push_back(std::make_pair(time,value));
            currDelay = 0;
        }
        else
            currDelay++;
    }
    else if (isRecording&&time>=maxRecordTime){
        isRecording = false;
        savePlot();
    }
}

class Recorder1D : public Recorder<double>{
public:
    Recorder1D(int _delay, std::string _name, std::string _plotAtr) : Recorder<double>(_plotAtr, _delay, _name){
    }
    /// plot graph
    void savePlot(void);
private:
};

class Recorder6D : public Recorder<walkers::Mat34>{
public:
    Recorder6D(int _delay, std::string _name, std::string _plotAtr) : Recorder<walkers::Mat34>(_plotAtr, _delay, _name){
    }
    /// loadMocapData
    void loadMocapData(const std::string& filename, const walkers::Mat34& offset);
    /// loadMocapData - timestamps only
    void loadTimestamps(const std::string& filename, const walkers::Mat34& offset);
    /// loadOrdersData
    void loadOrdersData(const std::string& filename, const walkers::Mat34& offset);
    /// loadAHRSData
    void loadAHRSData(const std::string& filename, const walkers::Mat34& offset);
    /// loadOldPlannerData
    void loadOldPlannerData(const std::string& filename);
    /// loadMocapData
    void loadRefMotion(const std::string& filename, const walkers::Mat34& offset);
    /// loadSLAMData
    void loadSLAMdata(const std::string& filename, const walkers::Mat34& offset);
    /// get next element
    bool getNextElement(std::pair<double,walkers::Mat34>& element);
    /// load reference motion
    void loadRefMotion(const std::string& filename, const walkers::Mat34& offset, const Recorder6D& ahrsData, const walkers::Vec3& offsetIMU);
    /// synchronizeWithCamera
    void synchronizeWithCamera(const std::string& matchedFilename);
    /// plot graph
    void savePlot(void);
    /// save GT traj
    void saveGTtraj(std::string filename, walkers::Mat34 offset);
private:
    /// currentElement
    typename std::list<std::pair<double,walkers::Mat34>>::iterator currentElement;
};

class RecorderFoot : public Recorder<planner::Foot>{
public:
    RecorderFoot(int _delay, std::string _name, std::string _plotAtr) : Recorder<planner::Foot>(_plotAtr, _delay, _name){
    }
    /// loadOldPlannerData
    void loadOldPlannerData(const std::string& filename);
    /// plot graph
    void savePlot(void);
private:
};

class RecorderRobotState : public Recorder<walkers::RobotState>{
public:
    RecorderRobotState(int _delay, std::string _name, std::string _plotAtr) : Recorder<walkers::RobotState>(_plotAtr, _delay, _name){
    }
    /// load robot state
    void loadMessorRobotState(const std::string& filename);
    /// get next element
    bool getNextElement(std::pair<double,walkers::RobotState>& element);
    /// plot graph
    void savePlot(void);
private:
    /// currentElement
    typename std::list<std::pair<double,walkers::RobotState>>::iterator currentElement;
};

class RecorderRobotState3D : public Recorder<planner::RobotState3D>{
public:
    RecorderRobotState3D(int _delay, std::string _name, std::string _plotAtr) : Recorder<planner::RobotState3D>(_plotAtr, _delay, _name){
    }
    /// load robot state
    void loadRobotState(const std::string& filename);
    /// get next element
    bool getNextElement(std::pair<double,planner::RobotState3D>& element);
    /// plot graph
    void savePlot(void);
private:
    /// currentElement
    typename std::list<std::pair<double,planner::RobotState3D>>::iterator currentElement;
};
#endif //_RECORDER_H
