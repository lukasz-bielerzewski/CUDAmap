#include "Grabber/realSense2.h"
#include <memory>
#include <stdexcept>
#include <chrono>
#include <thread>

using namespace grabber;
using namespace std::chrono;

/// A single instance of Generic grabber
RealSense2Grabber::Ptr grabberRS2;

RealSense2Grabber::RealSense2Grabber(void) : Grabber("RealSense2 Grabber", TYPE_REALSENSE2) {
    startT = std::chrono::high_resolution_clock::now();
    initialize();
}

RealSense2Grabber::~RealSense2Grabber(void) {

}

void RealSense2Grabber::initialize(void) {
    sensorWrapper.reset(new RealSenseWrapper);
}

bool RealSense2Grabber::grab(void) {
    //image.convertTo(image, CV_8UC1, 255.0/1024.0);
    SensorFrame tmp;
    cv::Size frameSize = cv::Size(640, 480);
    tmp.image = cv::Mat::zeros(frameSize, CV_8UC3);
    tmp.depth = cv::Mat::zeros(frameSize, CV_16UC1);

    sensorWrapper->acquireFrames(tmp.image, tmp.depth);

    tmp.timestamp = (double)std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::high_resolution_clock::now() - startT).count();
    mtx.lock();
    sensorFrames.push(tmp);
    mtx.unlock();
    return true;
}

/// Grab sequence of image and save sequence to files (duration in seconds)
void RealSense2Grabber::getSequence(const int duration){
    high_resolution_clock::time_point start = high_resolution_clock::now();
    movie.clear();
    while (1){ //recording
        grab(); // grab frame
        movie.push_back(sensorFrame);
        if (duration_cast<std::chrono::duration<int> >(system_clock::now() - start).count()>duration){
            break;
        }
    }

    //save sequence
    /*int i = 1;
    for (std::vector<SensorFrame>::iterator it = movie.begin(); it!=movie.end(); it++){
        std::ostringstream oss;
        oss << filePrefix << i << ".jpg";
        cv::imwrite( oss.str(), it->image );
        i++;
    }
    exportTimestamps(filePrefix);
    std::cout << "sequence saved: " << movie.size() << " frames\n";*/
}

/// Returns current frame
const SensorFrame& RealSense2Grabber::getSensorFrame(void) {
    bool waitForImage = true;
    while (waitForImage){
        mtx.lock();
        if (sensorFrames.size())
            waitForImage = false;
        mtx.unlock();
    }
    mtx.lock();
    sensorFrame = sensorFrames.front();
    sensorFrames.pop();
    mtx.unlock();
    //sensorFrame.cloud = model.depth2cloud(sensorFrame.depth);
    return sensorFrame;
}

/// run grabber thread
void RealSense2Grabber::calibrate(void) {

}

grabber::Grabber* grabber::createGrabberRealSense2(void) {
    grabberRS2.reset(new RealSense2Grabber());
    return grabberRS2.get();
}

grabber::Grabber* grabber::createGrabberRealSense2(std::string configFile) {
    grabberRS2.reset(new RealSense2Grabber(configFile));
    return grabberRS2.get();
}
