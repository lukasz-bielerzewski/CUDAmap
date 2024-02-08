#include "Grabber/realSense.h"
#include <memory>
#include <stdexcept>
#include <chrono>
#include <thread>

using namespace grabber;
using namespace std::chrono;

/// A single instance of Generic grabber
RealSenseGrabber::Ptr grabberRS;

RealSenseGrabber::RealSenseGrabber(void) : Grabber("RealSense Grabber", TYPE_REALSENSE) {
    startT = std::chrono::high_resolution_clock::now();
    initialize();
}

RealSenseGrabber::~RealSenseGrabber(void) {

}

void RealSenseGrabber::initialize(void) {
    // Access the first available RealSense device
    rsDev = rsCtx.get_device(0);

    // Configure Infrared stream to run at VGA resolution at 30 frames per second
    rsDev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);
    rsDev->enable_stream(rs::stream::color, 640, 480, rs::format::bgr8, 30);
    // Start streaming
    rsDev->start();

    rsDev->set_option(rs::option::r200_lr_auto_exposure_enabled, 1);
}

bool RealSenseGrabber::grab(void) {
    //image.convertTo(image, CV_8UC1, 255.0/1024.0);
    SensorFrame tmp;
    rsDev->wait_for_frames();
    tmp.image = cv::Mat(cv::Size(640, 480), CV_8UC3, (void*)rsDev->get_frame_data(rs::stream::color), cv::Mat::AUTO_STEP);
    tmp.depth = cv::Mat(cv::Size(640, 480), CV_16UC1, (void*)rsDev->get_frame_data(rs::stream::depth), cv::Mat::AUTO_STEP);
    //tmp.image = image.clone();
    tmp.timestamp = (double)std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::high_resolution_clock::now() - startT).count();
    mtx.lock();
    sensorFrames.push(tmp);
    mtx.unlock();
    return true;
}

/// Grab sequence of image and save sequence to files (duration in seconds)
void RealSenseGrabber::getSequence(const int duration){
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
const SensorFrame& RealSenseGrabber::getSensorFrame(void) {
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
void RealSenseGrabber::calibrate(void) {

}

grabber::Grabber* grabber::createGrabberRealSense(void) {
    grabberRS.reset(new RealSenseGrabber());
    return grabberRS.get();
}

grabber::Grabber* grabber::createGrabberRealSense(std::string configFile) {
    grabberRS.reset(new RealSenseGrabber(configFile));
    return grabberRS.get();
}
