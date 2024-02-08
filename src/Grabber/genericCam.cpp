#include "../include/Grabber/genericCam.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <thread>

using namespace grabber;
using namespace std::chrono;

/// A single instance of Generic grabber
GenericGrabber::Ptr grabberGen;

GenericGrabber::GenericGrabber(void) : grabber::Grabber("Generic Grabber", TYPE_RGB), capture(0) {
    startT = std::chrono::high_resolution_clock::now();
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FPS, 30);
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FOCUS, 230);
}

GenericGrabber::~GenericGrabber(void) {
}

bool GenericGrabber::grab(void) {
    if (!capture.isOpened()) { //check if video device has been initialised
        std::cout << "cannot open camera\n";
        return false;
    }
    SensorFrame tmp;
    capture.read(tmp.image);
    tmp.timestamp = (double)std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::high_resolution_clock::now() - startT).count();
    mtx.lock();
    sensorFrames.push(tmp);
    mtx.unlock();
    return true;
}

/// Grab sequence of image and save sequence to files (duration in seconds)
void GenericGrabber::getSequence(const int duration){
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

/// run grabber thread
void GenericGrabber::calibrate(void) {

}

grabber::Grabber* grabber::createGrabberGenericCam(void) {
    grabberGen.reset(new GenericGrabber());
    return grabberGen.get();
}
