#include "../include/Grabber/xtionAsus.h"
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
AsusGrabber::Ptr grabberXtion;

AsusGrabber::AsusGrabber(void) : grabber::Grabber("Asus Grabber", TYPE_PRIMESENSE) {
    startT = std::chrono::high_resolution_clock::now();
    std::cout << "init ni\n";
    openNIWrap.InitOpenNI();
    std::cout << "inited\n";
}

AsusGrabber::~AsusGrabber(void) {

}

AsusGrabber::Config::Config(std::string configFilename){
    setlocale(LC_NUMERIC,"C");
    tinyxml2::XMLDocument config;
    std::string filename = "../../resources/" + configFilename;
    config.LoadFile(filename.c_str());
    if (config.ErrorID())
        std::cout << "unable to load Kinect config file.\n";
    tinyxml2::XMLElement * model = config.FirstChildElement( "Model" );
    model->FirstChildElement( "focalLength" )->QueryDoubleAttribute("fx", &cameraModel.focalLength[0]);
    model->FirstChildElement( "focalLength" )->QueryDoubleAttribute("fy", &cameraModel.focalLength[1]);
    model->FirstChildElement( "focalAxis" )->QueryDoubleAttribute("Cx", &cameraModel.focalAxis[0]);
    model->FirstChildElement( "focalAxis" )->QueryDoubleAttribute("Cy", &cameraModel.focalAxis[1]);
    model->FirstChildElement( "variance" )->QueryDoubleAttribute("sigmaU", &cameraModel.varU);
    model->FirstChildElement( "variance" )->QueryDoubleAttribute("sigmaV", &cameraModel.varV);
    model->FirstChildElement( "varianceDepth" )->QueryDoubleAttribute("c3", &cameraModel.distVarCoefs[0]);
    model->FirstChildElement( "varianceDepth" )->QueryDoubleAttribute("c2", &cameraModel.distVarCoefs[1]);
    model->FirstChildElement( "varianceDepth" )->QueryDoubleAttribute("c1", &cameraModel.distVarCoefs[2]);
    model->FirstChildElement( "varianceDepth" )->QueryDoubleAttribute("c0", &cameraModel.distVarCoefs[3]);
    tinyxml2::XMLElement * posXML = config.FirstChildElement( "pose" );
    double query[4];
    posXML->QueryDoubleAttribute("qw", &query[0]); posXML->QueryDoubleAttribute("qx", &query[1]); posXML->QueryDoubleAttribute("qy", &query[2]); posXML->QueryDoubleAttribute("qz", &query[3]);
    walkers::Quaternion q(query[0], query[1], query[2], query[3]);
    posXML->QueryDoubleAttribute("x", &query[0]); posXML->QueryDoubleAttribute("y", &query[1]); posXML->QueryDoubleAttribute("z", &query[2]);
    walkers::Vec3 pos(query[0], query[1], query[2]);
    cameraModel.pose = q*pos;
}

bool AsusGrabber::grab(void) {

    //image.convertTo(image, CV_8UC1, 255.0/1024.0);
    SensorFrame tmp;
    openNIWrap.AcquireDepthFrame(tmp.depth);
    openNIWrap.AcquireColorFrame(tmp.image);
    //tmp.image = image.clone();
    tmp.timestamp = (double)std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::high_resolution_clock::now() - startT).count();
    mtx.lock();
    sensorFrames.push(tmp);
    mtx.unlock();
    return true;
}

/// Grab sequence of image and save sequence to files (duration in seconds)
void AsusGrabber::getSequence(const int duration){
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
const SensorFrame& AsusGrabber::getSensorFrame(void) {
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
void AsusGrabber::calibrate(void) {

}

grabber::Grabber* grabber::createGrabberAsusXtionPro(void) {
    grabberXtion.reset(new AsusGrabber());
    return grabberXtion.get();
}

grabber::Grabber* grabber::createGrabberAsusXtionPro(std::string configFile) {
    grabberXtion.reset(new AsusGrabber(configFile));
    return grabberXtion.get();
}
