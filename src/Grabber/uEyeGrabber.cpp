#include "../include/Grabber/uEyeGrabber.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <fstream>
#include <limits>

using namespace flytracker;
using namespace std::chrono;

/// A single instance of UEye grabber
UEyeGrabber::Ptr grabberUEye;

UEyeGrabber::UEyeGrabber(void) : Grabber("uEye Grabber", TYPE_RGB) {
    frameId=0;
    width = 752;
    height = 480;
    using std::cout;
    using std::endl;
    image = cv::Mat(height, width, CV_8UC3);
    hCam = 0;
    char* ppcImgMem;
    int pid;
    INT nAOISupported = 0;
    double on = 1;
    double empty;
    int retInt = IS_SUCCESS;
    UINT nPixelClock = 28;
    double FPS = 60;
    retInt = is_InitCamera(&hCam, 0);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_InitCamera error");
        throw UeyeException(errorMsg);
    }
    retInt = is_SetColorMode(hCam, IS_CM_BGR8_PACKED);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_SetColorMode error");
        throw UeyeException(errorMsg);
    }
    retInt = is_ImageFormat(hCam, IMGFRMT_CMD_GET_ARBITRARY_AOI_SUPPORTED, (void*) &nAOISupported, sizeof(nAOISupported));
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_ImageFormat error");
        throw UeyeException(errorMsg);
    }
    retInt = is_AllocImageMem(hCam, width, height, 24, &ppcImgMem, &pid);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_AllocImageMem error");
        throw UeyeException(errorMsg);
    }
    retInt = is_SetImageMem(hCam, ppcImgMem, pid);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_SetImageMem error");
        throw UeyeException(errorMsg);
    }
    //set auto settings
    retInt = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_WHITEBALANCE, &on, &empty);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_SetAutoParameter error");
        throw UeyeException(errorMsg);
    }
    retInt = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_GAIN, &on, &empty);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_SetAutoParameter error");
        throw UeyeException(errorMsg);
    }
    // Set this pixel clock
    retInt = is_PixelClock(hCam, IS_PIXELCLOCK_CMD_SET,(void*)&nPixelClock, sizeof(nPixelClock));
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_PixelClock error");
        throw UeyeException(errorMsg);
    }
    retInt = is_SetFrameRate (hCam, FPS, &FPS);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_SetFrameRate error");
        throw UeyeException(errorMsg);
    }
    /*retInt = is_SetExternalTrigger(hCam, IS_SET_TRIGGER_SOFTWARE);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_SetExternalTrigger error");
        throw UeyeException(errorMsg);
    }*/
    retInt = is_CaptureVideo(hCam, IS_WAIT);
    //retInt = is_FreezeVideo(hCam, IS_WAIT);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_CaptureVideo error");
        throw UeyeException(errorMsg);
    }
}

UEyeGrabber::~UEyeGrabber(void) {
    int retInt = is_ExitCamera(hCam);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_ExitCamera error");
        throw UeyeException(errorMsg);
    }
}

bool UEyeGrabber::grab(void) {
    VOID* pMem;
    //mtx.lock();
    is_ForceTrigger(hCam);
    int retInt = is_GetImageMem(hCam, &pMem);
    if (retInt != IS_SUCCESS) {
        std::string errorMsg("is_GetImageMem error");
        throw UeyeException(errorMsg);
    }
    double  fps;
    is_GetFramesPerSecond(hCam, &fps);
        std::cout << "fps: " << fps << std::endl;
    memcpy(image.ptr(), pMem, width * height * 3);
    SensorFrame tmp;
    tmp.image = image.clone();
    tmp.timestamp = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::high_resolution_clock::now() - startT).count();
    mtx.lock();
    sensorFrames.push(tmp);
    mtx.unlock();
    return true;
}

/// Grab sequence of image and save sequence to files (duration in seconds)
void UEyeGrabber::getSequence(const uint_fast32_t duration){
    high_resolution_clock::time_point start = high_resolution_clock::now();
    movie.clear();
    movie.reserve(duration*80);
    while (1){ //recording
        grab(); // grab frame
        ULONG cnt = is_CameraStatus (hCam, IS_SEQUENCE_CNT, IS_GET_STATUS);
        if(cnt != frameId){
            frameId = cnt;
            //mtx.lock();
            movie.push_back(sensorFrame);
            //mtx.unlock();
            if (duration_cast<std::chrono::duration<unsigned> >(system_clock::now() - start).count()>duration){
                break;
            }
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

/// calibrate camera
void UEyeGrabber::calibrate(void) {

}

flytracker::Grabber* flytracker::createGrabberUEye(void) {
    grabberUEye.reset(new UEyeGrabber());
    return grabberUEye.get();
}
