/** @file genericCam.h
 *
 * implementation - Generic Camera Grabber
 *
 */

#ifndef GENERIC_CAMERA_H_INCLUDED
#define GENERIC_CAMERA_H_INCLUDED

#include "grabber.h"

namespace grabber {
    /// create a single grabber (Generic Camera)
    Grabber* createGrabberGenericCam(void);
}

/// Grabber implementation
class GenericGrabber : public grabber::Grabber {
    public:
        /// Pointer
        typedef std::unique_ptr<GenericGrabber> Ptr;

        /// Construction
        GenericGrabber(void);

        /// Destructor
        ~GenericGrabber(void);

        /// Grab image and/or point cloud
        bool grab();

        /// Grab sequence of image (duration in seconds)
        void getSequence(const int duration);

        /// Calibrate sensor
        void calibrate(void);

    private:
        /// cv grabber
        cv::VideoCapture capture;

        /// start timestamp
        std::chrono::high_resolution_clock::time_point startT;
};

#endif // GENERIC_CAMERA_H_INCLUDED
