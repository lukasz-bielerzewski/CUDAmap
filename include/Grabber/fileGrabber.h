/** @file fileGrabber.h
 *
 * implementation - Grabber loads images from file
 *
 */

#ifndef FILE_GRABBER_H_INCLUDED
#define FILE_GRABBER_H_INCLUDED

#include "grabber.h"
//#include "xtionAsus.h"

namespace grabber {
    /// create a single grabber (Generic Camera)
    Grabber* createFileGrabber(void);
}

/// Grabber implementation
class FileGrabber : public grabber::Grabber {
    public:
        /// Pointer
        typedef std::unique_ptr<FileGrabber> Ptr;

        /// Construction
        FileGrabber(void);

        /// Destructor
        ~FileGrabber(void);

        /// Grab image and/or point cloud
        bool grab();

        /// Set sequence properties
        void setSequence(const uint_fast32_t startFrameNo, const std::string& imagePrefix, const std::string& depthPrefix, const std::string& cloudPrefix);

        /// Grab sequence of image and save sequence to files (duration in number of frames)
        void getSequence(const int duration);

        /// Calibrate sensor
        void calibrate(void);

    private:
        /// start timestamp
        std::chrono::high_resolution_clock::time_point startT;

        /// file prefix (images)
        std::string imageSeqPrefix;

        /// file prefix (depth images)
        std::string depthSeqPrefix;

        /// file prefix (point clouds)
        std::string cloudSeqPrefix;

        /// file number
        uint_fast32_t fileNo;

        ///sensor model
        grabber::CameraModel cameraModel;
};

#endif // FILE_GRABBER_H_INCLUDED
