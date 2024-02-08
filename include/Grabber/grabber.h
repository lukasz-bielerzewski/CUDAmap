/** @file grabber.h
 *
 * Point Cloud Grabber interface
 *
 */

#ifndef _GRABBER_H_
#define _GRABBER_H_

#include "Defs/grabber_defs.h"
#include "calibration.h"
#include <opencv2/highgui/highgui.hpp>
#include <mutex>

namespace grabber {
	/// Grabber interface
	class Grabber {
        public:

            /// Grabber type
            enum Type {
                    /// RGB camera
                    TYPE_RGB,
                    /// 2D Depth sensor
                    TYPE_2D_DEPTH,
                    /// 3D Depth sensor
                    TYPE_3D_DEPTH,
                    /// PrimeSense-based (Kinect, Asus, PrimeSense)
                    TYPE_PRIMESENSE,
                    /// Intel RealSense
                    TYPE_REALSENSE,
                    /// Intel RealSense v2
                    TYPE_REALSENSE2,
                    /// Read data from files
                    TYPE_FILE
            };

            /// overloaded constructor
            Grabber(const std::string _name, Type _type) : name(_name), type(_type) {};

            /// Name of the grabber
            virtual const std::string& getName() const { return name; }

            /// Returns current 2D image
            virtual const SensorFrame& getSensorFrame(void) {
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
                return sensorFrame;
            }

            /// Grab image and/or point cloud
            virtual bool grab() = 0;

            /// Grab sequence of image and save sequence to files (duration in seconds)
            virtual void getSequence(const int duration) = 0;

            /// Calibrate sensor
            virtual void calibrate() = 0;

            /// save timestamps to file
            virtual void exportTimestamps(const std::string& filePrefix) const {
                std::ostringstream oss; oss << filePrefix << ".tim";
                std::ofstream file(oss.str());

                int i = 1;
                for (std::vector<SensorFrame>::const_iterator it = movie.begin(); it!=movie.end(); it++){
                    file << i << " " << std::setprecision(std::numeric_limits<double>::digits10 + 1) << it->timestamp << std::endl;
                    i++;
                }
                file.close();
            }

            /// save image to file
            virtual void saveFrame(const std::string& filename, cv::Mat image) const {
                cv::imwrite( filename, image );
            }

            /// compute median
            static uint16_t median(const cv::Mat& inputImg, int u, int v, int kernelSize){
                int size = kernelSize/2;
                std::vector<short unsigned int> values;
                for (int i=v-size;i<v+size+1;i++){
                    for (int j=u-size;j<u+size+1;j++){
                        if ((i>=0)&&(j>=0)&&(j<inputImg.rows)&&(i<inputImg.cols)){
                            values.push_back(inputImg.at<uint16_t>(j,i));
                        }
                    }
                }
                std::sort(values.begin(), values.end());
                if (values.size()>0){
                    return (values[values.size()/2]);
                }
                else
                    return 0;
            }

            ///Apply median filter on the image
            static void medianFilter(const cv::Mat& inputImg, cv::Mat& outputImg, int kernelSize){
                for (int i = 0; i<inputImg.rows; i++){
                    for (int j=0;j<inputImg.cols;j++){
                        outputImg.at<uint16_t>(i,j)=median(inputImg, i, j, kernelSize);
                    }
                }
            }

            /// filter depth image
            static void filterDepthImage(const cv::Mat& input, cv::Mat& output, const std::string& filterType, int kernelSize){
                if (filterType == "median"){
                    if (kernelSize<6)
                        cv::medianBlur(input, output, kernelSize);
                    else {
                        medianFilter(input, output, kernelSize);
                    }
                }
                else if (filterType == "mean"){
                    cv::GaussianBlur(input, output, cv::Size(kernelSize,kernelSize),0,0);
                }
            }

            /// Virtual descrutor
            virtual ~Grabber() {}

        protected:
            /// Grabber name
            const std::string name;

            /// Grabber type
            Type type;

            /// 2D image
            SensorFrame sensorFrame;

            /// sequence
            std::queue<SensorFrame> sensorFrames;

            /// movie (sequence of frames)
            SensorFrame::Seq movie;

            /// Calibration methods
            Calibration calibrator;

            /// Mutex
            std::mutex mtx;
	};
};

#endif // _GRABBER_H_
