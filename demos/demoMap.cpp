// CUDA
#include "CUDAmap.h"

using namespace walkers;
TimeMeasurements timeloadimagepair;
TimeMeasurements timedepth2cloud;
TimeMeasurements timeloop;
TimeMeasurements timepointtrans;
TimeMeasurements pointcl2elli;

mapping::Point3D transformPoint(const  grabber::PointCloud& cloud1, const walkers::Mat34& camPose, int index)
{
    const auto& point = cloud1[index];
    walkers::Mat34 pointCL(walkers::Mat34::Identity());

    pointCL(0, 3) = point.x;
    pointCL(1, 3) = point.y;
    pointCL(2, 3) = point.z;

    pointCL = camPose * pointCL;

    // Adjustments if needed
    pointCL(1, 3) = -pointCL(1, 3);

    mapping::Point3D point3D(pointCL(0, 3), pointCL(1, 3), pointCL(2, 3), point.r, point.g, point.b);

    return point3D;
}

// Function to be executed by each thread
void processPoints(const grabber::PointCloud& cloud1, const walkers::Mat34& camPose, int start, int end, std::vector<mapping::Point3D>& result)
{
    for(int i = start; i < end; i++)
    {
        result[i] = transformPoint(cloud1, camPose, i);
    }
}

void readTrajectoryData(const std::string& filePath, std::vector<std::vector<double>>& trajectoryData)
{
    // Open the file
    std::ifstream file(filePath);
    if(!file.is_open())
    {
        std::cout << "Failed to open trajectory file: " << filePath.c_str();
        return;
    }

    // Read and parse the data line by line
    std::string line;
    while(std::getline(file, line))
    {
        std::istringstream iss(line);
        std::vector<double> rowData;
        double value;

        // Skip the first column and read the rest
        iss >> value;
        while(iss >> value)
        {
            rowData.push_back(value);
        }

        if(!rowData.empty())
        {
            trajectoryData.push_back(rowData);
        }
    }

    // Close the file
    file.close();
}

void loadImagePair(int i, cv::Mat& depth, cv::Mat& rgb)
{
    std::string inputDepth = "../../resources/foto/salon/depth/" + std::to_string(i) + ".png";
    std::string inputRGB = "../../resources/foto/salon/rgb/" + std::to_string(i) + ".png";

    std::cout << "inputDepth: " << inputDepth << "\n \n";
    std::cout << "inputRGB: " << inputRGB << "\n \n";

    depth = cv::imread(inputDepth, cv::IMREAD_ANYDEPTH);
    rgb = cv::imread(inputRGB, cv::IMREAD_COLOR);
}

void processSimulation(QGLVisualizer* _visu, Gaussmap* map3D)
{
    char option = 0;
    std::string state = "";
    std::vector<std::vector<double>> trajectoryData;

    while (option!='q')
    {
        if (state=="read_clouds")
        {
            std::cout << "state \n";
            ///Image2CloudConv
            std::string kinectModelConfig = "cameraModels/KinectModel_ICL.xml";
            grabber::CameraModel kinectModel(kinectModelConfig);

            ///camera position
            walkers::Mat34 camPose;

            // Images containers
            cv::Mat depthImage;
            cv::Mat colorImage;

            for(size_t imageNo=0; imageNo<1; imageNo++)
            {
                std::chrono::steady_clock::time_point begin_test = std::chrono::steady_clock::now();

                std::chrono::steady_clock::time_point begin_load = std::chrono::steady_clock::now();
                loadImagePair(int(imageNo+1), depthImage, colorImage);
                std::chrono::steady_clock::time_point end_load = std::chrono::steady_clock::now();
                timeloadimagepair.addMeasurement(end_load-begin_load);

                std::chrono::steady_clock::time_point begin_depth2load = std::chrono::steady_clock::now();
                grabber::PointCloud cloud1 = kinectModel.depth2cloud(depthImage,colorImage,0.0002,0.0,8.0);
                std::chrono::steady_clock::time_point end_depth2load = std::chrono::steady_clock::now();
                timedepth2cloud.addMeasurement(end_depth2load-begin_depth2load);

                /// DEBUG - to be removed

                for(int i = 0; i < 20; ++i)
                {
                    std::cout << "image[" << i << "]: x: " << cloud1[i].x << "; y: " << cloud1[i].y << "; z: " << cloud1[i].z <<
                        "; r: " << cloud1[i].r << "; g: " << cloud1[i].g << "; b: " << cloud1[i].b << std::endl;
                    std::cout << std::endl;
                }

                /// END OF DEBUG

                //camPose = walkers::toSE3(Eigen::Vector3d(trajectoryData[imageNo][0], trajectoryData[imageNo][1],trajectoryData[imageNo][2]),
                //                         Quaternion(trajectoryData[imageNo][6], trajectoryData[imageNo][3], trajectoryData[imageNo][4], trajectoryData[imageNo][5]));

                /// point cloud

                std::chrono::steady_clock::time_point begin_pointtrans = std::chrono::steady_clock::now();
                //const int numThreads = 8;
                //std::vector<std::thread> threads;
                //std::vector<mapping::Point3D> combinedCloudMap(cloud1.size());

                // Split the work among threads
                // int chunkSize = static_cast<int>(cloud1.size()) / numThreads;
                // int start = 0;

                // for (int i = 0; i < numThreads; i++) {
                //     int end = start + chunkSize;
                //     threads.emplace_back(processPoints, std::ref(cloud1), std::ref(camPose), start, end, std::ref(combinedCloudMap));
                //     start = end;
                // }

                // Wait for all threads to finish
                // for (auto& thread : threads) {
                //     thread.join();
                // }
                std::chrono::steady_clock::time_point end_pointtrans = std::chrono::steady_clock::now();
                timepointtrans.addMeasurement(end_pointtrans-begin_pointtrans);

                std::chrono::steady_clock::time_point begin_ndtom = std::chrono::steady_clock::now();
                //map3D->insertCloud(combinedCloudMap,walkers::Mat34::Identity(),mapping::updateMethodType::TYPE_NDTOM,false);
                std::chrono::steady_clock::time_point end_ndtom = std::chrono::steady_clock::now();
                pointcl2elli.addMeasurement(end_ndtom-begin_ndtom);
                //combinedCloudMap.clear();

                std::chrono::steady_clock::time_point end_test = std::chrono::steady_clock::now();
                timeloop.addMeasurement(end_test-begin_test);
            }

            state = "";
        }

        if (state=="read_clouds_cuda") {
            std::cout << "state \n";
            ///Image2CloudConv
            std::string kinectModelConfig = "cameraModels/KinectModel_ICL.xml";
            grabber::CameraModel kinectModel(kinectModelConfig);

            ///camera position
            walkers::Mat34 camPose;
            double camPoseArray[16];

            // Images containers
            cv::Mat depthImage;
            cv::Mat colorImage;

            // Need to load first pair of images to set some initial values
            loadImagePair(1, depthImage, colorImage);
            int cloudSize = depthImage.cols * depthImage.rows;

            // CUDA host containers/variables
            cv::cuda::GpuMat gpuDepthImage;
            cv::cuda::GpuMat gpuColorImage;
            SimplePoint* gpuPointCloud;
            SimplePoint* pointCloud = new SimplePoint[static_cast<size_t>(cloudSize)];
            //SimplePoint* outputPointCloud = new SimplePoint[static_cast<size_t>(cloudSize)];
            //SimplePoint* gpuInputPointCloud;
            //SimplePoint* gpuOutputPointCloud;

            //std::vector<mapping::Point3D> PointCloud;
            //PointCloud.reserve(static_cast<size_t>(cloudSize));

            // Allocate memory for frequently used variables/containers
            //allocateDepth2CloudMemory(gpuDepthImage, gpuColorImage, depthImage, colorImage, gpuPointCloud, kinectModel, cloudSize);
            //allocateTransformPointsMemory(gpuInputPointCloud, gpuOutputPointCloud, cloudSize);

            allocateMemory(gpuDepthImage, gpuColorImage, depthImage, colorImage, kinectModel, 0.0002, 0.0, 8.0, gpuPointCloud);

            //1507->1508
            for (size_t imageNo=0;imageNo<1;imageNo++)
            {
                std::chrono::steady_clock::time_point begin_test = std::chrono::steady_clock::now();

                // Loading images from PC memory
                std::chrono::steady_clock::time_point begin_load = std::chrono::steady_clock::now();
                loadImagePair(int(imageNo+1), depthImage, colorImage);
                std::chrono::steady_clock::time_point end_load = std::chrono::steady_clock::now();
                timeloadimagepair.addMeasurement(end_load-begin_load);

                // Depth image to point cloud transformation
                std::chrono::steady_clock::time_point begin_depth2load = std::chrono::steady_clock::now();
                //CUDADepth2Cloud(gpuDepthImage, depthImage, gpuColorImage, colorImage, kinectModel, 0.0002, 0.0, 8.0, gpuPointCloud, pointCloud, cloudSize);

                CUDAmap(gpuDepthImage, depthImage, gpuColorImage, colorImage, gpuPointCloud, pointCloud);

                std::chrono::steady_clock::time_point end_depth2load = std::chrono::steady_clock::now();
                timedepth2cloud.addMeasurement(end_depth2load-begin_depth2load);

                /// DEBUG - to be removed

                for(int i = 0; i < 20; ++i)
                {
                    std::cout << "image[" << i << "]: x: " << pointCloud[i].x << "; y: " << pointCloud[i].y << "; z: " << pointCloud[i].z <<
                        "; r: " << pointCloud[i].r << "; g: " << pointCloud[i].g << "; b: " << pointCloud[i].b << std::endl;
                    std::cout << std::endl;
                }

                /// END OF DEBUG

                // Camera pose
                //camPose = walkers::toSE3(Eigen::Vector3d(trajectoryData[imageNo][0], trajectoryData[imageNo][1],trajectoryData[imageNo][2]),
                //                         Quaternion(trajectoryData[imageNo][6], trajectoryData[imageNo][3], trajectoryData[imageNo][4], trajectoryData[imageNo][5]));

                // Flatten the matrix into a 1D array
                // for(int i = 0; i < 4; ++i)
                // {
                //     for(int j = 0; j < 4; ++j)
                //     {
                //         camPoseArray[i * 4 + j] = camPose(i, j);
                //     }
                // }

                // Manually set the last row for affine transformation matrix
                // camPoseArray[3 * 4 + 0] = 0.0;
                // camPoseArray[3 * 4 + 1] = 0.0;
                // camPoseArray[3 * 4 + 2] = 0.0;
                // camPoseArray[3 * 4 + 3] = 1.0;

                // Point cloud
                std::chrono::steady_clock::time_point begin_pointtrans = std::chrono::steady_clock::now();
                //CUDATransformPoints(camPoseArray, pointCloud, outputPointCloud, gpuInputPointCloud, gpuOutputPointCloud, cloudSize);
                std::chrono::steady_clock::time_point end_pointtrans = std::chrono::steady_clock::now();
                timepointtrans.addMeasurement(end_pointtrans-begin_pointtrans);

                // for(size_t i = 0; i < static_cast<size_t>(cloudSize); i += static_cast<size_t>(cloudSize/50))
                // {
                //     std::cout << "Point[" << i << "]: " << outputPointCloud[i].x << " " << outputPointCloud[i].y << " " << outputPointCloud[i].z << " " <<
                //         outputPointCloud[i].r << " " << outputPointCloud[i].g << " " << outputPointCloud[i].b << " " << outputPointCloud[i].a << std::endl;
                // }

                // PointCloud.resize(static_cast<size_t>(cloudSize));

                // unsigned int numThreads = 8;
                // std::vector<std::thread> threads(numThreads);

                // Calculate chunk size for each thread
                //size_t chunkSize = static_cast<size_t>(cloudSize) / numThreads;

                // Create and start threads
                // for (unsigned int i = 0; i < numThreads; ++i)
                // {
                //     size_t start = i * chunkSize;
                //     size_t end = (i == numThreads - 1) ? static_cast<size_t>(cloudSize) : start + chunkSize;

                //     threads[i] = std::thread([start, end, &outputPointCloud, &PointCloud]() {
                //         for (size_t j = start; j < end; ++j)
                //         {
                //             const SimplePoint& sp = outputPointCloud[j];
                //             int r = static_cast<int>(sp.r);
                //             int g = static_cast<int>(sp.g);
                //             int b = static_cast<int>(sp.b);
                //             int a = static_cast<int>(sp.a);

                //             //std::cout << "Point[" << j << "]: " << sp.x << " " << sp.y << " " << sp.z << " " << r << " " << g << " " << b << " " << a << std::endl;

                //             mapping::Point3D pd(sp.x, sp.y, sp.z, r, g, b, a);
                //             PointCloud[j] = pd;
                //         }
                //     });
                // }

                // Join threads
                // for (auto& thread : threads)
                // {
                //     thread.join();
                // }

                // for(size_t i = 0; i < static_cast<size_t>(cloudSize); i += static_cast<size_t>(cloudSize/50))
                // {
                //     std::cout << "Point[" << i << "]: " << PointCloud[i].position.x() << " " << PointCloud[i].position.y() << " " << PointCloud[i].position.z() << " " <<
                //         PointCloud[i].color.r << " " << PointCloud[i].color.g << " " << PointCloud[i].color.b << " " << PointCloud[i].color.a << std::endl;
                // }

                // Map - NDT-OM
                std::chrono::steady_clock::time_point begin_ndtom = std::chrono::steady_clock::now();
                //map3D->insertCloud(PointCloud,walkers::Mat34::Identity(),mapping::updateMethodType::TYPE_NDTOM,false);
                std::chrono::steady_clock::time_point end_ndtom = std::chrono::steady_clock::now();
                pointcl2elli.addMeasurement(end_ndtom-begin_ndtom);

                //PointCloud.clear();

                std::chrono::steady_clock::time_point end_test = std::chrono::steady_clock::now();
                timeloop.addMeasurement(end_test-begin_test);
            }

            // Deallocate memory
            //deallocateDepth2CloudMemory(gpuDepthImage, gpuColorImage, gpuPointCloud);
            //deallocateTransformPointsMemory(gpuInputPointCloud, gpuOutputPointCloud);
            deallocateMemory(gpuDepthImage, gpuColorImage, gpuPointCloud);
            delete[] pointCloud;
            //delete[] outputPointCloud;

            state = "";
        }

        std::cout << "Select option (type '?' for help): ";

        std::cin >> option;

        if(option=='q')
        {
            _visu->closeWindow();
            std::cout << "Quit.\n";
        }

        else if(option=='s')
        {
            std::vector<planner::PoseSE3> pathICL;
            /// path with camera pointers
            readTrajectoryData("../../resources/foto/salon/livingRoom0.gt.freiburg", trajectoryData);

            state = "read_clouds";
        }

        else if(option=='c')
        {
            std::vector<planner::PoseSE3> pathICL;
            /// path with camera pointers
            readTrajectoryData("../../resources/foto/salon/livingRoom0.gt.freiburg", trajectoryData);

            state = "read_clouds_cuda";
        }

        else if(option == '?')
        {
            std::cout << "Available options:\n"
                      << "s - start turning images to elipsoid clouds\n"
                      << "c - start turning images to elipsoid clouds using CUDA\n"
                      << "t - display mean a standard deviation of measured times";
        }

        else if(option == 't')
        {
            std::cout<<"Average for image pair loading: "<< timeloadimagepair.calculateMean() << " [µs] \n Standard Deviatios for image pair loading "
                      <<timeloadimagepair.calculateStandardDeviation()<<" [µs] \n \n";
            std::cout<<"Average for depth image to cloud conversion: "<< timedepth2cloud.calculateMean() << " [µs] \n Standard Deviatios for depth image to cloud conversion "
                      << timedepth2cloud.calculateStandardDeviation() <<" [µs] \n \n";
            std::cout<<"Average for point translation calculation time: "<< timepointtrans.calculateMean() << " [µs] \n Standard Deviatios for point translation calculation time "
                      <<timepointtrans.calculateStandardDeviation()<<" [µs] \n \n";
            std::cout<<"Average for point cloud to ellipsoid cloud calculation time: "<< pointcl2elli.calculateMean()
                      << " [µs] \n Standard Deviatios for point cloud to ellipsoid cloud calculation time "
                      <<pointcl2elli.calculateStandardDeviation()<<" [µs] \n \n";

            std::cout<<"Average for test "<< timeloop.calculateMean() << " [µs] \n Standard Deviatios for test "<< timeloop.calculateStandardDeviation() <<" [µs] \n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


int main(int argc, char** argv)
{
    try
    {
        setlocale(LC_NUMERIC,"C");
        tinyxml2::XMLDocument config;
        config.LoadFile("../../resources/configGlobal.xml");
        if(config.ErrorID())
        {
            std::cout << "unable to load config file.\n";
            return 0;
        }

        auto rootXML = config.FirstChildElement( "configGlobal" );
        std::string simConfig(rootXML->FirstChildElement( "environment" )->FirstChildElement("config")->GetText());
        std::string simType(rootXML->FirstChildElement( "environment" )->FirstChildElement("type")->GetText());

        std::string plannerConfig(rootXML->FirstChildElement( "Planner" )->FirstChildElement("config")->GetText());
        std::string plannerType(rootXML->FirstChildElement( "Planner" )->FirstChildElement("type")->GetText());

        std::string motionCtrlConfig(rootXML->FirstChildElement( "MotionController" )->FirstChildElement("config")->GetText());
        std::string motionCtrlType(rootXML->FirstChildElement( "MotionController" )->FirstChildElement("type")->GetText());

        std::string robotConfig(rootXML->FirstChildElement( "Robot" )->FirstChildElement("config")->GetText());
        std::string robotType(rootXML->FirstChildElement( "Robot" )->FirstChildElement("type")->GetText());

        std::string mapConfig(rootXML->FirstChildElement( "Mapping" )->FirstChildElement("config")->GetText());
        std::string mapType(rootXML->FirstChildElement( "Mapping" )->FirstChildElement("type")->GetText());

        std::string visualizerConfig(rootXML->FirstChildElement( "Visualizer" )->FirstChildElement("config")->GetText());
        std::string visualizerType(rootXML->FirstChildElement( "Visualizer" )->FirstChildElement("type")->GetText());

        std::string gamepadConfig(rootXML->FirstChildElement( "HMIDevice" )->FirstChildElement("config")->GetText());
        std::string gamepadName(rootXML->FirstChildElement( "HMIDevice" )->FirstChildElement("name")->GetText());

        std::string coldetConfig(rootXML->FirstChildElement( "CollisionDetection" )->FirstChildElement("config")->GetText());
        std::string coldetType(rootXML->FirstChildElement( "CollisionDetection" )->FirstChildElement("type")->GetText());

        QApplication application(argc,argv);

        setlocale(LC_NUMERIC,"C");
        glutInit(&argc, argv);

        QGLVisualizer visu(visualizerConfig, robotConfig, robotType, coldetType, coldetConfig);
        visu.setWindowTitle("Simulator viewer");
        visu.show();

        /// elevation map
        ElevationMap map(mapConfig);

        /// map NDTOM
        int mapSize = 1024;
        double resolution = 0.05;
        double raytraceFactor = 0.05;
        int pointThreshold = 10;
        std::unique_ptr<mapping::Map> map3D = mapping::createMapGauss(mapSize, resolution, raytraceFactor, pointThreshold);
        ((Gaussmap*)map3D.get())->attachVisualizer(&visu);
        mapping::PointCloud cloud;

        // Run main loop.
        std::thread processThr(processSimulation,&visu, (Gaussmap*)map3D.get());
        application.exec();
        processThr.join();
        std::cout << "Finished\n";
        return 1;
    }

    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
