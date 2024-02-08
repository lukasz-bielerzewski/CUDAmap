#include "../include/Defs/defs.h"
#include <tinyxml2.h>
#include "../include/Visualizer/Qvisualizer.h"
#include "../include/Simulator/simulatorPhysX.h"
#include "../include/Simulator/simulatorODE.h"
#include "Utilities/stopwatch.h"
#include "Mapping/elevationMap.h"
#include "include/Defs/planner_defs.h"
#include "RobotModelMat/AtlasRobot.h"
//#include "mapping/mapping.h"
#include <GL/glut.h>
#include "Defs/qt.h"
#include "time.h"
#include "Optimizer/cmaes.h"
#include <iostream>

bool init(false);

void processSimulation(simulator::Simulator* sim, QApplication* application){//, Planner* planner){
    Stopwatch<std::chrono::milliseconds> watch;
    watch.start();
    bool doOnce(true);
    while (sim->getCurrentTime()<1.4){
        if (doOnce&&sim->getCurrentTime()>0.7){
            double tmp[] = {0.7854,0.51888,-1.0897, 0.0,0.41888,-1.9897, -0.7854,0.51888,-1.0897, 0.7854,0.41888,-1.9897, 0.0,0.51888,-1.0897, -0.7854,0.41888,-1.9897};
            std::vector<double> angle(tmp,tmp+18);
            sim->setPosition(angle);
            doOnce=false;
        }
        //std::cout << watch.stop() << " simulation running\n";
    }
    application->exit();
    std::cout << "sim Time: " << sim->getCurrentTime() << "\n";
    std::cout << "real Time: " << watch.stop()/1000 << "\n";
}

double runPhysX(Eigen::VectorXd x) {
    try {
        setlocale(LC_NUMERIC,"C");
        tinyxml2::XMLDocument config;
        config.LoadFile("../../resources/configGlobal.xml");
        if (config.ErrorID())
            std::cout << "unable to load config file.\n";
        auto rootXML = config.FirstChildElement( "configGlobal" );
        std::string simConfig(rootXML->FirstChildElement( "environment" )->FirstChildElement("config")->GetText());
        std::string simType(rootXML->FirstChildElement( "environment" )->FirstChildElement("type")->GetText());

        std::string sceneConfig(rootXML->FirstChildElement( "Scene" )->FirstChildElement("config")->GetText());

        std::string plannerConfig(rootXML->FirstChildElement( "Planner" )->FirstChildElement("config")->GetText());
        std::string plannerType(rootXML->FirstChildElement( "Planner" )->FirstChildElement("type")->GetText());

        std::string robotConfig(rootXML->FirstChildElement( "Robot" )->FirstChildElement("config")->GetText());
        std::string robotType(rootXML->FirstChildElement( "Robot" )->FirstChildElement("type")->GetText());

        std::string mapConfig(rootXML->FirstChildElement( "Mapping" )->FirstChildElement("config")->GetText());
        std::string mapType(rootXML->FirstChildElement( "Mapping" )->FirstChildElement("type")->GetText());

        std::string visualizerConfig(rootXML->FirstChildElement( "Visualizer" )->FirstChildElement("config")->GetText());
        std::string visualizerType(rootXML->FirstChildElement( "Visualizer" )->FirstChildElement("type")->GetText());

        int argc;
        char* argv[] = { NULL };
        QApplication application(argc,argv);
        setlocale(LC_NUMERIC,"C");
        if (!init){
            glutInit(&argc, argv);
            init = true;
        }
        simulator::Simulator* sim = nullptr;
        if (simType.compare("physx") == 0) {
            Eigen::VectorXd xcopy(x);
            for (int i=0;i<7;i++)
                xcopy(i)= ((x(i)+1.0)/2.0);
            xcopy(0)*=10000;
            xcopy(4)*=1000;
            std::cout << "framerate: " << int(xcopy(0)) <<" ,staticFriction: " << xcopy(1) << ", dynamicFriction: " << xcopy(2) << ", restitution: " << xcopy(3) << ", P: " << xcopy(4) << ", I: " << xcopy(5) << ", D: " << xcopy(6) << "\n";
            sim = simulator::createPhysXSimulator(simConfig,robotConfig, sceneConfig, (int)xcopy(0),xcopy(1),xcopy(2),xcopy(3),xcopy(4),xcopy(5),xcopy(6));
        }
        else if (simType.compare("ode") == 0) {
            sim = simulator::createODESimulator(simConfig);
        }
        else {
            std::cout << "XML Error: Wrong simulation type.\n";
        }

        QGLVisualizer visu(visualizerConfig);
        visu.setWindowTitle("Simulator viewer");
        visu.show();
        sim->attachVisualizer(&visu);
        // elevation map
        //ElevationMap map(mapConfig);
        double tmp[] = {0.7854,0.41888,-1.9897, 0.0,0.41888,-1.9897, -0.7854,0.41888,-1.9897, 0.7854,0.41888,-1.9897, 0.0,0.41888,-1.9897, -0.7854,0.41888,-1.9897};
        std::vector<double> angle(tmp,tmp+18);
        //vector<double> angle(18,0);

        walkers::Vec3 initPos;
        sim->startSimulation();
        sim->setPosition(angle);
        sim->readRobotPosition(initPos);
        std::cout<<"init pos x:"<<initPos.x()<<",y:"<<initPos.y() << ",z: " << initPos.z() <<std::endl;
        // Run main loop.
        std::thread processThr(processSimulation, sim, &application);
        application.exec();
        processThr.join();
        walkers::Vec3 finalPos;
        sim->readRobotPosition(finalPos);
        std::cout << "FDS\n";
        std::vector<walkers::Mat34> trajectory = ((SimulatorPhysX *)sim)->getTrajectory();
        std::cout << "FDS1\n";
        sim->stopSimulation();
        std::cout<<"init pos x:"<<finalPos.x()<<",y:"<<finalPos.y() << ",z: " << finalPos.z() <<std::endl;
        double dist = sqrt(pow(finalPos.x()-initPos.x(),2.0)+pow(finalPos.y()-initPos.y(),2.0)+pow(finalPos.z()-initPos.z(),2.0));
        std::cout << "dist " << dist << "\n";

        walkers::Mat34 prevPose;
        prevPose.translation() = initPos.vector();
        dist = 0;
        for (auto & robotPose : trajectory){
            dist+=sqrt(pow(robotPose(0,3)-prevPose(0,3),2.0)+pow(robotPose(1,3)-prevPose(1,3),2.0)+pow(robotPose(2,3)-prevPose(2,3),2.0));
            prevPose = robotPose;
        }
        dist = dist/double(trajectory.size());
        std::cout << "Finished\n";
        return dist;
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
    return std::numeric_limits<double>::max();
}

int main(void) {
    try
    {
        setlocale(LC_NUMERIC,"C");
        tinyxml2::XMLDocument config;
        config.LoadFile("../../resources/configGlobal.xml");
        if (config.ErrorID())
            std::cout << "unable to load config file.\n";
        auto rootXML = config.FirstChildElement( "configGlobal" );
        std::string optimizerType(rootXML->FirstChildElement( "Optimizer" )->FirstChildElement( "name" )->GetText());
        std::string function("runPhysX");

        std::unique_ptr<Optimizer> optimizer;
        if (optimizerType == "CMAES")
        {
            std::string configFile(rootXML->FirstChildElement( "Optimizer" )->FirstChildElement( "config" )->GetText());
            optimizer = createCMAES(configFile);
        }
        else // Default
            optimizer = createCMAES();

        std::map<std::string, std::function<double(const Eigen::VectorXd&)>>  funcMap =
        {
            { "runPhysX", runPhysX}
        };

        // create objects and print configuration
        std::cout << "Current optimizer: " << optimizer->getName() << std::endl;
        std::cout << "Current function: " << function << std::endl;

        clock_t start, stop;
        double czas;
        start = clock();
        optimizer->Optimize(funcMap, function);
        stop = clock();
        czas = (double)(stop - start) / CLOCKS_PER_SEC;
        optimizer->output();

        Eigen::VectorXd xmin;
        optimizer->getResult(xmin);
        std::cout << "Cost function value: " << runPhysX(xmin) << "\n";
        std::cout << std::endl << "Execution time: " << czas << " s" << std::endl;

        }

    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
