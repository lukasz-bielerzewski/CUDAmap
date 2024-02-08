// @author: Dominik Belter
// based on the Marek Kopicki's code (boost server)

#include "RPC/motionControlServer.h"
#include "../include/Defs/defs.h"
#include <tinyxml2.h>

#include "../include/Simulator/simulator.h"
#ifdef BUILD_WITH_SIMULATOR
#include "../include/Simulator/simulatorPhysX.h"
#include "../include/Simulator/simulatorODE.h"
#endif

#include "Board/boardDynamixel.h"
#ifdef BUILD_WALKERS_BOARD_GALGO
#include "Board/Galgo/boardGalgo.h"
#endif
#include "MotionControl/controllerMessor2.h"
#ifdef BUILD_WALKERS_BOARD_GALGO
#include "MotionControl/controllerGalgo.h"
#endif

#ifdef BUILD_VISUALIZER
#include "Defs/qt.h"
#endif
#include "Utilities/objectsMesh.h"

#include <iostream>
#include <thread>

#ifdef BUILD_VISUALIZER
ObjectsMesh objects3D;
#endif

void requestHandler(walkers::Stream& stream, simulator::Simulator& simulator, controller::Board& board, walkers::RobotController& controller, bool useSim) {
    // first 4 bytes encode request type
    std::uint32_t cmd;
    walkers::StreamRead(stream, cmd);
    // respond to request
    switch (cmd) {
    case HandlerRPCMotion::ROB_START_SIMULATION:
    {
        std::cout << "start sim\n";
        if (useSim)
            simulator.startSimulation();
        std::cout << "sim started\n";
    //    else
     //       throw std::runtime_error("Request: " + std::to_string(cmd) + " unavailable for device.\n");
        walkers::StreamWrite(stream, (int)0);
        break;
    }
    case HandlerRPCMotion::ROB_STOP_SIMULATION:
    {
        std::cout << "stop sim\n";
        if (useSim)
            simulator.stopSimulation();
        else
            throw std::runtime_error("Request: " + std::to_string(cmd) + " unavailable for device.\n");
        walkers::StreamWrite(stream, (int)0);
        break;
    }
    case HandlerRPCMotion::ROB_ATTACH_VISUALIZER:
    {
        std::cout << "atttach sim\n";
        throw std::runtime_error("Request: " + std::to_string(cmd) + " not implemented\n");
        break;
    }
    case HandlerRPCMotion::ROB_READ_ROBOT_POS:
    {
        walkers::Vec3 position;
        if (useSim)
            simulator.readRobotPosition(position);
        else
            throw std::runtime_error("Request: " + std::to_string(cmd) + " unavailable for device.\n");
        walkers::StreamWrite(stream, position);
        break;
    }
    case HandlerRPCMotion::ROB_GET_HEIGHTFIELD_TYPE:
    {
        int type;
        if (useSim)
            type = simulator.getTerrainType();
        else
            throw std::runtime_error("Request: " + std::to_string(cmd) + " unavailable for device.\n");
        walkers::StreamWrite(stream, type);
        break;
    }
    case HandlerRPCMotion::ROB_GET_HEIGHTFIELD:
    {
        simulator::RenderObjectHeightmap heightmap;
        if (useSim)
            simulator.getHeightfield(heightmap);
        //else
//            throw std::runtime_error("Request: " + std::to_string(cmd) + " unavailable for device.\n");
        walkers::StreamWrite(stream, heightmap.heightmap.size());
        walkers::StreamWrite(stream, heightmap.heightmap[0].size());
        for (size_t i=0;i<heightmap.heightmap.size();i++){
            for (size_t j=0;j<heightmap.heightmap[i].size();j++){
                walkers::StreamWrite(stream, heightmap.heightmap[i][j]);
                for (int colNo=0;colNo<3;colNo++)
                    walkers::StreamWrite(stream, heightmap.heightmapColors[i][j].at(colNo));
            }
        }
        for (int colNo=0;colNo<3;colNo++)
            walkers::StreamWrite(stream, heightmap.color.at(colNo));
        walkers::StreamWrite(stream, heightmap.mat);
        walkers::StreamWrite(stream, heightmap.type);
        walkers::StreamWrite(stream, heightmap.x);
        walkers::StreamWrite(stream, heightmap.scaleX);
        walkers::StreamWrite(stream, heightmap.y);
        walkers::StreamWrite(stream, heightmap.scaleY);
        walkers::StreamWrite(stream, heightmap.z);
        walkers::StreamWrite(stream, heightmap.scaleZ);
        break;
    }
    case HandlerRPCMotion::ROB_GET_RENDER_OBJECTS:
    {
        simulator::RenderObject::Seq renderObjects;
#ifdef BUILD_VISUALIZER
        if (useSim)
            simulator.getVisualizationObjects(renderObjects);
        else
            throw std::runtime_error("Request: " + std::to_string(cmd) + " unavailable for device.\n");
#endif
        walkers::StreamWrite(stream, renderObjects.size());
        for (size_t i=0;i<renderObjects.size();i++){
            for (int j=0;j<4;j++)
                walkers::StreamWrite(stream, renderObjects[i].color[j]);
            walkers::StreamWrite(stream, renderObjects[i].mat);
            walkers::StreamWrite(stream, renderObjects[i].type);
            walkers::StreamWrite(stream, renderObjects[i].x);
            walkers::StreamWrite(stream, renderObjects[i].scaleX);
            walkers::StreamWrite(stream, renderObjects[i].y);
            walkers::StreamWrite(stream, renderObjects[i].scaleY);
            walkers::StreamWrite(stream, renderObjects[i].z);
            walkers::StreamWrite(stream, renderObjects[i].scaleZ);
        }
        break;
    }
    case HandlerRPCMotion::ROB_GET_3DOBJECTS:
    {
        //Objects3DS objects3Ds; // something is wrong with OBjects3DS type
#ifdef BUILD_VISUALIZER
        if (useSim)
            simulator.get3DObjects(objects3D);
        else
            throw std::runtime_error("Request: " + std::to_string(cmd) + " unavailable for device.\n");
        walkers::StreamWrite(stream, objects3D.objects.size());
        for (size_t i = 0; i<objects3D.objects.size();i++){
            walkers::StreamWrite(stream, objects3D.filenames[i]);
            walkers::StreamWrite(stream, objects3D.objects[i].name);
            walkers::StreamWrite(stream, objects3D.objects[i].vertices.size());
            for (size_t j=0;j<objects3D.objects[i].vertices.size();j++){
                 walkers::StreamWrite(stream, objects3D.objects[i].vertices[j]);
            }
            walkers::StreamWrite(stream, objects3D.objects[i].polygons.size());
            for (size_t j=0;j<objects3D.objects[i].polygons.size();j++){
                 walkers::StreamWrite(stream, objects3D.objects[i].polygons[j]);
            }
        }
#endif
        break;
    }
    case HandlerRPCMotion::ROB_SET_POSITION_JOINT:
    {
        std::uint32_t legNo; std::uint32_t jointNo; double angle;
        walkers::StreamRead(stream, legNo);
        walkers::StreamRead(stream, jointNo);
        walkers::StreamRead(stream, angle);
        if (useSim)
            simulator.setPosition(legNo, jointNo, angle);
        else
            board.setPosition(legNo, jointNo, angle);
        break;
    }
    case HandlerRPCMotion::ROB_SET_POSITION_LEG:
    {
        std::uint32_t legNo; std::uint32_t jointsNo; std::vector<double> angles;
        walkers::StreamRead(stream, legNo);
        walkers::StreamRead(stream, jointsNo);
        angles.resize(jointsNo);
        for (size_t i=0;i<jointsNo;i++)
            walkers::StreamRead(stream, angles[i]);
            /*
        if (useSim)
            simulator.setPosition(legNo, angles);
        else
            board.setPosition(legNo, angles);
            */
        break;
    }
    case HandlerRPCMotion::ROB_SET_POSITION_ROBOT:
    {
        std::uint32_t jointsNo; std::vector<double> angles;
        walkers::StreamRead(stream, jointsNo);
        angles.resize(jointsNo);
        for (size_t i=0;i<jointsNo;i++){
            walkers::StreamRead(stream, angles[i]);
        }/*
        if (useSim)
            simulator.setPosition(angles);
        else
            board.setPosition(angles);
            */
        break;
    }
    case HandlerRPCMotion::ROB_GET_POSITION_JOINT:
    {
        std::uint32_t legNo; std::uint32_t jointNo; double angle;
        walkers::StreamRead(stream, legNo);
        walkers::StreamRead(stream, jointNo);
        if (useSim)
            simulator.readPosition(legNo, jointNo, angle);
        else
            board.readPosition(legNo, jointNo, angle);
        walkers::StreamWrite(stream, angle);
        break;
    }
    case HandlerRPCMotion::ROB_GET_POSITION_LEG:
    {
        std::uint32_t legNo; std::vector<double> angles;
        walkers::StreamRead(stream, legNo);
        if (useSim)
            simulator.readPosition(legNo, angles);
        else
            board.readPosition(legNo, angles);
        walkers::StreamWrite(stream, (std::uint32_t)angles.size());
        for (const auto & angle : angles)
            walkers::StreamWrite(stream, angle);
        break;
    }
    case HandlerRPCMotion::ROB_GET_POSITION_ROBOT:
    {
        std::vector<double> angles;
        if (useSim)
            simulator.readPosition(angles);
        else
            board.readPosition(angles);
        walkers::StreamWrite(stream, (std::uint32_t)angles.size());
        for (const auto & angle : angles)
            walkers::StreamWrite(stream, angle);
        break;
    }
    case HandlerRPCMotion::ROB_SET_SPEED_JOINT:
    {
        std::uint32_t legNo; std::uint32_t jointNo; double speed;
        walkers::StreamRead(stream, legNo);
        walkers::StreamRead(stream, jointNo);
        walkers::StreamRead(stream, speed);
        if (useSim)
            simulator.setSpeed(legNo, jointNo, speed);
        else
            board.setSpeed(legNo, jointNo, speed);
        break;
    }
    case HandlerRPCMotion::ROB_SET_SPEED_LEG:
    {
        std::uint32_t legNo; std::uint32_t jointsNo; std::vector<double> speeds;
        walkers::StreamRead(stream, legNo);
        walkers::StreamRead(stream, jointsNo);
        speeds.resize(jointsNo);
        for (size_t i=0;i<jointsNo;i++)
            walkers::StreamRead(stream, speeds[i]);
        if (useSim)
            simulator.setSpeed(legNo, speeds);
        else
            board.setSpeed(legNo, speeds);
        break;
    }
    case HandlerRPCMotion::ROB_SET_SPEED_ROBOT:
    {
        std::uint32_t jointsNo; std::vector<double> speeds;
        walkers::StreamRead(stream, jointsNo);
        speeds.resize(jointsNo);
        for (size_t i=0;i<jointsNo;i++){
            walkers::StreamRead(stream, speeds[i]);
        }
        if (useSim)
            simulator.setSpeed(speeds);
        else
            board.setSpeed(speeds);
        break;
    }
    case HandlerRPCMotion::ROB_GET_LEG_CONTACT:
    {
        std::uint32_t legNo;
        walkers::StreamRead(stream, legNo);
        bool contact;
        if (useSim)
            contact = simulator.readContact(legNo);
        else
            contact = board.readContact(legNo);
        walkers::StreamWrite(stream, contact);
        break;
    }
    case HandlerRPCMotion::ROB_GET_CONTACTS:
    {
        std::vector<bool> contacts;
        if (useSim)
            simulator.readContacts(contacts);
        else
            board.readContacts(contacts);
        walkers::StreamWrite(stream, (std::uint32_t)contacts.size());
        for (size_t legNo=0;legNo<contacts.size();legNo++)
            walkers::StreamWrite(stream, contacts[legNo]);
        break;
    }
    case HandlerRPCMotion::ROB_MOVE_PLATFORM_NEUTRAL:
    {
        walkers::Mat34 motion;
        double speed;
        walkers::StreamRead(stream, motion);
        walkers::StreamRead(stream, speed);
        controller.movePlatformNeutral(motion, speed);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_MOVE_PLATFORM:
    {
        walkers::Mat34 motion;
        double speed;
        walkers::StreamRead(stream, motion);
        walkers::StreamRead(stream, speed);
        controller.movePlatform(motion, speed);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_MOVE_LEG_SINGLE:
    {
        walkers::Mat34 motion;
        double speed;
        std::uint32_t legNo, lastMove, smartMotionMode, inputCoordinateSystem;
        walkers::StreamRead(stream, legNo);
        walkers::StreamRead(stream, motion);
        walkers::StreamRead(stream, speed);
        walkers::StreamRead(stream, lastMove);
        walkers::StreamRead(stream, smartMotionMode);
        walkers::StreamRead(stream, inputCoordinateSystem);
        controller.moveLegSingle(legNo,motion,speed,(bool)lastMove,(int)smartMotionMode,(walkers::RobotController::MotionCoordinate)inputCoordinateSystem);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_TRIPOD_STEP:
    {
        walkers::Mat34 motion;
        double speed, offsetUp;
        walkers::StreamRead(stream, motion);
        walkers::StreamRead(stream, speed);
        walkers::StreamRead(stream, offsetUp);
        controller.tripodStep(motion, speed, offsetUp);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_WAVEGAIT_STEP:
    {
        walkers::Mat34 motion;
        double speed, offsetUp;
        walkers::StreamRead(stream, motion);
        walkers::StreamRead(stream, speed);
        walkers::StreamRead(stream, offsetUp);
        controller.waveGait(motion, speed, offsetUp);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_TRIPOD_ADAPTIVE_STEP:
    {
        walkers::Mat34 motion;
        double speed, offsetUp;
        walkers::StreamRead(stream, motion);
        walkers::StreamRead(stream, speed);
        walkers::StreamRead(stream, offsetUp);
        std::uint32_t adaptType;
        walkers::StreamRead(stream, adaptType);
        controller.tripodAdaptiveStep(motion, speed, offsetUp, (walkers::RobotController::AdaptationType)adaptType);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_START_LOG_THREAD:
    {
        controller.startLogThread();
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_PLACE_FOOT:
    {
        std::uint32_t legNo;
        double deltaZ, speed;
        walkers::StreamRead(stream, legNo);
        walkers::StreamRead(stream, deltaZ);
        walkers::StreamRead(stream, speed);
        controller.placeFoot(legNo, deltaZ, speed);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_PLACE_FEET:
    {
        double deltaZ, speed;
        walkers::StreamRead(stream, deltaZ);
        walkers::StreamRead(stream, speed);
        controller.placeFeet(deltaZ, speed);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
    case HandlerRPCMotion::ROB_GET_TIME:
    {
        double currTime = controller.getGlobalTime();
        walkers::StreamWrite(stream, currTime);
        break;
    }
    case HandlerRPCMotion::ROB_EXECUTE_PATH:
    {
        std::uint32_t pathSize; std::uint32_t legsNo;
        std::vector<walkers::Mat34> bodyPath;
        std::vector<planner::Foot::Seq> feetPaths;
        walkers::StreamRead(stream, pathSize);
        for (size_t i=0;i<pathSize;i++){
            walkers::Mat34 pose;
            walkers::StreamRead(stream, pose);
            bodyPath.push_back(pose);
        }
        walkers::StreamRead(stream, legsNo);
        feetPaths.resize(legsNo);
        for (size_t i=0;i<legsNo;i++){
            walkers::StreamRead(stream, pathSize);
            feetPaths[i].resize(pathSize);
            for (size_t j=0;j<pathSize;j++){
                planner::Foot foot;
                walkers::StreamRead(stream, foot);
                feetPaths[i][j]=foot;
            }
        }
        double speed;
        walkers::StreamRead(stream, speed);
        controller.executePath(bodyPath, feetPaths, speed);
        walkers::StreamWrite(stream, std::uint32_t(1));
        break;
    }
#ifdef BUILD_WALKERS_BOARD_GALGO
    case HandlerRPCMotion::ROB_DEFAULT_GALGO:
    {
        walkers::RobotController* rc = &controller;
        ((ControllerGalgo*)rc)->setStartPosition();
        break;
    }
    case HandlerRPCMotion::ROB_STEP_GALGO:
    {
        walkers::RobotController::StepParams params;
        walkers::StreamRead(stream, params);
        walkers::RobotController* rc = &controller;
        bool result = ((ControllerGalgo*)rc)->stepGalgo(params);
        walkers::StreamWrite(stream, result);
        break;
    }
#endif
    default:
        throw std::runtime_error("Unknown request: " + std::to_string(cmd));
    }
}

void waitForHandler(walkers::Stream::Ptr stream, simulator::Simulator* simulator, controller::Board* board, walkers::RobotController* controller, bool useSim) {
    try {
        // handle requests
        for (;;) {
            requestHandler(*stream, *simulator, *board, *controller, useSim);
        }
    }
    catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
    }
}

/// server thread
void serverRunning(unsigned short int port, simulator::Simulator* sim, controller::Board* board, walkers::RobotController* controller, bool useSim){
    // run server
    walkers::Server server(port);
    std::vector<std::thread> handlerThrs;
    for (;;) {
        // wait for incomming connection
        std::cout << "Listening on port: " << port << "\n";
        walkers::Stream::Ptr stream = server.accept<walkers::StreamSocket>();
        // debug info
        const boost::asio::ip::tcp::socket& socket = static_cast<const walkers::StreamSocket&>(*stream).getSocket();
        std::cout << "Accepted connection from host: " << socket.remote_endpoint().address().to_string() << ", port: " << socket.remote_endpoint().port() << "\n";
        std::thread th(waitForHandler, stream, sim, board, controller, useSim);
        handlerThrs.push_back(std::move(th));
    }
    for (auto& th : handlerThrs)
        th.join();
}

//------------------------------------------------------------------------------
void HandlerRPCMotion::Config::load(std::string configFilename){
    setlocale(LC_NUMERIC,"C");
    tinyxml2::XMLDocument configSrv;
    configSrv.LoadFile(std::string("../../resources/" + configFilename).c_str());
    if (configSrv.ErrorID())
        std::cout << "unable to load server config file.\n";
    configSrv.FirstChildElement("Server")->FirstChildElement("parameters")->QueryUnsignedAttribute("port",&port);
    configSrv.FirstChildElement("Server")->FirstChildElement("parameters")->QueryBoolAttribute("useSimulation",&useSimulation);
    robotName = configSrv.FirstChildElement("Server")->FirstChildElement("robot")->Attribute("name");
    std::cout << "robot " << robotName << "\n";
}

int main(void) {
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

        std::string serverConfig(rootXML->FirstChildElement( "ServerSimulator" )->FirstChildElement("config")->GetText());
        //std::string serverType(rootXML->FirstChildElement( "ServerSimulator" )->FirstChildElement("type")->GetText());

        std::string robotConfig(rootXML->FirstChildElement( "Robot" )->FirstChildElement("config")->GetText());
        //std::string robotType(rootXML->FirstChildElement( "Robot" )->FirstChildElement("type")->GetText());

        std::string motionCtrlConfig(rootXML->FirstChildElement( "MotionController" )->FirstChildElement("config")->GetText());
        //std::string motionCtrlType(rootXML->FirstChildElement( "MotionController" )->FirstChildElement("type")->GetText());

        HandlerRPCMotion::Config configServer;
        configServer.load(serverConfig);

        //QApplication application(argc,argv);
        //setlocale(LC_NUMERIC,"C");
        //glutInit(&argc, argv);

        simulator::Simulator* sim;
#ifdef BUILD_WITH_SIMULATOR
        if (simType.compare("physx") == 0) {
            sim = simulator::createPhysXSimulator(simConfig,robotConfig, sceneConfig);
        }
        else if (simType.compare("ode") == 0) {
            sim = simulator::createODESimulator(simConfig);
        }
        else {
            std::cout << "XML Error: Wrong simulation type.\n";
            return 0;
        }
#endif

        controller::Board* board = nullptr;
        /// Robot controller
        walkers::RobotController* robotController = nullptr;
        if (configServer.useSimulation) {
            robotController = walkers::createControllerMessor2(motionCtrlConfig, robotConfig, (controller::Board*)sim);
        }
        else {
            if (configServer.robotName=="Messor"){
                robotController = walkers::createControllerMessor2(motionCtrlConfig, robotConfig, board);
                board = controller::createBoardDynamixel();
                ((ControllerMessor2*) robotController)->updateDevice(board);
            }
            else if (configServer.robotName=="Galgo"){
#ifdef BUILD_WALKERS_BOARD_GALGO
                robotController = walkers::createControllerGalgo(motionCtrlConfig, robotConfig, board);
                board = controller::createBoardGalgo("boardGalgo.xml");
                ((ControllerGalgo*) robotController)->updateDevice(board);
#else
                throw std::runtime_error("Server config specifies used robot as Galgo, but current build doesn't include Galgo (BUILD_WALKERS_BOARD_GALGO is OFF)");
#endif
            }
        }

        //std::thread serverThr(serverRunning, (unsigned short int) configServer.port, sim, board, robotController, configServer.useSimulation);
        std::cout << "try port " << configServer.port << "\n";
        serverRunning((unsigned short int) configServer.port, sim, board, robotController, configServer.useSimulation);
        //application.exec();
        sim->stopSimulation();
        //serverThr.join();
    }
    catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
        return 1;
    }

    return 0;
}
