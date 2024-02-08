#include "include/Defs/defs.h"
#include "include/Simulator/simulatorODE.h"
#include "include/Simulator/simulatorPhysX.h"
#include <tinyxml2.h>
#include <iostream>

int main()
{
    try
    {
        using namespace simulator;
        using namespace std;

        setlocale(LC_NUMERIC,"C");
        tinyxml2::XMLDocument config;
        config.LoadFile("../../resources/configGlobal.xml");
        if (config.ErrorID())
            std::cout << "unable to load global config file.\n";
        auto rootXML = config.FirstChildElement( "configGlobal" );
        std::string simConfig = rootXML->FirstChildElement( "Environment" )->Attribute("config");
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
