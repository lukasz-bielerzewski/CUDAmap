# cmake requirements
cmake_minimum_required(VERSION 3.10)

# Build options have to be before PROJECT(...)
SET(CMAKE_CONFIGURATION_TYPES "Release" CACHE PATH "Configuration types")
SET(CMAKE_BUILD_TYPE "Release" CACHE PATH "Current build configuration")

# Example Project configuration
PROJECT(VISUALIZER)

# CUDA flags
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(VISUALIZER LANGUAGES CXX CUDA)
enable_language(CUDA)

SET(CXX_STANDARD "-std=c++17")
SET(CMAKE_CXX_FLAGS "${CXX_STANDARD} -Wall -Wextra -Wshadow -Wconversion -pthread -fopenmp")

# Folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# Settings
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/resources")
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include")
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include/Defs")
option(LOG_DEBUG "Turn on debugging log" OFF)
mark_as_advanced(LOG_DEBUG)
if(LOG_DEBUG)
    add_definitions(-DLOG_DEBUG)
endif(LOG_DEBUG)

# core libraries
option(BUILD_DEFS "Build SIMULATOR Defs library" YES)
mark_as_advanced(BUILD_DEFS)
option(BUILD_MAPPING "Build Mapping" ON)
mark_as_advanced(BUILD_MAPPING)
option(BUILD_VISUALIZER "Build Visualizer" ON)
mark_as_advanced(BUILD_VISUALIZER)
option(BUILD_VISUALIZER_UTILITIES "Build VISUALIZER Utilities library" ON)
mark_as_advanced(BUILD_VISUALIZER_UTILITIES)
option(CV_BUILD_NONFREE "openCV build nonfree" OFF)
mark_as_advanced(CV_BUILD_NONFREE)
option(BUILD_GRABBER "Build grabber" OFF)
mark_as_advanced(BUILD_GRABBER)
option(BUILD_IMAGE_PROCESSING "Build image processing module" ON)
mark_as_advanced(BUILD_IMAGE_PROCESSING)
option(BUILD_PLAYGROUNDS "Build simple playgrounds" ON)
mark_as_advanced(BUILD_PLAYGROUNDS)

# demos
option(BUILD_PCD2PNG_DEMO "Build pcd 2 png converter" OFF)
mark_as_advanced(BUILD_PCD2PNG_DEMO)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

if(BUILD_VISUALIZER)
    add_definitions(-DBUILD_VISUALIZER)
endif(BUILD_VISUALIZER)

if(BUILD_SIMULATOR)
    add_definitions(-DBUILD_WITH_SIMULATOR)
endif(BUILD_SIMULATOR)

if(BUILD_GRABBER)
    add_definitions(-DBUILD_WITH_GRABBER)
endif(BUILD_GRABBER)

if(CV_BUILD_NONFREE)
    add_definitions(-DCV_BUILD_NONFREE)
endif(CV_BUILD_NONFREE)

#additional dependencies
# Boost
FIND_PACKAGE(Boost COMPONENTS system filesystem thread asio)

#Eigen
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/3rdParty/Eigen)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_definitions(-Wno-deprecated -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

#QGLViewer library
FIND_PACKAGE(OpenGL)
include_directories(${OPENGL_INCLUDE_DIR})
MESSAGE("opengl: " ${OPENGL_LIBRARY})
link_directories(${OPENGL_LIBRARY})

#Qt5
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Xml REQUIRED)
find_package(Qt5OpenGL REQUIRED)
MESSAGE("widgets: " ${Qt5Core_LIBRARIES})
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Xml_INCLUDE_DIRS})
include_directories(${Qt5OpenGL_INCLUDE_DIRS})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # CMake doesn't add "-pie" by default for executables (CMake issue #14983)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIC")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
SET(CMAKE_CACHEFILE_DIR_FLAGS "${CMAKE_C_FLAGS} -fPIC")

#QGLViewer library
#FIND_PACKAGE(QGLViewer)
#include_directories(${QGLVIEWER_INCLUDE_DIR})
#link_directories(${QGLVIEWER_LIBRARY})

#NatNet (3rdParty)
INCLUDE_DIRECTORIES ( "${CMAKE_SOURCE_DIR}/3rdParty" )

#dynamixel3
INCLUDE_DIRECTORIES ( "${CMAKE_SOURCE_DIR}/3rdParty/dynamixel3/include" )

if(BUILD_GRABBER OR BUILD_IMAGE_PROCESSING)
    # OpenCV - local version compiled from source
    #set(OpenCV_DIR ~/Libs/opencv/build CACHE PATH "OpenCV dir")
    FIND_PACKAGE(OpenCV REQUIRED)
    include_directories( ${OpenCV_INCLUDE_DIRS} )
#    set(OPENCV_INCLUDE ~/Libs/opencv/include CACHE PATH "Path prefix for OpenCV include")
#    set(OPENCV_LIBRARY ~/Libs/opencv CACHE PATH "Path prefix for OpenCV library")
#    include_directories(SYSTEM ${OPENCV_INCLUDE})
#    link_directories(${OPECV_LIBRARY}/Lib/linux64)
endif(BUILD_GRABBER OR BUILD_IMAGE_PROCESSING)

# PCL
set(PCL_DIR "/path/to/pcl-1.13/share/pcl-1.13")
find_package(PCL 1.13 REQUIRED COMPONENTS common gpu_containers io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Executable output directory
SET(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Executable output directory")
mark_as_advanced(RUNTIME_OUTPUT_DIRECTORY)

# Dynamic library output directory
SET(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Dynamic library output directory")
mark_as_advanced(LIBRARY_OUTPUT_DIRECTORY)

# Static library output directory
SET(ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Static library output directory")
mark_as_advanced(ARCHIVE_OUTPUT_DIRECTORY)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_DIRECTORY})
foreach(CONFIGURATION_TYPE ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIGURATION_TYPE} CONFIGURATION_TYPE)
        SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${RUNTIME_OUTPUT_DIRECTORY})
        SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${LIBRARY_OUTPUT_DIRECTORY})
        SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${ARCHIVE_OUTPUT_DIRECTORY})
endforeach(CONFIGURATION_TYPE CMAKE_CONFIGURATION_TYPES)

if(MSVC)
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR CMake)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

# set up include-directories
#include_directories(
#  "${CMAKE_SOURCE_DIR}"   # to find foo/foo.h
#  "${CMAKE_SOURCE_DIR}/include")  # to find foo/config.h

# Create the VisualizerConfig.cmake
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
file(RELATIVE_PATH REL_LIB_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_LIB_DIR}")

set(CONF_INCLUDE_DIRS "${RELATIVE_PATH}" "${CMAKE_SOURCE_DIR}")

set(CONF_LIB_DIRS "${RELATIVE_PATH}" "/build/lib")

configure_file(VisualizerConfig.cmake.in
  "${CMAKE_SOURCE_DIR}/VisualizerConfig.cmake" @ONLY)

# ... for the install tree
set(CONF_INCLUDE_DIRS "${CONF_INCLUDE_DIRS}/include")
set(CONF_LIB_DIRS "${CMAKE_SOURCE_DIR}/build/lib")

#MESSAGE ("This is message.")
#MESSAGE (${INSTALL_CMAKE_DIR})
#MESSAGE (${CONF_INCLUDE_DIRS})
#MESSAGE (${CONF_LIB_DIRS})

configure_file(VisualizerConfig.cmake.in
  "${CMAKE_SOURCE_DIR}/VisualizerConfig.cmake" @ONLY)

# Install the VisualizerConfig.cmake
install(FILES
  "${CMAKE_SOURCE_DIR}/VisualizerConfig.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

###############################################################################
#
# External Octree library
#
###############################################################################

include_directories(${CMAKE_SOURCE_DIR}/include/octree)
file(GLOB OCTREE_HEADERS
    "${CMAKE_SOURCE_DIR}/include/octree/*.h"
    "${CMAKE_SOURCE_DIR}/include/octree/*.tcc"
)
INSTALL(FILES ${OCTREE_HEADERS} DESTINATION include/octree)


###############################################################################
#
# SIMULATOR defs library
#
###############################################################################

if(BUILD_DEFS)
        file(GLOB DEFS_SOURCES
            "${CMAKE_SOURCE_DIR}/src/Defs/*.cpp"
        )
        file(GLOB DEFS_HEADERS
            "${CMAKE_SOURCE_DIR}/include/Defs/*.h"
        )
        if(NOT BUILD_LOCALIZATION_G2O)
            list(REMOVE_ITEM DEFS_HEADERS ${CMAKE_SOURCE_DIR}/include/Defs/g2o.h)
        endif(NOT BUILD_LOCALIZATION_G2O)
        ADD_LIBRARY(Defs STATIC ${DEFS_SOURCES} ${DEFS_HEADERS})
#        set_target_properties(Defs PROPERTIES LINKER_LANGUAGE CXX)
# 	INSTALL(TARGETS visualizerDefs RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
        INSTALL(FILES ${DEFS_HEADERS} DESTINATION include/Defs/)

endif(BUILD_DEFS)


###############################################################################
#
# Regression library
#
###############################################################################

if(BUILD_REGRESSION)
        file(GLOB REGRESSION_SOURCES
            "${CMAKE_SOURCE_DIR}/src/Regression/*.cpp"
        )
        file(GLOB REGRESSION_HEADERS
            "${CMAKE_SOURCE_DIR}/include/Regression/*.h"
        )
        ADD_LIBRARY(Regression STATIC ${REGRESSION_SOURCES} ${REGRESSION_HEADERS})
        TARGET_LINK_LIBRARIES(Regression Optimizer Utilities tinyxml2 Defs)
#        INSTALL(TARGETS Utilities RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
#        INSTALL(FILES ${VISUALIZER_HEADERS} DESTINATION include/Visualizer/Visualizer/)
endif(BUILD_REGRESSION)

###############################################################################
#
# MAPPING library
#
###############################################################################

if(BUILD_MAPPING)
        file(GLOB MAPPING_SOURCES
            "${CMAKE_SOURCE_DIR}/src/Mapping/*.cpp"
        )
        file(GLOB MAPPING_HEADERS
            "${CMAKE_SOURCE_DIR}/include/Mapping/*.h"
        )
        ADD_LIBRARY(Mapping STATIC ${MAPPING_SOURCES} ${MAPPING_HEADERS})
        TARGET_LINK_LIBRARIES(Mapping Utilities Defs tinyxml2)
#        INSTALL(TARGETS Utilities RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
#        INSTALL(FILES ${VISUALIZER_HEADERS} DESTINATION include/visualizer/Visualizer/)
endif(BUILD_MAPPING)

###############################################################################
#
# SIMULATOR Visualizer library
#
###############################################################################

if(BUILD_VISUALIZER)
        file(GLOB VISUALIZER_SOURCES
            "${CMAKE_SOURCE_DIR}/src/Visualizer/*.cpp"
        )
        file(GLOB VISUALIZER_HEADERS
            "${CMAKE_SOURCE_DIR}/include/Visualizer/*.h"
        )
        ADD_LIBRARY(Visualizer STATIC ${VISUALIZER_SOURCES} ${VISUALIZER_HEADERS})
        TARGET_LINK_LIBRARIES(Visualizer ${OpenCV_LIBS} tinyxml2 Defs GL GLU glut QGLViewer-qt5 Qt5Core Qt5Widgets Qt5Xml Qt5OpenGL Qt5Gui Mapping ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})
#        INSTALL(TARGETS Utilities RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
#        INSTALL(FILES ${VISUALIZER_HEADERS} DESTINATION include/visualizer/Visualizer/)
endif(BUILD_VISUALIZER)


###############################################################################
#
# VISUALIZER Utilities library
#
###############################################################################

if(BUILD_VISUALIZER_UTILITIES)
        file(GLOB UTILITIES_SOURCES
            "${CMAKE_SOURCE_DIR}/src/Utilities/*.cpp"
        )
        file(GLOB UTILITIES_HEADERS
            "${CMAKE_SOURCE_DIR}/include/Utilities/*.h"
        )
        add_library(Utilities STATIC ${UTILITIES_SOURCES} ${UTILITIES_HEADERS}
            include/Utilities/TimeRange.h)
        TARGET_LINK_LIBRARIES(Utilities Defs ${ASSIMP_LIBRARY})
        if(BUILD_VISUALIZER)
            TARGET_LINK_LIBRARIES(Utilities GL GLU glut assimp)
        else(BUILD_VISUALIZER)
            TARGET_LINK_LIBRARIES(Utilities assimp)
        endif(BUILD_VISUALIZER)
        TARGET_LINK_LIBRARIES(Utilities assimp)
        INSTALL(TARGETS Utilities RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
        INSTALL(FILES ${TRACKER_HEADERS} DESTINATION include/visualizer/Utilities/)
endif(BUILD_VISUALIZER_UTILITIES)

###############################################################################
#
# VISUALIZER Grabber library
#
###############################################################################

if(BUILD_GRABBER)
        file(GLOB GRABBER_SOURCES
            "${CMAKE_SOURCE_DIR}/src/Grabber/*.cpp"
        )
        file(GLOB GRABBER_HEADERS
            "${CMAKE_SOURCE_DIR}/include/Grabber/*.h"
        )
        ADD_LIBRARY(GrabberDev STATIC ${GRABBER_SOURCES} ${GRABBER_HEADERS})
        TARGET_LINK_LIBRARIES(GrabberDev ${OpenCV_LIBS} tinyxml2 boost_system)

        INSTALL(TARGETS GrabberDev RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
        INSTALL(FILES ${GRABBER_HEADERS} DESTINATION include/Grabber/Grabber/)
endif(BUILD_GRABBER)

###############################################################################
#
# Image processing library
#
###############################################################################

if(BUILD_IMAGE_PROCESSING)
        file(GLOB IMAGE_PROCESSING_SOURCES
            "${CMAKE_SOURCE_DIR}/src/ImageProcessing/*.cpp"
        )
        file(GLOB IMAGE_PROCESSING_HEADERS
            "${CMAKE_SOURCE_DIR}/include/ImageProcessing/*.h"
        )
        ADD_LIBRARY(ImageProcessing STATIC ${IMAGE_PROCESSING_SOURCES} ${IMAGE_PROCESSING_HEADERS})
        TARGET_LINK_LIBRARIES(ImageProcessing ${OpenCV_LIBS} Utilities tinyxml2 Defs)
        if (BUILD_GRABBER)
            TARGET_LINK_LIBRARIES(ImageProcessing ${OpenCV_LIBS} GrabberDev Utilities tinyxml2 Defs)
        endif (BUILD_GRABBER)
#        INSTALL(TARGETS Utilities RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
#        INSTALL(FILES ${VISUALIZER_HEADERS} DESTINATION include/visualizer/Visualizer/)
endif(BUILD_IMAGE_PROCESSING)

###############################################################################
#
# SIMULATOR demo
#
###############################################################################

if(BUILD_SIMULATOR AND BUILD_VISUALIZER AND BUILD_SIMULATOR_DEMO AND BUILD_SIMULATOR_DEMO_MAIN)
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath ./")
        SET(DEMO_SOURCES ./main.cpp)
        ADD_EXECUTABLE(simulator ${DEMO_SOURCES})
        TARGET_LINK_LIBRARIES(simulator tinyxml2 Simulator ode glut)
        INSTALL(TARGETS simulator RUNTIME DESTINATION bin)

endif(BUILD_SIMULATOR AND BUILD_VISUALIZER AND BUILD_SIMULATOR_DEMO AND BUILD_SIMULATOR_DEMO_MAIN)

###############################################################################
#
# SIMULATOR demo Visualizer
#
###############################################################################

if(BUILD_VISUALIZER)
        SET(DEMO_SOURCES ./demos/demoVisu.cpp)
        ADD_EXECUTABLE(demoVisu ${DEMO_SOURCES})
        target_compile_options(demoVisu PRIVATE -fPIC)
        TARGET_LINK_LIBRARIES(demoVisu Defs Mapping Visualizer  tinyxml2 glut)
        INSTALL(TARGETS demoVisu RUNTIME DESTINATION bin)

        SET(DEMO_SOURCES ./demos/demoMap.cpp ./demos/CUDAmap.h)
        ADD_EXECUTABLE(demoMap ${DEMO_SOURCES}
            demos/CUDA_Depth2Cloud.cu
            demos/CUDA_TransformPoints.cu
            demos/CUDA_UpdateNDTOM.cu
            demos/CUDAmap.cu)
        target_compile_options(demoMap PRIVATE -fPIC)
        TARGET_LINK_LIBRARIES(demoMap Defs Mapping Visualizer  tinyxml2 glut ${PCL_LIBRARIES})
        INSTALL(TARGETS demoMap RUNTIME DESTINATION bin)

        SET(DEMO_SOURCES ./demos/showCloud.cpp)
        ADD_EXECUTABLE(showCloud ${DEMO_SOURCES})
        target_compile_options(showCloud PRIVATE -fPIC)
        TARGET_LINK_LIBRARIES(showCloud Defs Mapping Visualizer tinyxml2 glut)
        INSTALL(TARGETS showCloud RUNTIME DESTINATION bin)

endif(BUILD_VISUALIZER)


###############################################################################
#
# build 3D maps from files
#
###############################################################################

if(BUILD_MAPPING AND BUILD_VISUALIZER)

        ADD_EXECUTABLE(load3Dmap ./demos/load3Dmap.cpp)
        target_compile_options(load3Dmap PRIVATE -fPIC)
        TARGET_LINK_LIBRARIES(load3Dmap Visualizer Mapping ImageProcessing Utilities Defs tinyxml2 ${OpenCV_LIBS} glut boost_system)
        INSTALL(TARGETS load3Dmap RUNTIME DESTINATION bin)

        ADD_EXECUTABLE(modelFromRGBD ./demos/modelFromRGBD.cpp)
        target_compile_options(modelFromRGBD PRIVATE -fPIC)
        TARGET_LINK_LIBRARIES(modelFromRGBD stdc++fs Visualizer Mapping ImageProcessing Utilities Defs tinyxml2 ${OpenCV_LIBS} glut boost_system)

endif(BUILD_MAPPING AND BUILD_VISUALIZER)


###############################################################################
#
# build pcd2png converter
#
###############################################################################

if(BUILD_VISUALIZER)
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath ./")
        SET(DEMO_SOURCES ./demos/pcd2png.cpp)
        ADD_EXECUTABLE(pcd2png ${DEMO_SOURCES})
        TARGET_LINK_LIBRARIES(pcd2png ImageProcessing Utilities Defs tinyxml2 ${PCL_IO_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
        INSTALL(TARGETS pcd2png RUNTIME DESTINATION bin)

        SET(DEMO_SOURCES ./demos/imgs2cloud.cpp)
        ADD_EXECUTABLE(imgs2cloud ${DEMO_SOURCES})
        target_compile_options(imgs2cloud PRIVATE -fPIC)
        TARGET_LINK_LIBRARIES(imgs2cloud Visualizer ImageProcessing Utilities Defs tinyxml2 ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OpenCV_LIBS} glut)

        INSTALL(TARGETS imgs2cloud RUNTIME DESTINATION bin)
endif(BUILD_VISUALIZER)

