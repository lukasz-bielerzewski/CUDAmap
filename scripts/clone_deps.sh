#!/bin/bash

#
printf "Install PhysX manually (Ask for permissions from NVidia).\n Clone latest release of version 3.4 to ~/Libs/PhysX-3.4.\n"

read -p "The following libraries are required. The script will download them and compile: OpenCV, OpenNI, SensorKinect, NITE, coldet, libccd, FCL, libftdi2, PCL
The following system dependencies are required: libxxf86vm-dev libode-dev freeglut3-dev qtdeclarative5-dev openjdk-8-jdk pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev doxygen graphviz mono-complete git cmake libeigen3-dev libode-dev libopenni2-dev libboost-dev libboost-thread-dev libeigen3-dev m4 libboost1.71-all-dev libxxf86vm-dev liblapack-dev libblas-dev libboost-dev libarmadillo-dev libgmp-dev libopenmpi-dev libvtk7-dev libhdf5-mpi-dev libqglviewer2-qt5 libqglviewer-dev-qt5 libopenexr-dev
Do you want to install them [y/n]? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo apt-get update -y
    sudo apt-get install -y libxxf86vm-dev libtinyxml2-dev libflann-dev libode-dev freeglut3-dev qtdeclarative5-dev openjdk-8-jdk pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev doxygen graphviz mono-complete git cmake libeigen3-dev libode-dev libopenni2-dev libboost-dev libboost-thread-dev libeigen3-dev m4 libxxf86vm-dev libassimp-dev libgmp-dev libopenmpi-dev libvtk7-dev libhdf5-mpi-dev libqglviewer2-qt5 libqglviewer-dev-qt5 libopenexr-dev libdc1394-dev libgstreamer1.0-dev python-dev-is-python3 python2-dev clang
fi

read -p "Would you like to set your numeric locale to en_US.UTF-8? [y/n]? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "export LC_NUMERIC=en_US.UTF-8" >> $HOME/.bashrc
fi

#create symlinks
sudo ln -s /usr/lib/x86_64-linux-gnu/libpthread.so.0 /usr/lib/x86_64-linux-gnu/libpthread.so
sudo ln -s /usr/lib/x86_64-linux-gnu/libdl.so.2 /usr/lib/x86_64-linux-gnu/libdl.so

source_dir=$(pwd)

# download and compile OpenCV
mkdir ~/Libs
cd ~/Libs
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x
cd ..
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.x
cd ../opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_GTK_2_X=ON \
      -D WITH_OPENGL=ON \
      -D WITH_OPENNI2=OFF \
      -D WITH_1394=ON \
      -D WITH_GSTREAMER=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON ..
make -j4
sudo make -j4 install

#install PCL
cd ~/Libs
git clone https://github.com/PointCloudLibrary/pcl
cd pcl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_QHULL=OFF ..
make -j4
sudo make -j4 install

# compile visualizer
cd ~/Sources
mkdir build-visualizer-Desktop-Default
cd build-visualizer-Desktop-Default
cmake ../visualizer
make -j4
cd ~/Sources/visualizer/build/bin

printf "Done!\n"
printf "Go to ~/Sources/visualizer/build/bin and run examples (./demoVisualizer)\n"
