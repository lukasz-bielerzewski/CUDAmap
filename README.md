**Author/Maintenance**:
Dominik Belter

# Visualizer installation guide
Installation instruction was tested on Ubuntu 20.04 operating system

## Installer

The clone_deps.sh script installs all required software from repositories, clones requires libraries and compiles them:

     - mkdir ~/Sources
     - cd ~/Sources
     - git clone https://github.com/LRMPUT/walkers
     - mkdir ~/Libs

### 1. Docker in VSCode:
    1. Install Docker and VS Code
    2. Install VS Code Extensions: Dev Containers and Remote Development extensions
    3. Allow non-network local connections to display
	$ xhost +local:docker
    4 Open this folder in VS Code:
        $ code .
    5. Click the Reopen in Container button:

### 2. Go to ~/Sources/walkers/scripts and open clonde_deps.sh:
There is two ways of install walkers and libraries:
- Just use ./clone_dephs.sh in terminal. It is the fastest way, but there is also risk that script won't install some dependencies.
  
  > [HINT] Give execute permission to ./clone_dephs.sh and then run the script:
      $ chmod +x ./clone_dephs.sh

- Install all libraries just by inserting commands from script to Terminal

### 3. Go to ~/Sources/walkers/scripts and open clonde_deps.sh:
Go to Visualizer-main/resources and create directory named "foto"
inside of foto create directiry that will be named after the room of which scan will be used.
Inside room directory unpack downloaded rgb-d images with camera trajectory data.
example:
Visualizer-main/resources/foto/salon/rgb
Visualizer-main/resources/foto/salon/depth
Visualizer-main/resources/foto/salon/trajectory.txt

Be sure to make according path changes in demoMap.cpp. 

### 4. Sample demo:
If you want you can run sample demo in path `build/bin/demoVisu`

**Contributors**:
