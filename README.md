
## SlicerSOFA and Slicer ROS2 integration

 This repository contains 2 projects that demonstrate the pertinence and feasibility of the integration of 2 Slicer 3D extensions. The base of these projects is a script developped by Steve Pieper that can be found [here](https://github.com/pieper/SlicerSOFA/blob/4e226fc96627648e158538b3983fcab98bac59cf/Experiments/liver-torus-probe/scene.py#L4).

- The first extension is the Slicer SOFA extension, developped by Rafael Palomar that enables the user to visualize object ( here, organs) deform according to physical constraints (here, the tip of a robot).
- The second one is the Slicer ROS2 extension, developped by Laura Connoly,Aravind S. Kumar and Anton Deguet.

To use these projects, you will need to build the Slicer ROS2 extension and download the Phantom Omni driver ( you can also use a pretend version). Thus, you need to build Slicer 3D from source. I used ROS2 Jazzy version. The whole setup has been tested and works fine with Ubuntu 24.04. It will not work on a virtual machine using a real Phantom Omni.

Here are some important steps to be able to use it all :

- When you build from source Slicer 3D, you should NOT download QT from their website but rather use the built in.
- That is the [repository](https://github.com/Slicer/SlicerSOFA.git) for the Slicer SOFA extension.
- Follow this link to install ROS2 Jazzy : [ROS2 Jazzy extension](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) .
- Follow this link to install the Slicer ROS2 extension : [Slicer ROS2 installation doc](https://slicer-ros2.readthedocs.io/en/devel/pages/getting-started.html) .
- Follow this link to get the Phantom Omni link to Slicer ROS2 : [SlicerROS2 Phantom Omni link](https://github.com/jhu-saw/ros2_sensable_omni_model). Please note that if you wish to use the real robot you must not use

 ```sh
ros2 run sensable_omni_model pretend_omni_joint_state_publisher
```

### Steps to follow if you want to use the real Phantom Omni
Instead, you must download the Phantom Omni drivers from that [the cisst/SAW project](https://github.com/jhu-cisst-external/3ds-touch-openhaptics?tab=readme-ov-file#introduction). You want to install the **openhaptics-3.4.sh** file and the **3ds-touch-drivers-2022.sh** 
 To simplify installation and ensure compatibility between dependencies, I used the jhu-saw/vcs repository : you should download the ROS 2 version, following the instructions from [here](https://github.com/jhu-saw/vcs)

 ```sh
# you will have trouble installing some libraries that are too old for Ubuntu 24.04 so rather than doing
sudo apt install libxml2-dev libraw1394-dev libncurses5-dev qtcreator swig sox espeak cmake-curses-gui cmake-qt-gui git subversion gfortran libcppunit-dev libqt5xmlpatterns5-dev libbluetooth-dev libhidapi-dev python3-pyudev gfortran-9 ros-humble-joint-state-publisher* ros-humble-xacro
```

 ```sh
#you should do that 
wget http://ftp.de.debian.org/debian/pool/main/n/ncurses/libncurses5_6.1+20181013-2+deb10u2_amd64.deb
wget http://ftp.de.debian.org/debian/pool/main/n/ncurses/libtinfo5_6.1+20181013-2+deb10u2_amd64.deb
sudo dpkg -i libtinfo5_6.1+20181013-2+deb10u2_amd64.deb
sudo dpkg -i libncurses5_6.1+20181013-2+deb10u2_amd64.deb

sudo apt install python3-vcstool python3-colcon-common-extensions python3-pykdl
sudo apt install libxml2-dev libraw1394-dev libncurses5-dev qtcreator swig sox espeak cmake-curses-gui cmake-qt-gui git subversion gfortran libcppunit-dev libqt5xmlpatterns5-dev libbluetooth-dev libhidapi-dev python3-pyudev gfortran-9 ros-jazzy-joint-state-publisher* ros-jazzy-xacro
```

After, follow the rest of the instructions. The build might take some time depending on your computer.

To test if you've successfully installed the driver, you should first source your ROS workspaces and then do : 
 ```sh
sudo chmod 777 /dev/ttyACM0 #replace the zero with the right number 
ros2 launch sensable_omni_model omni.launch.py  
ros2 run sensable_phantom sensable_phantom    #in a new terminal
```

You can then use the scripts (after changing some paths so they align with where you downloaded the extensions) by launching them in the Slicer 3D console. 

If you have any questions or suggestions you can contact me at eleonore.germond@imt-atlantique.net . 
