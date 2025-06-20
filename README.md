
## SlicerSOFA and Slicer ROS2 integration

 This repository contains 2 projects that demonstrate the pertinence and feasibility of the integration of 2 Slicer 3D extensions

- The first one is the Slicer SOFA extension, developped by Rafael Palomar that enables the user to visualize object ( here, organs) deform according to physical constraints (here, the tip of a robot).
- The second one is the Slicer ROS2 extension, developped by Laura Connoly, Anton Deguet and other (i guess, to check).

To use these projects, you will need to build the Slicer ROS2 extension and download the Phantom Omni driver ( you can also use a pretend version). Thus, you need to build Slicer 3D from source. I used ROS2 Jazzy version. The whole setup has been tested and works fine with Ubuntu 24.04. It will not work on a virtual machine using a real Phantom Omni.

Here are two video demos of the code :