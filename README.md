# Problem Statement

Design a simulation environment with any 6 DOF robot arm and 2 finger gripper attached to it in mujoco environment

- The robot arm must be able to pick and place a cube like object in the scene. You can hard code the pose where cube object is placed. No vision system integration is needed.
- Given a 3d pose of the cube object
    - Convert the given cartesian pose to joint position via inverse kinematics module (any open source libraries)
    - Create motion plans via motion planner like RRT or PRM or CHOMP (any open source libraries) 
    - Execute the generated plans on mujoco environment on the robot arm to perform pick and place task using position control

You can use python for this assignment and using ROS for this assignment is optional. Usage of any third party libraries are completely allowed.

## Overview

## Setup
Setup the `conda` environment for our repository by running
```
conda env create -f environment.yml
conda activate push_mujoco
```
NOTE: MuJoCo Python require at least one of the three OpenGL rendering backends: EGL (headless, hardware-accelerated), GLFW (windowed, hardware-accelerated), and OSMesa (purely software-based). Refer to Google Deepmind's [dm_control](https://github.com/google-deepmind/dm_control) repository for further instructions.

## Usage

## Acknowledgement
### MuJoCo Assets
```bash
├── franka_emika_panda
│   ├── assets
│   │   ├── finger_0.obj
│   │   ├── ...
│   └── panda_push.xml
│   └── scene_push.xml
```
The robot arm model used is from Google Deepmind's [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) repository. Specifically, we use the [Franka Emika Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) model which is a 7-axis robot arm.
- `panda_push.xml` MJCF description file is modified from their `panda_nohand.xml` MJCF description file to include a rounded rod-shaped end effector for the block pushing task.
- `scene_push.xml` MCJF description file is heavily modified from their `scene_xml`MCJF description file to include objects in our custom MuJoCo simulation environment like a table, a coloured block, two cameras for image input and 3 coloured target regions.
- `assets` directory includes all assets from the repository for the Franka Emika Panda robot arm. 


### Inverse Kinematics
The implementation of inverse kinematics for the MuJoCo simulator in `inverse_kinematics.py` is taken from Google Deepmind's [dm_control](https://github.com/google-deepmind/dm_control) repository which is meant for Reinforcement Learning in physics-based simulation like MuJoCo.

## Package Versions
-   mujoco=3.3.0