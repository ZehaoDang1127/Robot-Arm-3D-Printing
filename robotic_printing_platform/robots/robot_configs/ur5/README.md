# Universal Robots UR5 Configuration

This folder is the complete robot-specific package used by the planner for the
original UR5 (not UR5e). It follows the same folder contract as
`franka_panda/`: a relative URDF plus a `robot_config.json` containing the
planning chain, joint names, home pose, limits, and nominal reach.

The kinematic dimensions are the published UR5 standard-DH dimensions used by
Universal Robots' ROS description. The 0.85 m nominal reach, six rotating
joints, and +/-360 degree planning ranges come from the UR5 technical
specification:

- https://www.universal-robots.com/media/1828033/ur5_tech_spec_web_en.pdf
- https://github.com/UniversalRobots/Universal_Robots_ROS2_Description

The +/-360 degree ranges are planning bounds rather than a substitute for the
joint limits and safety configuration on a particular physical robot. Calibrate
the UR controller, base/bed transform, and nozzle TCP before executing a
trajectory on hardware.

Set `planner_config.json -> robot.config_dir` to this folder, or run the
pipeline with `--robot ur5`.
