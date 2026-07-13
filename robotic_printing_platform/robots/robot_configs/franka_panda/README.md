# Franka Panda Robot Configuration

This folder is the complete robot-specific package used by the planner.

To replace the robot, copy this folder, then edit:

- `robot_config.json`: model name, URDF filename, base/end links, active joint names, home joint values, joint limits, and nominal reach.
- `robot.urdf`: serial-chain robot geometry and joint limits.

Keep paths inside `robot_config.json` relative to this folder. Then set
`planner_config.json -> robot.config_dir` to the replacement folder, normally
under `robotic_printing_platform/robots/robot_configs/`.
