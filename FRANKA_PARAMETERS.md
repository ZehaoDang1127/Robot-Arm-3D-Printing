# Franka Panda Parameters Extracted From GitHub Reference

Source repository:
`Bochicchio3/Controlling_the_KINOVA7DOF`

Source file:
[`panda7dof_robot_gen.m`](https://github.com/Bochicchio3/Controlling_the_KINOVA7DOF/blob/master/panda7dof_robot_gen.m)

The source file defines a Franka Emika Panda model using Peter Corke's MATLAB
Robotics Toolbox with modified Denavit-Hartenberg links:

```matlab
Link([theta, d, a, alpha], "modified")
```

The extracted Python version is in `franka_panda_parameters.py`.

## Comparison With Previous Stage-3 Model

Before this extraction, `stage3_franka_ik.py` used an inline hand-written
kinematic chain. That chain was convenient, but it mixed a standard-DH-style
implementation with constants that did not exactly match the extracted MATLAB
modified-DH model.

The important differences are:

- The extracted model is explicitly **modified DH**.
- The previous inline model used a different first-link twist: `alpha=-pi/2`
  instead of the extracted modified-DH `alpha=0`.
- The previous inline model added `d=0.107 m` on link 7; the extracted table
  has `d=0` and `a=0.088 m` for link 7.
- The extracted model includes masses, centers of mass, and inertia tensors.
  The previous Stage-3 model used only kinematics.

`stage3_franka_ik.py` now imports `MODIFIED_DH_LINKS` and
`STANDARD_PANDA_JOINT_LIMITS_RAD` from `franka_panda_parameters.py` instead of
duplicating the robot geometry inline.

Run this command to print the comparison:

```bash
python compare_franka_models.py
```

## Modified DH Geometry

Units are meters and radians.

| link | theta offset | d | a | alpha |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 0.333 | 0 | 0 |
| 2 | 0 | 0 | 0 | -pi/2 |
| 3 | 0 | 0.316 | 0 | pi/2 |
| 4 | 0 | 0 | 0.0825 | pi/2 |
| 5 | 0 | 0.384 | -0.0825 | -pi/2 |
| 6 | 0 | 0 | 0 | pi/2 |
| 7 | 0 | 0 | 0.088 | pi/2 |

Length constants:

| name | value m |
| --- | ---: |
| d1 | 0.333 |
| d3 | 0.316 |
| d5 | 0.384 |
| a4 | 0.0825 |
| a5 | -0.0825 |
| a7 | 0.088 |

## Link Masses

| link | mass kg |
| --- | ---: |
| 1 | 3.4525 |
| 2 | 3.4821 |
| 3 | 4.0562 |
| 4 | 3.4822 |
| 5 | 2.1633 |
| 6 | 2.3466 |
| 7 | 0.31290 |

Total extracted link mass: 19.2958 kg.

## Centers Of Mass

Each COM vector is stored as `PANDA.links(i).r` in the source file.

| link | rx m | ry m | rz m |
| --- | ---: | ---: | ---: |
| 1 | 0 | -0.03 | 0.12 |
| 2 | 0.0003 | 0.059 | 0.042 |
| 3 | 0 | 0.03 | 0.13 |
| 4 | 0 | 0.067 | 0.034 |
| 5 | 0.0001 | 0.021 | 0.076 |
| 6 | 0 | 0.0006 | 0.0004 |
| 7 | 0 | 0 | 0.02 |

## Inertia Tensors

Each tensor is a 3x3 matrix in kg m^2.

### Link 1

```text
[[0.0747, 0.0085, 0],
 [0.0085, 0.0574, 0],
 [0, 0, 0.0239]]
```

### Link 2

```text
[[0.0390, -0.0086, -0.0037],
 [-0.0086, 0.0279, -6.1633e-05],
 [-0.0037, -6.1633e-05, 0.0199]]
```

### Link 3

```text
[[0.006052050623697, 0.000000262383560, 0.000001120384479],
 [0.000000262383560, 0.005990028254028, -0.001308542301422],
 [0.000001120384479, -0.001308542301422, 0.001861529721327]]
```

### Link 4

```text
[[0.006052050623697, -0.000000262507583, -0.000001120888863],
 [-0.000000262507583, 0.005990028254028, -0.001308542301422],
 [-0.000001120888863, -0.001308542301422, 0.001861529721327]]
```

### Link 5

```text
[[0.005775526977146, -0.000000448127278, 0.000000782342032],
 [-0.000000448127278, 0.005348473437925, 0.001819965983941],
 [0.000000782342032, 0.001819965983941, 0.002181233531810]]
```

### Link 6

```text
[[0.001882302441080, 0.000000003150206, -0.000000072256604],
 [0.000000003150206, 0.001889339660303, -0.000012066987492],
 [-0.000000072256604, -0.000012066987492, 0.002133520179065]]
```

### Link 7

```text
[[0.0003390625, 0, 0],
 [0, 0.0003390625, 0],
 [0, 0, 0.000528125]]
```

## Joint Limits

The referenced MATLAB file does **not** define joint limits. The Python module
therefore includes standard Panda joint limits separately as
`STANDARD_PANDA_JOINT_LIMITS_RAD`, but those limits are not extracted from this
GitHub source file.
