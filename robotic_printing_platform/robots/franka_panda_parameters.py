"""
Franka Emika Panda parameters extracted from:

https://github.com/Bochicchio3/Controlling_the_KINOVA7DOF/blob/master/panda7dof_robot_gen.m

The source MATLAB file builds a Peter Corke Robotics Toolbox SerialLink model
using *modified* Denavit-Hartenberg links:

    Link([theta, d, a, alpha], "modified")

It also assigns link masses, center-of-mass vectors, and inertia tensors.
The GitHub source does not define joint limits; the standard Panda limits below
are included separately for planner convenience and are marked as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOURCE_REPOSITORY = "Bochicchio3/Controlling_the_KINOVA7DOF"
SOURCE_FILE = "panda7dof_robot_gen.m"
SOURCE_URL = (
    "https://github.com/Bochicchio3/Controlling_the_KINOVA7DOF/"
    "blob/master/panda7dof_robot_gen.m"
)


@dataclass(frozen=True)
class ModifiedDHLink:
    """One modified-DH revolute link: [theta_offset, d, a, alpha]."""

    theta_offset_rad: float
    d_m: float
    a_m: float
    alpha_rad: float


@dataclass(frozen=True)
class DynamicLink:
    """Rigid-body dynamic parameters for one link."""

    mass_kg: float
    com_m: tuple[float, float, float]
    inertia_kg_m2: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


# Length constants from the MATLAB source.
D1_M = 0.333
D3_M = 0.316
D5_M = 0.384
A4_M = 0.0825
A5_M = -0.0825
A7_M = 0.088


# Modified DH table from:
#   L_i = Link([0, d_i, a_i, alpha_i], "modified")
MODIFIED_DH_LINKS: tuple[ModifiedDHLink, ...] = (
    ModifiedDHLink(0.0, D1_M, 0.0, 0.0),
    ModifiedDHLink(0.0, 0.0, 0.0, -np.pi / 2.0),
    ModifiedDHLink(0.0, D3_M, 0.0, np.pi / 2.0),
    ModifiedDHLink(0.0, 0.0, A4_M, np.pi / 2.0),
    ModifiedDHLink(0.0, D5_M, A5_M, -np.pi / 2.0),
    ModifiedDHLink(0.0, 0.0, 0.0, np.pi / 2.0),
    ModifiedDHLink(0.0, 0.0, A7_M, np.pi / 2.0),
)


LINK_DYNAMICS: tuple[DynamicLink, ...] = (
    DynamicLink(
        mass_kg=3.4525,
        com_m=(0.0, -0.03, 0.12),
        inertia_kg_m2=((0.0747, 0.0085, 0.0), (0.0085, 0.0574, 0.0), (0.0, 0.0, 0.0239)),
    ),
    DynamicLink(
        mass_kg=3.4821,
        com_m=(0.0003, 0.059, 0.042),
        inertia_kg_m2=((0.0390, -0.0086, -0.0037), (-0.0086, 0.0279, -6.1633e-05), (-0.0037, -6.1633e-05, 0.0199)),
    ),
    DynamicLink(
        mass_kg=4.0562,
        com_m=(0.0, 0.03, 0.13),
        inertia_kg_m2=((0.006052050623697, 0.000000262383560, 0.000001120384479), (0.000000262383560, 0.005990028254028, -0.001308542301422), (0.000001120384479, -0.001308542301422, 0.001861529721327)),
    ),
    DynamicLink(
        mass_kg=3.4822,
        com_m=(0.0, 0.067, 0.034),
        inertia_kg_m2=((0.006052050623697, -0.000000262507583, -0.000001120888863), (-0.000000262507583, 0.005990028254028, -0.001308542301422), (-0.000001120888863, -0.001308542301422, 0.001861529721327)),
    ),
    DynamicLink(
        mass_kg=2.1633,
        com_m=(0.0001, 0.021, 0.076),
        inertia_kg_m2=((0.005775526977146, -0.000000448127278, 0.000000782342032), (-0.000000448127278, 0.005348473437925, 0.001819965983941), (0.000000782342032, 0.001819965983941, 0.002181233531810)),
    ),
    DynamicLink(
        mass_kg=2.3466,
        com_m=(0.0, 0.0006, 0.0004),
        inertia_kg_m2=((0.001882302441080, 0.000000003150206, -0.000000072256604), (0.000000003150206, 0.001889339660303, -0.000012066987492), (-0.000000072256604, -0.000012066987492, 0.002133520179065)),
    ),
    DynamicLink(
        mass_kg=0.31290,
        com_m=(0.0, 0.0, 0.02),
        inertia_kg_m2=((0.0003390625, 0.0, 0.0), (0.0, 0.0003390625, 0.0), (0.0, 0.0, 0.000528125)),
    ),
)


# Not present in the referenced MATLAB source. Included for planning checks.
STANDARD_PANDA_JOINT_LIMITS_RAD = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=float,
)


def modified_dh_array() -> np.ndarray:
    """Return an Nx4 array with columns theta_offset, d, a, alpha."""
    return np.array(
        [[link.theta_offset_rad, link.d_m, link.a_m, link.alpha_rad] for link in MODIFIED_DH_LINKS],
        dtype=float,
    )


def masses_array() -> np.ndarray:
    return np.array([link.mass_kg for link in LINK_DYNAMICS], dtype=float)


def com_array() -> np.ndarray:
    return np.array([link.com_m for link in LINK_DYNAMICS], dtype=float)


def inertia_array() -> np.ndarray:
    return np.array([link.inertia_kg_m2 for link in LINK_DYNAMICS], dtype=float)


def total_link_mass_kg() -> float:
    return float(masses_array().sum())

