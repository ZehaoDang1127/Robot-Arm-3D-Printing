"""Compare the old inline Stage-3 kinematic constants with extracted parameters."""

from __future__ import annotations

import math

import numpy as np

from franka_panda_parameters import MODIFIED_DH_LINKS, modified_dh_array


OLD_INLINE_STANDARD_DH = np.array(
    [
        [0.0, 0.333, 0.0, -math.pi / 2],
        [0.0, 0.0, 0.0, math.pi / 2],
        [0.0, 0.316, 0.0, math.pi / 2],
        [0.0, 0.0, 0.0825, math.pi / 2],
        [0.0, 0.384, -0.0825, -math.pi / 2],
        [0.0, 0.0, 0.0, math.pi / 2],
        [0.0, 0.107, 0.088, math.pi / 2],
    ],
    dtype=float,
)


def main() -> None:
    extracted = modified_dh_array()
    print("Extracted modified-DH table [theta_offset, d, a, alpha]:")
    print(extracted)
    print()
    print("Previous inline Stage-3 table interpreted as [theta_offset, d, a, alpha]:")
    print(OLD_INLINE_STANDARD_DH)
    print()
    print("Geometry differences worth noticing:")
    print("- The extracted table is Peter Corke modified DH, not the previous inline standard-DH chain.")
    print("- Link 1 alpha is 0 in the extracted table; the previous inline model used -pi/2.")
    print("- Link 7 d is 0 in the extracted table; the previous inline model used d=0.107 m.")
    print("- The extracted dynamics include masses, COM vectors, and inertia tensors; the old Stage-3 model did not.")
    print()
    print(f"Extracted link count: {len(MODIFIED_DH_LINKS)}")


if __name__ == "__main__":
    main()

