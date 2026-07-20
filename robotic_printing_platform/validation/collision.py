"""Lightweight capsule-based collision warnings for robot printing paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinkCapsule:
    """A robot link approximated by a segment swept by a radius."""

    name: str
    start_m: np.ndarray
    end_m: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class AxisAlignedBox:
    minimum_m: np.ndarray
    maximum_m: np.ndarray


def merge_boxes(first: AxisAlignedBox | None, second: AxisAlignedBox | None) -> AxisAlignedBox | None:
    if first is None:
        return second
    if second is None:
        return first
    return AxisAlignedBox(
        minimum_m=np.minimum(first.minimum_m, second.minimum_m),
        maximum_m=np.maximum(first.maximum_m, second.maximum_m),
    )


def box_from_point(point_m: np.ndarray, padding_m: float = 0.0) -> AxisAlignedBox:
    point = np.asarray(point_m, dtype=float)
    padding = np.full(3, padding_m, dtype=float)
    return AxisAlignedBox(point - padding, point + padding)


def segment_segment_distance_m(
    first_start_m: np.ndarray,
    first_end_m: np.ndarray,
    second_start_m: np.ndarray,
    second_end_m: np.ndarray,
) -> float:
    """Minimum distance between two finite 3D line segments."""
    p0 = np.asarray(first_start_m, dtype=float)
    p1 = np.asarray(first_end_m, dtype=float)
    q0 = np.asarray(second_start_m, dtype=float)
    q1 = np.asarray(second_end_m, dtype=float)
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b

    if a < 1e-12 and c < 1e-12:
        return float(np.linalg.norm(p0 - q0))
    if a < 1e-12:
        s = 0.0
        t = float(np.clip(e / c, 0.0, 1.0))
    elif c < 1e-12:
        t = 0.0
        s = float(np.clip(-d / a, 0.0, 1.0))
    else:
        if denominator < 1e-12:
            s = 0.0
        else:
            s = float(np.clip((b * e - c * d) / denominator, 0.0, 1.0))
        t = float(np.clip((a * e - b * d) / denominator, 0.0, 1.0))
        s = float(np.clip((b * t - d) / a, 0.0, 1.0))

    return float(np.linalg.norm(w + s * u - t * v))


def point_box_distance_m(point_m: np.ndarray, box: AxisAlignedBox) -> float:
    point = np.asarray(point_m, dtype=float)
    delta = np.maximum(box.minimum_m - point, 0.0) + np.minimum(box.maximum_m - point, 0.0)
    return float(np.linalg.norm(delta))


def capsule_box_distance_m(capsule: LinkCapsule, box: AxisAlignedBox, samples: int = 33) -> float:
    """Conservative lightweight sampled segment-to-box distance minus radius."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    points = np.linspace(capsule.start_m, capsule.end_m, samples)
    return min(point_box_distance_m(point, box) for point in points) - capsule.radius_m


def collision_warnings(
    capsules: list[LinkCapsule],
    *,
    bed_box: AxisAlignedBox,
    bed_z_m: float,
    bed_clearance_m: float,
    printed_volume: AxisAlignedBox | None = None,
    nozzle_clearance_m: float = 0.0005,
) -> list[str]:
    """Return non-blocking warnings for bed, self, and deposited-part clearance."""
    warnings = []
    arm_capsules = [capsule for capsule in capsules if capsule.name != "tool"]
    for capsule in arm_capsules:
        min_z = min(float(capsule.start_m[2]), float(capsule.end_m[2])) - capsule.radius_m
        if min_z < bed_z_m + bed_clearance_m:
            warnings.append(
                f"{capsule.name}: capsule below bed clearance "
                f"({min_z:.4f} m < {bed_z_m + bed_clearance_m:.4f} m)"
            )
        if capsule_box_distance_m(capsule, bed_box) < bed_clearance_m:
            warnings.append(f"{capsule.name}: capsule near or inside the print-bed boundary")

    for i, first in enumerate(capsules):
        for second in capsules[i + 2:]:  # adjacent capsules share a joint by design
            clearance = segment_segment_distance_m(
                first.start_m, first.end_m, second.start_m, second.end_m
            ) - first.radius_m - second.radius_m
            if clearance < 0.0:
                warnings.append(f"{first.name}/{second.name}: capsule self-collision")

    tool = next((capsule for capsule in capsules if capsule.name == "tool"), None)
    if tool is not None and printed_volume is not None:
        nozzle_distance = point_box_distance_m(tool.end_m, printed_volume) - tool.radius_m
        if nozzle_distance < nozzle_clearance_m:
            warnings.append("tool: nozzle holder is near or inside previously printed material")
    return warnings
