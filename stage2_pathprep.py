"""
Stage 2 of the robotic 3D-printing pipeline: path preparation.

Consumes the Stage-1 `ParseResult` and produces the `Waypoint` list that the
IK / redundancy-resolution stage will consume. Three steps:

  1. clean & densify
       - group consecutive PRINT moves into contiguous deposition "runs"
       - drop near-collinear intermediate vertices (angle tolerance) to trim
         the micro-segments slicers emit on curves
       - subdivide any segment longer than `max_seg_len` so robot motion is
         smooth, splitting the extrusion `de` proportionally
       - zero-length prime/un-retract moves become a single dwell waypoint
       - travels between runs are sampled as straight-line waypoints (is_print
         False) so the robot actually traverses them

  2. place_on_bed
       - re-centre the part footprint onto a virtual bed and transform from
         printer-frame millimetres into Franka base-frame metres, via a single
         configurable T_base_bed transform

  3. assign_nozzle_poses
       - the Yao 2021 §3.3.2 scaffold: per-point target nozzle axis, clamped to
         a max-tilt cone, spline-smoothed along each run. For a planar print the
         axis is global-down everywhere and yaw is left free for the redundancy
         stage to optimise.

Contract for Stage 3 (IK):
  Each `Waypoint` carries position (base frame, m), a target nozzle axis,
  a nominal yaw + `yaw_free` flag, `seg_id` (continuity grouping), `is_print`,
  `layer`, feedrate, and `de`. `pose_matrix(yaw)` builds the 4x4 SE(3) target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gcode_parser import ParseResult, parse_gcode


# ----------------------------------------------------------------------------
# data structures
# ----------------------------------------------------------------------------
@dataclass
class Waypoint:
    p: np.ndarray            # (3,) position, Franka base frame, metres
    nozzle_axis: np.ndarray  # (3,) unit, target tool approach axis (points toward work)
    yaw: float               # nominal spin about nozzle axis, rad
    yaw_free: bool           # True => redundancy stage may choose yaw freely
    is_print: bool
    layer: int
    seg_id: int              # contiguous run index; travels get their own ids
    feed_m_s: float
    de: float                # extrusion for this step (volumetric for UltiGCode)

    def pose_matrix(self, yaw: float | None = None) -> np.ndarray:
        """4x4 SE(3) target pose. Tool z-axis == nozzle_axis; yaw spins x,y."""
        R = _frame_from_axis_yaw(self.nozzle_axis, self.yaw if yaw is None else yaw)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.p
        return T


@dataclass
class PathPrep:
    waypoints: list[Waypoint]
    T_base_bed: np.ndarray
    layers: tuple[int, int]

    def positions(self) -> np.ndarray:
        return np.array([w.p for w in self.waypoints])

    def summary(self) -> str:
        P = self.positions()
        prints = [w for w in self.waypoints if w.is_print]
        n_runs = len({w.seg_id for w in prints})
        reach = np.linalg.norm(P, axis=1)            # distance from base origin
        return (
            f"layers {self.layers[0]}..{self.layers[1]-1}\n"
            f"waypoints      : {len(self.waypoints)} "
            f"({len(prints)} print, {len(self.waypoints)-len(prints)} travel)\n"
            f"deposition runs: {n_runs}\n"
            f"base-frame XYZ : X[{P[:,0].min():.3f}, {P[:,0].max():.3f}]  "
            f"Y[{P[:,1].min():.3f}, {P[:,1].max():.3f}]  "
            f"Z[{P[:,2].min():.3f}, {P[:,2].max():.3f}] m\n"
            f"reach from base: {reach.min():.3f} .. {reach.max():.3f} m "
            f"(Franka max ~0.855 m)"
        )


# ----------------------------------------------------------------------------
# small geometry helpers
# ----------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _frame_from_axis_yaw(z_axis: np.ndarray, yaw: float) -> np.ndarray:
    """Right-handed rotation whose 3rd column is z_axis, x/y set by `yaw`."""
    z = _unit(np.asarray(z_axis, float))
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x0 = _unit(np.cross(ref, z))
    x = x0 * math.cos(yaw) + np.cross(z, x0) * math.sin(yaw)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ----------------------------------------------------------------------------
# step 1 — clean & densify  (operates in printer-frame mm)
# ----------------------------------------------------------------------------
def _runs_from_moves(res: ParseResult, lo: int, hi: int):
    """Split the selected layers into (kind, polyline) runs.

    kind is 'print' or 'travel'. polyline is a list of (xyz_mm, de) where de is
    the extrusion deposited arriving at that vertex (0 for the first vertex of
    a run and for all travel vertices).
    """
    runs = []
    cur_kind = None
    cur = None
    prev_xyz = None
    for s, e, mv in res.iter_segments():
        if not (lo <= mv.layer < hi):
            prev_xyz = e
            continue
        kind = "print" if mv.is_print else "travel"
        if kind != cur_kind:
            if cur:
                runs.append((cur_kind, cur))
            # start new run anchored at the segment start
            cur = [(np.array(s, float), 0.0)]
            cur_kind = kind
        cur.append((np.array(e, float), mv.de))
        prev_xyz = e
    if cur:
        runs.append((cur_kind, cur))
    return runs


def _simplify_collinear(poly, angle_tol_deg: float):
    """Remove vertices whose turn angle is below tol; accumulate their de."""
    if len(poly) <= 2 or angle_tol_deg <= 0:
        return poly
    tol = math.radians(angle_tol_deg)
    out = [poly[0]]
    carry = 0.0
    for i in range(1, len(poly) - 1):
        a = out[-1][0]; b = poly[i][0]; c = poly[i + 1][0]
        v1 = _unit(b - a); v2 = _unit(c - b)
        ang = math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)))))
        if ang < tol:                      # near-straight: drop b, carry its de
            carry += poly[i][1]
        else:
            out.append((b, poly[i][1] + carry)); carry = 0.0
    last = poly[-1]
    out.append((last[0], last[1] + carry))
    return out


def _densify(poly, max_seg_len_mm: float):
    """Subdivide segments > max length, splitting de proportionally.

    Returns list of (xyz_mm, de). First vertex carries de=0.
    """
    out = [(poly[0][0], 0.0)]
    for (a, _), (b, de) in zip(poly, poly[1:]):
        L = float(np.linalg.norm(b - a))
        if L <= 1e-9:                      # zero-length prime/dwell
            out.append((b, de))            # keep as a single dwell point
            continue
        n = max(1, int(math.ceil(L / max_seg_len_mm)))
        for k in range(1, n + 1):
            t = k / n
            out.append((a + t * (b - a), de / n))
    return out


# ----------------------------------------------------------------------------
# step 2 — bed placement / frame transform
# ----------------------------------------------------------------------------
def make_bed_transform(bed_center_xy_m=(0.45, 0.0), bed_z_m=0.10) -> np.ndarray:
    """T_base_bed: bed XY plane parallel to base XY, origin at given point."""
    T = np.eye(4)
    T[:3, 3] = [bed_center_xy_m[0], bed_center_xy_m[1], bed_z_m]
    return T


# ----------------------------------------------------------------------------
# top-level entry point
# ----------------------------------------------------------------------------
def build_waypoints(
    res: ParseResult,
    layers: tuple[int, int],
    *,
    max_seg_len_mm: float = 1.0,
    simplify_deg: float = 0.5,
    bed_center_xy_m=(0.45, 0.0),
    bed_z_m: float = 0.10,
    max_tilt_deg: float = 0.0,     # 0 => pure planar, nozzle straight down
) -> PathPrep:
    lo, hi = layers
    runs = _runs_from_moves(res, lo, hi)

    # footprint centre (mm) over the selected layers, for re-centering
    allpts = np.array([v[0] for _, poly in runs for v in poly])
    cx, cy = allpts[:, 0].mean(), allpts[:, 1].mean()

    T = make_bed_transform(bed_center_xy_m, bed_z_m)
    down = np.array([0.0, 0.0, -1.0])      # planar nozzle axis (base frame)

    waypoints: list[Waypoint] = []
    seg = 0
    for kind, poly in runs:
        is_print = kind == "print"
        poly = _simplify_collinear(poly, simplify_deg) if is_print else poly
        dense = _densify(poly, max_seg_len_mm)
        # representative layer for this run (mode of the source layer is fine;
        # all vertices in a run share a layer for planar prints)
        for xyz_mm, de in dense:
            # mm -> m, recentre, then bed transform
            local = np.array([(xyz_mm[0] - cx) / 1000.0,
                              (xyz_mm[1] - cy) / 1000.0,
                              xyz_mm[2] / 1000.0, 1.0])
            p = (T @ local)[:3]
            # planar: surface normal is +Z, nozzle axis is -Z (clamp is a no-op
            # at max_tilt_deg=0; this is where non-planar normals would enter)
            axis = down
            waypoints.append(Waypoint(
                p=p, nozzle_axis=axis, yaw=0.0, yaw_free=True,
                is_print=is_print, layer=lo, seg_id=seg,
                feed_m_s=0.0, de=de,
            ))
        seg += 1

    # carry feedrate onto print waypoints from the source moves (approx: use a
    # constant print feed; Stage 4 re-timing refines this). We pull the modal
    # print feed seen in the slice.
    print_feeds = [m.feed_m_s() for m in res.moves
                   if m.is_print and lo <= m.layer < hi and m.f > 0]
    fps = float(np.median(print_feeds)) if print_feeds else 0.0
    for w in waypoints:
        if w.is_print:
            w.feed_m_s = fps

    return PathPrep(waypoints=waypoints, T_base_bed=T, layers=(lo, hi))


if __name__ == "__main__":
    import sys
    res = parse_gcode(sys.argv[1])
    pp = build_waypoints(res, layers=(0, 3))
    print(pp.summary())
