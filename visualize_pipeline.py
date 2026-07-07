"""Dependency-light SVG visualization for parsed paths and IK results."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from robotic_printing_platform.gcode import ParseResult
from robotic_printing_platform.path_planning import PathPrep
from robotic_printing_platform.robots import RobotTrajectory


def _project(points: np.ndarray, width: int, height: int, pad: int = 28, axes: tuple[int, int] = (0, 1)):
    pts = points[:, axes]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])

    def map_point(p):
        u = p[axes[0]]
        v = p[axes[1]]
        x = pad + (u - lo[0]) * scale
        y = height - (pad + (v - lo[1]) * scale)
        return x, y

    return map_point


def _polyline(points: list[tuple[float, float]], color: str, width: float, opacity: float) -> str:
    if len(points) < 2:
        return ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-opacity="{opacity}" '
        f'stroke-linecap="round" stroke-linejoin="round" />'
    )


def _write_svg(path: str | Path, title: str, body: str, width: int, height: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fbfbfb" />
<text x="20" y="26" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#202428">{escape(title)}</text>
{body}
</svg>
'''
    path.write_text(svg, encoding="utf-8")
    return path


def plot_gcode_parse(res: ParseResult, path: str | Path) -> Path:
    width, height = 920, 680
    pts = np.array([m.xyz for m in res.moves], dtype=float)
    mapper = _project(pts, width, height)
    body = ["<g>"]
    for start, end, mv in res.iter_segments():
        color = "#1f77b4" if mv.is_print else "#9aa0a6"
        sw = 1.25 if mv.is_print else 0.75
        op = 0.85 if mv.is_print else 0.35
        body.append(_polyline([mapper(np.array(start)), mapper(np.array(end))], color, sw, op))
    body.append('<text x="20" y="620" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#555">blue: print, gray: travel, top view in printer XY</text>')
    body.append("</g>")
    return _write_svg(path, "Parsed Cura G-code", "\n".join(body), width, height)


def plot_waypoints(pathprep: PathPrep, path: str | Path) -> Path:
    width, height = 920, 680
    P = pathprep.positions()
    mapper = _project(P, width, height)
    body = ["<g>"]
    current_seg = None
    current_print = False
    current: list[tuple[float, float]] = []
    for point, wp in zip(P, pathprep.waypoints):
        if current_seg is None:
            current_seg = wp.seg_id
            current_print = wp.is_print
        if wp.seg_id != current_seg:
            body.append(_polyline(current, "#d62728" if current_print else "#9aa0a6", 1.4 if current_print else 0.8, 0.9 if current_print else 0.45))
            current = []
            current_seg = wp.seg_id
            current_print = wp.is_print
        current.append(mapper(point))
    body.append(_polyline(current, "#d62728" if current_print else "#9aa0a6", 1.4 if current_print else 0.8, 0.9 if current_print else 0.45))
    bx, by = mapper(np.array([0.0, 0.0, 0.0]))
    body.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="5" fill="#111" />')
    body.append('<text x="20" y="620" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#555">red: print, gray: travel, black dot: robot base, top view in base XY</text>')
    body.append("</g>")
    return _write_svg(path, "Prepared Robot Waypoints", "\n".join(body), width, height)


def plot_waypoints_xz(pathprep: PathPrep, path: str | Path) -> Path:
    width, height = 920, 680
    P = pathprep.positions()
    mapper = _project(P, width, height, axes=(0, 2))
    body = ["<g>"]
    current_seg = None
    current_print = False
    current: list[tuple[float, float]] = []
    for point, wp in zip(P, pathprep.waypoints):
        if current_seg is None:
            current_seg = wp.seg_id
            current_print = wp.is_print
        if wp.seg_id != current_seg:
            body.append(_polyline(current, "#d62728" if current_print else "#9aa0a6", 1.4 if current_print else 0.8, 0.9 if current_print else 0.45))
            current = []
            current_seg = wp.seg_id
            current_print = wp.is_print
        current.append(mapper(point))
    body.append(_polyline(current, "#d62728" if current_print else "#9aa0a6", 1.4 if current_print else 0.8, 0.9 if current_print else 0.45))
    bx, bz = mapper(np.array([0.0, 0.0, 0.0]))
    body.append(f'<circle cx="{bx:.1f}" cy="{bz:.1f}" r="5" fill="#111" />')
    body.append('<text x="20" y="620" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#555">red: print, gray: travel, black dot: robot base, side view in base XZ</text>')
    body.append("</g>")
    return _write_svg(path, "Prepared Robot Waypoints XZ", "\n".join(body), width, height)


def plot_joint_trajectory(traj: RobotTrajectory, path: str | Path) -> Path:
    width, height = 920, 360
    Q = traj.q_array()
    if len(Q) < 2:
        body = '<text x="20" y="80" font-family="Segoe UI, Arial, sans-serif" font-size="13">No joint trajectory available.</text>'
        return _write_svg(path, "Franka Joint Motion", body, width, height)

    dq = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    maxv = max(float(np.max(dq)), 1e-9)
    xs = np.linspace(30, width - 30, len(dq))
    ys = height - 36 - (dq / maxv) * (height - 86)
    points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    body = [
        f'<line x1="30" y1="{height - 36}" x2="{width - 30}" y2="{height - 36}" stroke="#d6d9dd" />',
        _polyline(points, "#2ca02c", 1.5, 0.95),
        f'<text x="20" y="{height - 14}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#555">per-step joint displacement, max {maxv:.4f} rad</text>',
    ]
    for failed in traj.report.failed_indices:
        x = 30 + failed / max(1, traj.report.attempted - 1) * (width - 60)
        body.append(f'<line x1="{x:.1f}" y1="42" x2="{x:.1f}" y2="{height - 36}" stroke="#d62728" stroke-opacity="0.5" />')
    return _write_svg(path, "Optimized Franka Joint Motion", "\n".join(body), width, height)


def write_all_plots(
    res: ParseResult,
    pathprep: PathPrep,
    traj: RobotTrajectory | None,
    output_dir: str | Path = "outputs",
) -> dict[str, Path]:
    out = Path(output_dir)
    plots = {
        "gcode": plot_gcode_parse(res, out / "gcode_path.svg"),
        "waypoints": plot_waypoints(pathprep, out / "robot_waypoints.svg"),
        "waypoints_xz": plot_waypoints_xz(pathprep, out / "robot_waypoints_xz.svg"),
    }
    if traj is not None:
        plots["joints"] = plot_joint_trajectory(traj, out / "joint_trajectory.svg")
    return plots
