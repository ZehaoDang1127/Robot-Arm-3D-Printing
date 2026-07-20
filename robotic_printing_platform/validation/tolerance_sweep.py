"""Progressively evaluate IK quality at tighter position tolerances."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from robotic_printing_platform.path_planning import PathPrep
from robotic_printing_platform.robots.generic import RobotTrajectory, URDFIKConfig, solve_urdf_path
from robotic_printing_platform.validation.report import ErrorStatistics


@dataclass(frozen=True)
class ToleranceSweepEntry:
    position_tolerance_m: float
    ik_success_rate: float
    position_error_m: ErrorStatistics
    average_ik_iterations: float
    total_computation_time_s: float


@dataclass(frozen=True)
class ToleranceSweepReport:
    entries: list[ToleranceSweepEntry]

    def summary(self) -> str:
        lines = [
            "IK position-tolerance sweep",
            "tol (mm) | success | pos mean / max / p95 (mm) | avg iters | compute (s)",
        ]
        for entry in self.entries:
            stats = entry.position_error_m
            lines.append(
                f"{entry.position_tolerance_m * 1000.0:8.3f} | "
                f"{entry.ik_success_rate:7.2%} | "
                f"{stats.mean * 1000.0:7.3f} / {stats.maximum * 1000.0:7.3f} / {stats.p95 * 1000.0:7.3f} | "
                f"{entry.average_ik_iterations:9.2f} | {entry.total_computation_time_s:11.3f}"
            )
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def _error_statistics(trajectory: RobotTrajectory) -> ErrorStatistics:
    errors = np.asarray([point.pos_error_m for point in trajectory.points], dtype=float)
    if errors.size == 0:
        return ErrorStatistics(mean=0.0, maximum=0.0, p95=0.0)
    return ErrorStatistics(
        mean=float(np.mean(errors)),
        maximum=float(np.max(errors)),
        p95=float(np.percentile(errors, 95)),
    )


def _entry(trajectory: RobotTrajectory, position_tolerance_m: float, duration_s: float) -> ToleranceSweepEntry:
    attempted = trajectory.report.attempted
    return ToleranceSweepEntry(
        position_tolerance_m=position_tolerance_m,
        ik_success_rate=(float(trajectory.report.solved / attempted) if attempted else 0.0),
        position_error_m=_error_statistics(trajectory),
        average_ik_iterations=(
            float(np.mean([point.ik_iterations for point in trajectory.points]))
            if trajectory.points
            else 0.0
        ),
        total_computation_time_s=duration_s,
    )


def sweep_position_tolerances(
    path: PathPrep,
    cfg: URDFIKConfig,
    tolerances_mm: tuple[float, ...] | list[float] = (8.0, 5.0, 3.0, 2.0, 1.0),
) -> ToleranceSweepReport:
    """Solve one path repeatedly at progressively tighter position tolerances.

    The returned entries preserve the supplied order, which makes the intended
    8 -> 5 -> 3 -> 2 -> 1 mm convergence study explicit in saved reports.
    """
    if not tolerances_mm:
        raise ValueError("tolerances_mm must not be empty")
    entries = []
    for tolerance_mm in tolerances_mm:
        tolerance_mm = float(tolerance_mm)
        if not np.isfinite(tolerance_mm) or tolerance_mm <= 0.0:
            raise ValueError("each position tolerance must be finite and greater than zero")
        tolerance_m = tolerance_mm / 1000.0
        run_cfg = replace(cfg, pos_tol_m=tolerance_m)
        start = perf_counter()
        trajectory = solve_urdf_path(path, run_cfg)
        entries.append(_entry(trajectory, tolerance_m, perf_counter() - start))
    return ToleranceSweepReport(entries=entries)
