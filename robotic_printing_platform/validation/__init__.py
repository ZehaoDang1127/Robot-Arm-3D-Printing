"""Validation and quality reports for robot printing trajectories."""

__all__ = [
    "ErrorStatistics",
    "TrajectoryValidationReport",
    "validate_trajectory",
    "ToleranceSweepEntry",
    "ToleranceSweepReport",
    "sweep_position_tolerances",
]


def __getattr__(name: str):
    """Avoid importing trajectory reports while robot kinematics is loading."""
    if name in {"ErrorStatistics", "TrajectoryValidationReport", "validate_trajectory"}:
        from .report import ErrorStatistics, TrajectoryValidationReport, validate_trajectory

        values = {
            "ErrorStatistics": ErrorStatistics,
            "TrajectoryValidationReport": TrajectoryValidationReport,
            "validate_trajectory": validate_trajectory,
        }
    elif name in {"ToleranceSweepEntry", "ToleranceSweepReport", "sweep_position_tolerances"}:
        from .tolerance_sweep import ToleranceSweepEntry, ToleranceSweepReport, sweep_position_tolerances

        values = {
            "ToleranceSweepEntry": ToleranceSweepEntry,
            "ToleranceSweepReport": ToleranceSweepReport,
            "sweep_position_tolerances": sweep_position_tolerances,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals().update(values)
    return values[name]
