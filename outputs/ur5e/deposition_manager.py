"""Simulator-independent material-flow integration and deposition sampling.

The classes in this module deliberately know nothing about Isaac Sim or USD.
They turn timestamped, measured TCP poses and volumetric material flow into
segment/point emissions through a small sink protocol.  Simulator exporters
can therefore choose their own visual, voxel, mesh, or particle backend while
sharing the timing, continuity, and volume-conservation rules.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

_TIME_EPSILON_S = 1.0e-12
_VOLUME_EPSILON_MM3 = 1.0e-12
_ORIENTATION_EPSILON_RAD = 1.0e-12


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class _EvolutionMaterialProfile(Protocol):
    """Structural subset needed to build a standalone evolution model."""

    spreading_ratio: float
    spreading_time_s: float
    shrinkage_fraction: float
    shrinkage_time_s: float


@dataclass(frozen=True)
class BeadEvolutionState:
    """Dimensionless geometry scales for one deposited bead at a given age.

    ``volume_scale`` is the fraction of the initial deposited volume that
    remains.  For a fixed-length bead with an elliptical cross-section,
    ``width_scale * height_scale == volume_scale`` so the two directional
    scales conserve the remaining cross-sectional area.  A backend limited to
    round curves or spheres can use ``visual_radius_scale`` as a deliberately
    approximate scalar that responds to both spreading and shrinkage.
    """

    age_s: float
    spreading_scale: float
    volume_scale: float
    width_scale: float
    height_scale: float
    visual_radius_scale: float


@dataclass(frozen=True)
class BeadEvolutionModel:
    """Constant-parameter exponential spreading and shrinkage model.

    ``spreading_ratio`` is the asymptotic lateral width divided by the initial
    width and must be at least one.  ``shrinkage_fraction`` is the asymptotic
    fraction of the initially deposited volume that is lost.  The two time
    values are exponential e-folding time constants in seconds.  Neutral
    defaults preserve the deposited geometry indefinitely.
    """

    spreading_ratio: float = 1.0
    spreading_time_s: float = 1.0
    shrinkage_fraction: float = 0.0
    shrinkage_time_s: float = 1.0

    def __post_init__(self) -> None:
        spreading_ratio = _finite_float(self.spreading_ratio, "spreading_ratio")
        spreading_time_s = _finite_float(
            self.spreading_time_s, "spreading_time_s"
        )
        shrinkage_fraction = _finite_float(
            self.shrinkage_fraction, "shrinkage_fraction"
        )
        shrinkage_time_s = _finite_float(
            self.shrinkage_time_s, "shrinkage_time_s"
        )
        if spreading_ratio < 1.0:
            raise ValueError("spreading_ratio must be greater than or equal to one")
        if spreading_time_s <= 0.0:
            raise ValueError("spreading_time_s must be positive")
        if not 0.0 <= shrinkage_fraction < 1.0:
            raise ValueError("shrinkage_fraction must be in [0, 1)")
        if shrinkage_time_s <= 0.0:
            raise ValueError("shrinkage_time_s must be positive")
        object.__setattr__(self, "spreading_ratio", spreading_ratio)
        object.__setattr__(self, "spreading_time_s", spreading_time_s)
        object.__setattr__(self, "shrinkage_fraction", shrinkage_fraction)
        object.__setattr__(self, "shrinkage_time_s", shrinkage_time_s)

    @classmethod
    def from_material_profile(
        cls, profile: "_EvolutionMaterialProfile"
    ) -> "BeadEvolutionModel":
        """Create a model from the effective constants in a material profile."""

        return cls(
            spreading_ratio=profile.spreading_ratio,
            spreading_time_s=profile.spreading_time_s,
            shrinkage_fraction=profile.shrinkage_fraction,
            shrinkage_time_s=profile.shrinkage_time_s,
        )

    def state_at(self, age_s: float) -> BeadEvolutionState:
        """Return geometry scales after ``age_s`` seconds since deposition."""

        age = _finite_float(age_s, "age_s")
        if age < 0.0:
            raise ValueError("age_s must be non-negative")

        spreading_progress = -math.expm1(-age / self.spreading_time_s)
        shrinkage_progress = -math.expm1(-age / self.shrinkage_time_s)
        spreading_scale = 1.0 + (
            self.spreading_ratio - 1.0
        ) * spreading_progress
        volume_scale = 1.0 - self.shrinkage_fraction * shrinkage_progress
        width_scale = spreading_scale
        height_scale = volume_scale / spreading_scale
        visual_radius_scale = math.sqrt(spreading_scale * volume_scale)
        return BeadEvolutionState(
            age_s=age,
            spreading_scale=spreading_scale,
            volume_scale=volume_scale,
            width_scale=width_scale,
            height_scale=height_scale,
            visual_radius_scale=visual_radius_scale,
        )


@dataclass(frozen=True, eq=False)
class TcpPose:
    """A nozzle TCP pose in the simulation world frame.

    Position units are metres.  Quaternion order is explicitly ``xyzw``;
    adapters for APIs that return ``wxyz`` must reorder their values before
    constructing this object.
    """

    position_m: np.ndarray
    orientation_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    )

    def __post_init__(self) -> None:
        position = np.array(self.position_m, dtype=float, copy=True)
        orientation = np.array(self.orientation_xyzw, dtype=float, copy=True)
        if position.shape != (3,):
            raise ValueError(f"position_m must have shape (3,), got {position.shape}")
        if orientation.shape != (4,):
            raise ValueError(
                "orientation_xyzw must have shape (4,), "
                f"got {orientation.shape}"
            )
        if not np.all(np.isfinite(position)):
            raise ValueError("position_m must contain only finite values")
        if not np.all(np.isfinite(orientation)):
            raise ValueError("orientation_xyzw must contain only finite values")
        norm = float(np.linalg.norm(orientation))
        if norm <= 1.0e-15:
            raise ValueError("orientation_xyzw must have non-zero norm")
        orientation /= norm
        position.setflags(write=False)
        orientation.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_xyzw", orientation)

    def interpolate(self, other: "TcpPose", alpha: float) -> "TcpPose":
        """Interpolate position and take the shortest normalized quaternion path."""

        fraction = _finite_float(alpha, "alpha")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("alpha must be between zero and one")
        position = self.position_m + fraction * (other.position_m - self.position_m)
        first = self.orientation_xyzw
        second = other.orientation_xyzw
        if float(np.dot(first, second)) < 0.0:
            second = -second
        orientation = (1.0 - fraction) * first + fraction * second
        norm = float(np.linalg.norm(orientation))
        if norm <= 1.0e-15:
            orientation = first.copy()
        else:
            orientation /= norm
        return TcpPose(position, orientation)

    def angular_distance_rad(self, other: "TcpPose") -> float:
        """Return the shortest unsigned rotation between two orientations."""

        cosine_half_angle = min(
            1.0,
            max(0.0, abs(float(np.dot(self.orientation_xyzw, other.orientation_xyzw)))),
        )
        return 2.0 * math.acos(cosine_half_angle)


@dataclass(frozen=True)
class FlowInterval:
    """One half-open interval of constant volumetric material flow."""

    start_time_s: float
    end_time_s: float
    material_flow_mm3_s: float

    def __post_init__(self) -> None:
        start = _finite_float(self.start_time_s, "start_time_s")
        end = _finite_float(self.end_time_s, "end_time_s")
        rate = _finite_float(self.material_flow_mm3_s, "material_flow_mm3_s")
        if end <= start:
            raise ValueError("end_time_s must be greater than start_time_s")
        if rate < 0.0:
            raise ValueError("material_flow_mm3_s must be non-negative")
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "end_time_s", end)
        object.__setattr__(self, "material_flow_mm3_s", rate)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def volume_mm3(self) -> float:
        return self.material_flow_mm3_s * self.duration_s


@dataclass(frozen=True)
class FlowSlice(FlowInterval):
    """A portion of a schedule returned for a requested time range."""


def _point_value(point: object, name: str, default: object = None) -> object:
    if isinstance(point, Mapping):
        return point.get(name, default)
    return getattr(point, name, default)


class FlowSchedule:
    """A non-overlapping schedule of piecewise-constant material flow.

    Missing time ranges mean zero material flow.  ``split`` nevertheless
    returns slices covering the complete requested range so a caller can keep
    updating the manager's TCP anchor during travel.
    """

    def __init__(self, intervals: Iterable[FlowInterval] = ()) -> None:
        ordered = tuple(sorted(intervals, key=lambda interval: interval.start_time_s))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_time_s < previous.end_time_s:
                raise ValueError(
                    "flow intervals must not overlap: "
                    f"[{previous.start_time_s}, {previous.end_time_s}) and "
                    f"[{current.start_time_s}, {current.end_time_s})"
                )
        self._intervals = ordered
        self._starts = tuple(interval.start_time_s for interval in ordered)
        self._ends = tuple(interval.end_time_s for interval in ordered)

    @classmethod
    def from_trajectory_points(cls, points_or_trajectory: object) -> "FlowSchedule":
        """Build flow intervals from trajectory-point incoming-segment volume.

        Each point's ``extrusion_volume_mm3`` belongs to the segment arriving at
        that point.  Dividing it by the retimed segment duration preserves the
        commanded volume even when joint constraints slow the Cartesian path.
        """

        candidate = getattr(points_or_trajectory, "points", points_or_trajectory)
        points = tuple(candidate)  # type: ignore[arg-type]
        if not points:
            return cls()

        times = tuple(
            _finite_float(_point_value(point, "time_from_start_s"), "time_from_start_s")
            for point in points
        )
        volumes = tuple(
            _finite_float(
                _point_value(point, "extrusion_volume_mm3", 0.0),
                "extrusion_volume_mm3",
            )
            for point in points
        )
        for volume in volumes:
            if volume < 0.0:
                raise ValueError("extrusion_volume_mm3 must be non-negative")
        if volumes[0] > _VOLUME_EPSILON_MM3:
            raise ValueError(
                "the first trajectory point cannot carry incoming extrusion volume"
            )

        intervals: list[FlowInterval] = []
        for index in range(1, len(points)):
            start = times[index - 1]
            end = times[index]
            if end < start:
                raise ValueError("trajectory timestamps must be non-decreasing")
            volume = volumes[index]
            is_print = bool(_point_value(points[index], "is_print", volume > 0.0))
            if volume > _VOLUME_EPSILON_MM3 and not is_print:
                raise ValueError(
                    "a non-print trajectory point carries positive extrusion volume"
                )
            if end - start <= _TIME_EPSILON_S:
                if volume > _VOLUME_EPSILON_MM3:
                    raise ValueError(
                        "positive extrusion volume requires a positive segment duration"
                    )
                continue
            rate = volume / (end - start) if is_print else 0.0
            intervals.append(FlowInterval(start, end, rate))
        return cls(intervals)

    @property
    def intervals(self) -> tuple[FlowInterval, ...]:
        return self._intervals

    def flow_rate_at(self, time_s: float) -> float:
        """Return the half-open interval rate at ``time_s``."""

        time = _finite_float(time_s, "time_s")
        index = bisect_right(self._starts, time) - 1
        if index >= 0 and time < self._ends[index]:
            return self._intervals[index].material_flow_mm3_s
        return 0.0

    def split(self, start_time_s: float, end_time_s: float) -> tuple[FlowSlice, ...]:
        """Partition a range at every flow boundary, including zero-flow gaps."""

        start = _finite_float(start_time_s, "start_time_s")
        end = _finite_float(end_time_s, "end_time_s")
        if end < start:
            raise ValueError("end_time_s must not be earlier than start_time_s")
        if end == start:
            return ()

        boundaries = {start, end}
        first = bisect_right(self._ends, start)
        last = bisect_left(self._starts, end)
        for interval in self._intervals[first:last]:
            if start < interval.start_time_s < end:
                boundaries.add(interval.start_time_s)
            if start < interval.end_time_s < end:
                boundaries.add(interval.end_time_s)
        ordered = sorted(boundaries)
        return tuple(
            FlowSlice(
                piece_start,
                piece_end,
                self.flow_rate_at(piece_start + 0.5 * (piece_end - piece_start)),
            )
            for piece_start, piece_end in zip(ordered, ordered[1:])
        )

    def volume_between(self, start_time_s: float, end_time_s: float) -> float:
        """Integrate scheduled material volume exactly across interval boundaries."""

        return math.fsum(piece.volume_mm3 for piece in self.split(start_time_s, end_time_s))


class DepositionSink(Protocol):
    """Backend implemented by a visual, mesh, voxel, or particle emitter."""

    def emit_segment(
        self,
        start_pose: TcpPose,
        end_pose: TcpPose,
        volume_mm3: float,
        duration_s: float,
    ) -> None:
        ...

    def emit_point(
        self,
        pose: TcpPose,
        volume_mm3: float,
        duration_s: float,
    ) -> None:
        ...

    def reset(self, clear_geometry: bool = False) -> None:
        ...

    def flush(self) -> None:
        ...


@dataclass(frozen=True)
class DepositionUpdate:
    """Result of one constant-flow manager interval."""

    kind: str
    start_time_s: float | None
    end_time_s: float
    elapsed_time_s: float
    material_flow_mm3_s: float
    commanded_volume_mm3: float
    emitted_volume_mm3: float
    unrepresented_volume_mm3: float
    start_pose: TcpPose | None
    end_pose: TcpPose
    reason: str | None = None


@dataclass
class DepositionStatistics:
    """Cumulative accounting since the last statistics reset."""

    observed_samples: int = 0
    interval_updates: int = 0
    commanded_volume_mm3: float = 0.0
    emitted_volume_mm3: float = 0.0
    unrepresented_volume_mm3: float = 0.0
    emitted_segments: int = 0
    emitted_points: int = 0
    discontinuities: int = 0
    time_resets: int = 0
    disabled_intervals: int = 0


class DepositionManager:
    """Integrate material flow along timestamped measured TCP poses.

    ``material_flow_mm3_s`` passed to :meth:`update` is the flow command that
    applied over ``(previous_time, simulation_time_s]``.  The first call only
    establishes an anchor.  Call the manager during travel too, with zero flow,
    so a later printed segment starts at the current nozzle position.
    """

    def __init__(
        self,
        sink: DepositionSink,
        *,
        max_tcp_step_m: float | None = 0.02,
        stationary_tolerance_m: float = 1.0e-9,
        max_tcp_orientation_step_rad: float | None = None,
    ) -> None:
        if max_tcp_step_m is not None:
            max_tcp_step_m = _finite_float(max_tcp_step_m, "max_tcp_step_m")
            if max_tcp_step_m <= 0.0:
                raise ValueError("max_tcp_step_m must be positive or None")
        stationary_tolerance_m = _finite_float(
            stationary_tolerance_m, "stationary_tolerance_m"
        )
        if stationary_tolerance_m < 0.0:
            raise ValueError("stationary_tolerance_m must be non-negative")
        if max_tcp_orientation_step_rad is not None:
            max_tcp_orientation_step_rad = _finite_float(
                max_tcp_orientation_step_rad,
                "max_tcp_orientation_step_rad",
            )
            if max_tcp_orientation_step_rad <= 0.0:
                raise ValueError(
                    "max_tcp_orientation_step_rad must be positive or None"
                )

        self.sink = sink
        self.max_tcp_step_m = max_tcp_step_m
        self.stationary_tolerance_m = stationary_tolerance_m
        self.max_tcp_orientation_step_rad = max_tcp_orientation_step_rad
        self._last_pose: TcpPose | None = None
        self._last_time_s: float | None = None
        self._statistics = DepositionStatistics()
        self._closed = False

    @property
    def statistics(self) -> DepositionStatistics:
        return replace(self._statistics)

    @property
    def last_pose(self) -> TcpPose | None:
        return self._last_pose

    @property
    def last_time_s(self) -> float | None:
        return self._last_time_s

    def _discontinuity_reason(self, start: TcpPose, end: TcpPose) -> str | None:
        distance_m = float(np.linalg.norm(end.position_m - start.position_m))
        if self.max_tcp_step_m is not None and distance_m > self.max_tcp_step_m:
            return "tcp_position_jump"
        if (
            self.max_tcp_orientation_step_rad is not None
            and start.angular_distance_rad(end) > self.max_tcp_orientation_step_rad
        ):
            return "tcp_orientation_jump"
        return None

    def _result(
        self,
        *,
        kind: str,
        end_pose: TcpPose,
        end_time_s: float,
        material_flow_mm3_s: float,
        start_pose: TcpPose | None = None,
        start_time_s: float | None = None,
        commanded_volume_mm3: float = 0.0,
        emitted_volume_mm3: float = 0.0,
        unrepresented_volume_mm3: float = 0.0,
        reason: str | None = None,
    ) -> DepositionUpdate:
        elapsed = 0.0 if start_time_s is None else max(0.0, end_time_s - start_time_s)
        return DepositionUpdate(
            kind=kind,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            elapsed_time_s=elapsed,
            material_flow_mm3_s=material_flow_mm3_s,
            commanded_volume_mm3=commanded_volume_mm3,
            emitted_volume_mm3=emitted_volume_mm3,
            unrepresented_volume_mm3=unrepresented_volume_mm3,
            start_pose=start_pose,
            end_pose=end_pose,
            reason=reason,
        )

    def _update(
        self,
        pose: TcpPose,
        simulation_time_s: float,
        material_flow_mm3_s: float,
        *,
        enabled: bool,
        observed_sample: bool,
    ) -> DepositionUpdate:
        if self._closed:
            raise RuntimeError("deposition manager is closed")
        if not isinstance(pose, TcpPose):
            raise TypeError("pose must be a TcpPose")
        time = _finite_float(simulation_time_s, "simulation_time_s")
        rate = _finite_float(material_flow_mm3_s, "material_flow_mm3_s")
        if rate < 0.0:
            raise ValueError("material_flow_mm3_s must be non-negative")
        if observed_sample:
            self._statistics.observed_samples += 1

        if self._last_pose is None or self._last_time_s is None:
            self._last_pose = pose
            self._last_time_s = time
            return self._result(
                kind="anchor",
                end_pose=pose,
                end_time_s=time,
                material_flow_mm3_s=rate,
            )

        start_pose = self._last_pose
        start_time = self._last_time_s
        if time < start_time:
            self._last_pose = pose
            self._last_time_s = time
            self._statistics.discontinuities += 1
            self._statistics.time_resets += 1
            return self._result(
                kind="discontinuity",
                start_pose=start_pose,
                end_pose=pose,
                start_time_s=start_time,
                end_time_s=time,
                material_flow_mm3_s=rate,
                reason="time_regression",
            )

        elapsed = time - start_time
        if elapsed <= _TIME_EPSILON_S:
            moved = (
                float(np.linalg.norm(pose.position_m - start_pose.position_m))
                > self.stationary_tolerance_m
                or start_pose.angular_distance_rad(pose)
                > _ORIENTATION_EPSILON_RAD
            )
            self._last_pose = pose
            self._last_time_s = time
            if moved:
                self._statistics.discontinuities += 1
                return self._result(
                    kind="discontinuity",
                    start_pose=start_pose,
                    end_pose=pose,
                    start_time_s=start_time,
                    end_time_s=time,
                    material_flow_mm3_s=rate,
                    reason="pose_changed_without_elapsed_time",
                )
            return self._result(
                kind="inactive",
                start_pose=start_pose,
                end_pose=pose,
                start_time_s=start_time,
                end_time_s=time,
                material_flow_mm3_s=rate,
            )

        self._statistics.interval_updates += 1
        commanded_volume = rate * elapsed
        self._statistics.commanded_volume_mm3 += commanded_volume
        reason = self._discontinuity_reason(start_pose, pose)
        if reason is not None:
            self._last_pose = pose
            self._last_time_s = time
            self._statistics.discontinuities += 1
            self._statistics.unrepresented_volume_mm3 += commanded_volume
            return self._result(
                kind="discontinuity",
                start_pose=start_pose,
                end_pose=pose,
                start_time_s=start_time,
                end_time_s=time,
                material_flow_mm3_s=rate,
                commanded_volume_mm3=commanded_volume,
                unrepresented_volume_mm3=commanded_volume,
                reason=reason,
            )

        if commanded_volume <= _VOLUME_EPSILON_MM3:
            self._last_pose = pose
            self._last_time_s = time
            return self._result(
                kind="inactive",
                start_pose=start_pose,
                end_pose=pose,
                start_time_s=start_time,
                end_time_s=time,
                material_flow_mm3_s=rate,
            )

        if not enabled:
            self._last_pose = pose
            self._last_time_s = time
            self._statistics.disabled_intervals += 1
            self._statistics.unrepresented_volume_mm3 += commanded_volume
            return self._result(
                kind="skipped",
                start_pose=start_pose,
                end_pose=pose,
                start_time_s=start_time,
                end_time_s=time,
                material_flow_mm3_s=rate,
                commanded_volume_mm3=commanded_volume,
                unrepresented_volume_mm3=commanded_volume,
                reason="disabled",
            )

        distance_m = float(np.linalg.norm(pose.position_m - start_pose.position_m))
        if distance_m <= self.stationary_tolerance_m:
            self.sink.emit_point(pose, commanded_volume, elapsed)
            kind = "point"
            self._statistics.emitted_points += 1
        else:
            self.sink.emit_segment(start_pose, pose, commanded_volume, elapsed)
            kind = "segment"
            self._statistics.emitted_segments += 1
        self._last_pose = pose
        self._last_time_s = time
        self._statistics.emitted_volume_mm3 += commanded_volume
        return self._result(
            kind=kind,
            start_pose=start_pose,
            end_pose=pose,
            start_time_s=start_time,
            end_time_s=time,
            material_flow_mm3_s=rate,
            commanded_volume_mm3=commanded_volume,
            emitted_volume_mm3=commanded_volume,
        )

    def update(
        self,
        pose: TcpPose,
        simulation_time_s: float,
        material_flow_mm3_s: float,
        *,
        enabled: bool = True,
    ) -> DepositionUpdate:
        """Process one measured pose using the rate active since the last pose."""

        return self._update(
            pose,
            simulation_time_s,
            material_flow_mm3_s,
            enabled=enabled,
            observed_sample=True,
        )

    def update_from_schedule(
        self,
        pose: TcpPose,
        simulation_time_s: float,
        schedule: FlowSchedule,
        *,
        enabled: bool = True,
    ) -> tuple[DepositionUpdate, ...]:
        """Process one measured pose while splitting exact flow-rate boundaries.

        Only the endpoint is measured.  When a physics step straddles a schedule
        boundary, intermediate poses are linearly interpolated between the two
        measured endpoints.  A discontinuous measured step is rejected before
        interpolation so splitting cannot disguise a teleport.
        """

        if not isinstance(schedule, FlowSchedule):
            raise TypeError("schedule must be a FlowSchedule")
        if not isinstance(pose, TcpPose):
            raise TypeError("pose must be a TcpPose")
        time = _finite_float(simulation_time_s, "simulation_time_s")
        if self._last_pose is None or self._last_time_s is None:
            return (
                self._update(
                    pose,
                    time,
                    0.0,
                    enabled=enabled,
                    observed_sample=True,
                ),
            )

        start_pose = self._last_pose
        start_time = self._last_time_s
        if time <= start_time:
            return (
                self._update(
                    pose,
                    time,
                    0.0,
                    enabled=enabled,
                    observed_sample=True,
                ),
            )

        discontinuity = self._discontinuity_reason(start_pose, pose)
        if discontinuity is not None:
            volume = schedule.volume_between(start_time, time)
            return (
                self._update(
                    pose,
                    time,
                    volume / (time - start_time),
                    enabled=enabled,
                    observed_sample=True,
                ),
            )

        slices = schedule.split(start_time, time)
        updates: list[DepositionUpdate] = []
        for index, piece in enumerate(slices):
            alpha = (piece.end_time_s - start_time) / (time - start_time)
            interpolated_pose = start_pose.interpolate(pose, alpha)
            updates.append(
                self._update(
                    interpolated_pose,
                    piece.end_time_s,
                    piece.material_flow_mm3_s,
                    enabled=enabled,
                    observed_sample=(index == len(slices) - 1),
                )
            )
        return tuple(updates)

    def reset(
        self,
        clear_geometry: bool = False,
        *,
        pose: TcpPose | None = None,
        simulation_time_s: float | None = None,
        reset_statistics: bool = True,
    ) -> None:
        """Break deposition continuity and optionally establish a new anchor."""

        if (pose is None) != (simulation_time_s is None):
            raise ValueError("pose and simulation_time_s must be supplied together")
        if pose is not None and not isinstance(pose, TcpPose):
            raise TypeError("pose must be a TcpPose")
        time = (
            None
            if simulation_time_s is None
            else _finite_float(simulation_time_s, "simulation_time_s")
        )
        self.sink.reset(clear_geometry=clear_geometry)
        self._last_pose = pose
        self._last_time_s = time
        if reset_statistics:
            self._statistics = DepositionStatistics()
        self._closed = False

    def flush(self) -> None:
        self.sink.flush()

    def close(self) -> None:
        if not self._closed:
            self.sink.flush()
            self._closed = True


__all__ = [
    "BeadEvolutionModel",
    "BeadEvolutionState",
    "DepositionManager",
    "DepositionSink",
    "DepositionStatistics",
    "DepositionUpdate",
    "FlowInterval",
    "FlowSchedule",
    "FlowSlice",
    "TcpPose",
]
