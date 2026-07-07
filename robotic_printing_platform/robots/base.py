"""Interfaces for swappable robot planners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from robotic_printing_platform.path_planning import PathPrep


class RobotPlanner(ABC):
    """Convert robot-frame waypoints into a robot-specific trajectory."""

    @abstractmethod
    def solve(self, path: PathPrep):
        """Return a robot-specific trajectory object."""
