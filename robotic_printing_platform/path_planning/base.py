"""Interfaces for swappable path planning algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from robotic_printing_platform.gcode import ParseResult


class PathPlanningAlgorithm(ABC):
    """Convert parsed G-code moves into robot-frame waypoints."""

    @abstractmethod
    def build(self, res: ParseResult, layers: tuple[int, int]):
        """Return a path-preparation result for the selected layer interval."""

