"""Material-specific extrusion conversion.

G-code `E` values are usually filament length in millimetres. This module keeps
that raw command available while giving the rest of the pipeline a material
profile that can be swapped for another filament, pellet, paste, or syringe
process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialProfile:
    name: str = "PLA"
    filament_diameter_mm: float = 1.75
    flow_multiplier: float = 1.0
    density_g_cm3: float | None = 1.24

    @property
    def filament_area_mm2(self) -> float:
        radius = self.filament_diameter_mm * 0.5
        return math.pi * radius * radius

    def volume_mm3(self, e_delta_mm: float) -> float:
        return max(0.0, e_delta_mm) * self.filament_area_mm2 * self.flow_multiplier

    def mass_g(self, e_delta_mm: float) -> float | None:
        if self.density_g_cm3 is None:
            return None
        return self.volume_mm3(e_delta_mm) * self.density_g_cm3 / 1000.0


@dataclass(frozen=True)
class ExtrusionSample:
    material: str
    e_delta_mm: float
    volume_mm3: float
    mass_g: float | None

    @property
    def active(self) -> bool:
        return self.volume_mm3 > 0.0


def apply_material_profile(e_delta_mm: float, material: MaterialProfile | None = None) -> ExtrusionSample:
    profile = material or MaterialProfile()
    return ExtrusionSample(
        material=profile.name,
        e_delta_mm=e_delta_mm,
        volume_mm3=profile.volume_mm3(e_delta_mm),
        mass_g=profile.mass_g(e_delta_mm),
    )

