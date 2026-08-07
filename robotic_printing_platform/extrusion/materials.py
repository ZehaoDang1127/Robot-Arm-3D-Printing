"""Generic material-profile validation and extrusion conversion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_EXTRUSION_MODES = frozenset(
    {"filament_length", "volumetric", "syringe_plunger"}
)
MATERIAL_PROFILE_FIELDS = frozenset(
    {
        "profile_id",
        "name",
        "extrusion_mode",
        "filament_diameter_mm",
        "syringe_inner_diameter_mm",
        "flow_multiplier",
        "density_g_cm3",
        "physx_particle_contact_offset_m",
        "physx_viscosity",
        "physx_cohesion",
        "physx_adhesion",
        "physx_surface_tension",
        "physx_friction",
        "physx_damping",
        "spreading_ratio",
        "spreading_time_s",
        "shrinkage_fraction",
        "shrinkage_time_s",
    }
)


@dataclass(frozen=True)
class MaterialProfile:
    """A fully resolved material/process profile selected outside this module."""

    profile_id: str
    name: str
    extrusion_mode: str
    filament_diameter_mm: float | None = None
    syringe_inner_diameter_mm: float | None = None
    flow_multiplier: float = 1.0
    density_g_cm3: float | None = None
    physx_particle_contact_offset_m: float = 0.0005
    physx_viscosity: float = 1000.0
    physx_cohesion: float = 5.0
    physx_adhesion: float = 10.0
    physx_surface_tension: float = 0.02
    physx_friction: float = 1000.0
    physx_damping: float = 0.99
    spreading_ratio: float = 1.0
    spreading_time_s: float = 1.0
    shrinkage_fraction: float = 0.0
    shrinkage_time_s: float = 1.0

    def __post_init__(self) -> None:
        allowed_id_characters = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        )
        if not self.profile_id or any(
            character not in allowed_id_characters for character in self.profile_id
        ):
            raise ValueError(
                "profile_id must be non-empty and contain only letters, numbers, '_' or '-'"
            )
        if not self.name.strip():
            raise ValueError("material name must not be empty")
        if self.extrusion_mode not in SUPPORTED_EXTRUSION_MODES:
            raise ValueError(
                f"extrusion_mode must be one of {sorted(SUPPORTED_EXTRUSION_MODES)}, "
                f"got {self.extrusion_mode!r}"
            )
        if self.flow_multiplier <= 0.0 or not math.isfinite(self.flow_multiplier):
            raise ValueError("flow_multiplier must be a positive finite number")
        if self.density_g_cm3 is not None and (
            self.density_g_cm3 <= 0.0 or not math.isfinite(self.density_g_cm3)
        ):
            raise ValueError("density_g_cm3 must be null or a positive finite number")

        if self.extrusion_mode == "filament_length":
            diameter = self.filament_diameter_mm
            if diameter is None or diameter <= 0.0 or not math.isfinite(diameter):
                raise ValueError(
                    "filament_diameter_mm must be a positive finite number "
                    "for filament_length extrusion"
                )
        if self.extrusion_mode == "syringe_plunger":
            diameter = self.syringe_inner_diameter_mm
            if diameter is None or diameter <= 0.0 or not math.isfinite(diameter):
                raise ValueError(
                    "syringe_inner_diameter_mm must be a positive finite number "
                    "for syringe_plunger extrusion"
                )

        if (
            self.physx_particle_contact_offset_m <= 0.0
            or not math.isfinite(self.physx_particle_contact_offset_m)
        ):
            raise ValueError(
                "physx_particle_contact_offset_m must be a positive finite number"
            )
        nonnegative_physx_values = {
            "physx_viscosity": self.physx_viscosity,
            "physx_cohesion": self.physx_cohesion,
            "physx_adhesion": self.physx_adhesion,
            "physx_surface_tension": self.physx_surface_tension,
            "physx_friction": self.physx_friction,
            "physx_damping": self.physx_damping,
        }
        for field_name, value in nonnegative_physx_values.items():
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a non-negative finite number")

        if self.spreading_ratio < 1.0 or not math.isfinite(self.spreading_ratio):
            raise ValueError(
                "spreading_ratio must be a finite number greater than or equal to one"
            )
        positive_time_constants = {
            "spreading_time_s": self.spreading_time_s,
            "shrinkage_time_s": self.shrinkage_time_s,
        }
        for field_name, value in positive_time_constants.items():
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a positive finite number")
        if (
            not 0.0 <= self.shrinkage_fraction < 1.0
            or not math.isfinite(self.shrinkage_fraction)
        ):
            raise ValueError("shrinkage_fraction must be finite and in [0, 1)")

    @property
    def filament_area_mm2(self) -> float:
        if self.filament_diameter_mm is None:
            raise ValueError("filament area is unavailable for this material profile")
        radius = self.filament_diameter_mm * 0.5
        return math.pi * radius * radius

    def volume_mm3(self, e_delta_mm: float) -> float:
        positive_delta = max(0.0, e_delta_mm)
        if self.extrusion_mode == "volumetric":
            return positive_delta * self.flow_multiplier
        if self.extrusion_mode == "syringe_plunger":
            radius = float(self.syringe_inner_diameter_mm) * 0.5
            plunger_area_mm2 = math.pi * radius * radius
            return positive_delta * plunger_area_mm2 * self.flow_multiplier
        return positive_delta * self.filament_area_mm2 * self.flow_multiplier

    def mass_g(self, e_delta_mm: float) -> float | None:
        if self.density_g_cm3 is None:
            return None
        return self.volume_mm3(e_delta_mm) * self.density_g_cm3 / 1000.0


def material_profile_from_dict(
    data: Mapping[str, Any],
    *,
    source: str = "material profile",
) -> MaterialProfile:
    """Build a strict profile from JSON-compatible data."""

    unknown_fields = sorted(set(data) - MATERIAL_PROFILE_FIELDS)
    if unknown_fields:
        raise ValueError(f"{source} has unknown fields: {unknown_fields}")
    required_fields = {"profile_id", "name", "extrusion_mode"}
    missing_fields = sorted(required_fields - set(data))
    if missing_fields:
        raise ValueError(f"{source} is missing required fields: {missing_fields}")

    def optional_float(field_name: str) -> float | None:
        value = data.get(field_name)
        return None if value is None else float(value)

    return MaterialProfile(
        profile_id=str(data["profile_id"]),
        name=str(data["name"]),
        extrusion_mode=str(data["extrusion_mode"]),
        filament_diameter_mm=optional_float("filament_diameter_mm"),
        syringe_inner_diameter_mm=optional_float("syringe_inner_diameter_mm"),
        flow_multiplier=float(data.get("flow_multiplier", 1.0)),
        density_g_cm3=optional_float("density_g_cm3"),
        physx_particle_contact_offset_m=float(
            data.get("physx_particle_contact_offset_m", 0.0005)
        ),
        physx_viscosity=float(data.get("physx_viscosity", 1000.0)),
        physx_cohesion=float(data.get("physx_cohesion", 5.0)),
        physx_adhesion=float(data.get("physx_adhesion", 10.0)),
        physx_surface_tension=float(data.get("physx_surface_tension", 0.02)),
        physx_friction=float(data.get("physx_friction", 1000.0)),
        physx_damping=float(data.get("physx_damping", 0.99)),
        spreading_ratio=float(data.get("spreading_ratio", 1.0)),
        spreading_time_s=float(data.get("spreading_time_s", 1.0)),
        shrinkage_fraction=float(data.get("shrinkage_fraction", 0.0)),
        shrinkage_time_s=float(data.get("shrinkage_time_s", 1.0)),
    )


def load_material_profile(path: str | Path) -> MaterialProfile:
    """Load and validate one standalone material-profile JSON file."""

    profile_path = Path(path)
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"material profile {profile_path} must contain a JSON object")
    profile = material_profile_from_dict(data, source=str(profile_path))
    if profile.profile_id != profile_path.stem:
        raise ValueError(
            f"profile_id {profile.profile_id!r} must match filename {profile_path.name!r}"
        )
    return profile


@dataclass(frozen=True)
class ExtrusionSample:
    material: str
    e_delta_mm: float
    volume_mm3: float
    mass_g: float | None

    @property
    def active(self) -> bool:
        return self.volume_mm3 > 0.0


def apply_material_profile(
    e_delta_mm: float,
    material: MaterialProfile,
) -> ExtrusionSample:
    """Apply an explicitly resolved profile to one G-code extrusion delta."""

    return ExtrusionSample(
        material=material.name,
        e_delta_mm=e_delta_mm,
        volume_mm3=material.volume_mm3(e_delta_mm),
        mass_g=material.mass_g(e_delta_mm),
    )
