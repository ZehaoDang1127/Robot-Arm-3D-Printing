"""
Stage 1 of the robotic 3D-printing pipeline: G-code parser.

Turns Cura / Marlin-flavour G-code into a flat list of motion primitives
(`Move`) that the rest of the pipeline consumes. It runs a small modal state
machine so it correctly handles the things that trip up naive line-by-line
parsers:

  * absolute vs relative positioning            (G90 / G91)
  * absolute vs relative extrusion              (M82 / M83)
  * position resets without motion              (G92, e.g. `G92 E0`)
  * homing                                      (G28)
  * units                                       (G21 mm  /  G20 inch)
  * Cura layer comments                         (`;LAYER:n`, `;LAYER_COUNT:n`)
  * retractions / wipes / Z-hops                (treated as travel, de <= 0)

Coordinates stay in the PRINTER frame, in millimetres. Transforming into the
robot base frame, densifying the segments, and assigning nozzle poses all
happen in Stage 2 — this module deliberately does none of that, so it stays a
single, testable source of truth for "what did the slicer actually ask for".

The downstream contract:
  - Each `Move` is the ABSOLUTE TARGET of one motion command. The START of the
    move is the target of the previous move (use `iter_segments()` to get
    (start, end) pairs directly).
  - `de` is the filament delta for THAT move only (mm). de > 0  => deposition.
  - `is_print` is derived purely from `de > 0`, so retractions (de < 0) and
    pure travels (de == 0) are correctly classified regardless of G0/G1.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

# Matches a single axis token like "X12.3", "E-1.27", "F3000", ".5"
_TOKEN = re.compile(r"^([A-Za-z])(-?\d*\.?\d+)$")
_LAYER_RE = re.compile(r";LAYER:\s*(-?\d+)")
_LAYER_COUNT_RE = re.compile(r";LAYER_COUNT:\s*(\d+)")

_EPS = 1e-9


@dataclass
class Move:
    """One motion command, resolved to an absolute target in the printer frame."""
    x: float            # absolute target, mm
    y: float
    z: float
    e: float            # absolute extruder position after this move, mm
    de: float           # filament extruded on THIS move, mm (>0 deposit, <0 retract)
    f: float            # modal feedrate active for this move, mm/min
    has_e: bool         # True if this command included an explicit E word
    is_print: bool      # True iff de > 0
    layer: int          # 0-based layer index (Cura comment if present, else z-inferred)
    rapid: bool         # True if issued as G0 (rapid) rather than G1

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def feed_m_s(self) -> float:
        """Feedrate in m/s (G-code F is mm/min)."""
        return self.f / 60_000.0


@dataclass
class ParseResult:
    moves: list[Move]
    layer_count: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    n_print: int
    n_travel: int
    units: str
    used_z_inference: bool

    def iter_segments(self):
        """Yield (start_xyz, end_xyz, move) for every move after the first.

        `start` is the previous move's target, so this gives you the actual
        line segments to densify in Stage 2.
        """
        prev = None
        for mv in self.moves:
            if prev is not None:
                yield prev.xyz, mv.xyz, mv
            prev = mv

    def summary(self) -> str:
        lo, hi = self.bbox_min, self.bbox_max
        return (
            f"{len(self.moves)} moves  ({self.n_print} print, {self.n_travel} travel)\n"
            f"layers: {self.layer_count}"
            f"{'  [z-inferred]' if self.used_z_inference else '  [from ;LAYER comments]'}\n"
            f"units : {self.units}\n"
            f"bbox  : X[{lo[0]:.2f}, {hi[0]:.2f}]  "
            f"Y[{lo[1]:.2f}, {hi[1]:.2f}]  Z[{lo[2]:.2f}, {hi[2]:.2f}] mm"
        )


def parse_gcode(path: str | Path, *, infer_layers_by_z: bool = True) -> ParseResult:
    """Parse a Cura/Marlin G-code file into a ParseResult.

    Args:
        path: path to the .gcode file.
        infer_layers_by_z: if the file has no ;LAYER comments, increment the
            layer index whenever a print move reaches a new maximum Z.
    """
    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    e_cur = 0.0
    abs_pos = True       # G90 is the Cura/Marlin default
    abs_e = True         # M82 (absolute extrusion) assumed until told otherwise
    feed = 0.0
    layer = -1
    units = "mm"
    scale = 1.0          # mm per G-code unit (25.4 if G20/inch)
    z_max = -math.inf

    moves: list[Move] = []
    seen_layer_comment = False
    declared_layer_count: int | None = None

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue

        # --- comments: extract Cura layer metadata, then skip ----------------
        if line.startswith(";"):
            m = _LAYER_RE.match(line)
            if m:
                layer = int(m.group(1))
                seen_layer_comment = True
            mc = _LAYER_COUNT_RE.match(line)
            if mc:
                declared_layer_count = int(mc.group(1))
            continue

        code = line.split(";", 1)[0].strip()   # strip trailing inline comment
        if not code:
            continue

        parts = code.split()
        word = parts[0].upper()
        params: dict[str, float] = {}
        for p in parts[1:]:
            tm = _TOKEN.match(p)
            if tm:
                params[tm.group(1).upper()] = float(tm.group(2))

        # --- modal / setup commands ------------------------------------------
        if word == "G20":
            units, scale = "inch", 25.4
        elif word == "G21":
            units, scale = "mm", 1.0
        elif word == "G90":
            abs_pos = True
        elif word == "G91":
            abs_pos = False
        elif word == "M82":
            abs_e = True
        elif word == "M83":
            abs_e = False
        elif word == "G92":
            # set current position(s) WITHOUT moving (commonly `G92 E0`)
            for ax in ("X", "Y", "Z"):
                if ax in params:
                    pos[ax] = params[ax] * scale
            if "E" in params:
                e_cur = params["E"] * scale
        elif word == "G28":
            # home: listed axes -> 0, or all axes if none listed
            homed = [a for a in ("X", "Y", "Z") if a in params] or ["X", "Y", "Z"]
            for ax in homed:
                pos[ax] = 0.0

        # --- motion commands -------------------------------------------------
        elif word in ("G0", "G1"):
            target = dict(pos)
            for ax in ("X", "Y", "Z"):
                if ax in params:
                    v = params[ax] * scale
                    target[ax] = v if abs_pos else pos[ax] + v
            if "F" in params:
                feed = params["F"]          # mm/min, modal

            de = 0.0
            if "E" in params:
                e_val = params["E"] * scale
                if abs_e:
                    de = e_val - e_cur
                    e_cur = e_val
                else:
                    de = e_val
                    e_cur += e_val

            is_print = de > _EPS

            if infer_layers_by_z and not seen_layer_comment:
                if is_print and target["Z"] > z_max + 1e-6:
                    z_max = target["Z"]
                    layer += 1

            moves.append(Move(
                x=target["X"], y=target["Y"], z=target["Z"],
                e=e_cur, de=de, f=feed, has_e=("E" in params), is_print=is_print,
                layer=max(layer, 0), rapid=(word == "G0"),
            ))
            pos = target
        # everything else (M104/M109 temps, fan, etc.) is ignored on purpose

    # --- aggregate metadata --------------------------------------------------
    if moves:
        xs = [m.x for m in moves]; ys = [m.y for m in moves]; zs = [m.z for m in moves]
        bbox_min = (min(xs), min(ys), min(zs))
        bbox_max = (max(xs), max(ys), max(zs))
        n_print = sum(m.is_print for m in moves)
        if seen_layer_comment:
            layer_count = (declared_layer_count
                           if declared_layer_count is not None
                           else max(m.layer for m in moves) + 1)
        else:
            layer_count = max(m.layer for m in moves) + 1
    else:
        bbox_min = bbox_max = (0.0, 0.0, 0.0)
        n_print = 0
        layer_count = 0

    return ParseResult(
        moves=moves,
        layer_count=layer_count,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        n_print=n_print,
        n_travel=len(moves) - n_print,
        units=units,
        used_z_inference=(not seen_layer_comment),
    )


if __name__ == "__main__":
    import sys
    res = parse_gcode(sys.argv[1])
    print(res.summary())
