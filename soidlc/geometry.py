"""2D polygon geometry (in micrometres).

A :class:`Polygon` has an outer ring plus optional holes.  Rings are lists of
``(x, y)`` tuples.  Coordinates are plain floats in micrometres; the unit
system is enforced one level up, in elaboration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

Pt = Tuple[float, float]
Ring = List[Pt]


def signed_area(ring: Ring) -> float:
    a = 0.0
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def ensure_ccw(ring: Ring) -> Ring:
    return ring if signed_area(ring) >= 0 else list(reversed(ring))


def ensure_cw(ring: Ring) -> Ring:
    return ring if signed_area(ring) <= 0 else list(reversed(ring))


@dataclass
class Polygon:
    exterior: Ring
    holes: List[Ring] = field(default_factory=list)

    def normalized(self) -> "Polygon":
        return Polygon(ensure_ccw(self.exterior),
                       [ensure_cw(h) for h in self.holes])

    def translated(self, dx: float, dy: float) -> "Polygon":
        return Polygon(
            [(x + dx, y + dy) for x, y in self.exterior],
            [[(x + dx, y + dy) for x, y in h] for h in self.holes],
        )

    def rotated(self, deg: float) -> "Polygon":
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))

        def rot(r: Ring) -> Ring:
            return [(x * c - y * s, x * s + y * c) for x, y in r]

        return Polygon(rot(self.exterior), [rot(h) for h in self.holes])

    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.exterior]
        ys = [p[1] for p in self.exterior]
        return min(xs), min(ys), max(xs), max(ys)

    def area(self) -> float:
        a = abs(signed_area(self.exterior))
        for h in self.holes:
            a -= abs(signed_area(h))
        return a


def rect(w: float, h: float, cx: float = 0.0, cy: float = 0.0) -> Polygon:
    """Axis-aligned rectangle of width ``w`` / height ``h`` centred at (cx,cy)."""
    hw, hh = w / 2.0, h / 2.0
    return Polygon([
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
    ])


def rect_corner(x0: float, y0: float, w: float, h: float) -> Polygon:
    return Polygon([
        (x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h),
    ])


@dataclass
class Shape:
    """A 2D polygon tagged with the fabrication layer it belongs to."""

    layer: str
    polygon: Polygon
    label: str = ""
    mech: str = "released"   # "anchored" | "released" — drives release analysis


def bbox_of(shapes: List[Shape]) -> Tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for sh in shapes:
        x0, y0, x1, y1 = sh.polygon.bbox()
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    if not xs0:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs0), min(ys0), max(xs1), max(ys1)
