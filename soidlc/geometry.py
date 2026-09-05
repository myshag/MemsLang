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
    # A "band" is a ring whose single hole samples the same angles as its
    # exterior, so the mesher can cap it with quads instead of cutting a
    # bridge slit (which collapses into a non-manifold edge).  Every
    # transform below must carry the flag: elaborate translates, rotates and
    # mirrors each shape during placement, and a ring that lost it would
    # silently fall back to the broken path.
    band: bool = False

    def normalized(self) -> "Polygon":
        return Polygon(ensure_ccw(self.exterior),
                       [ensure_cw(h) for h in self.holes], self.band)

    def translated(self, dx: float, dy: float) -> "Polygon":
        return Polygon(
            [(x + dx, y + dy) for x, y in self.exterior],
            [[(x + dx, y + dy) for x, y in h] for h in self.holes],
            self.band,
        )

    def rotated(self, deg: float) -> "Polygon":
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        # exact trig for multiples of 90 deg keeps rectilinear geometry exact
        if abs(c - round(c)) < 1e-12:
            c = float(round(c))
        if abs(s - round(s)) < 1e-12:
            s = float(round(s))

        def rot(r: Ring) -> Ring:
            return [(x * c - y * s, x * s + y * c) for x, y in r]

        return Polygon(rot(self.exterior), [rot(h) for h in self.holes],
                       self.band)

    def mirrored(self, mx: bool, my: bool) -> "Polygon":
        fx = -1.0 if mx else 1.0
        fy = -1.0 if my else 1.0

        def m(r: Ring) -> Ring:
            return [(x * fx, y * fy) for x, y in r]

        return Polygon(m(self.exterior), [m(h) for h in self.holes],
                       self.band)

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


def wire(points: List[Pt], w: float) -> Polygon:
    """A polyline of width `w` as a closed polygon, with mitred joins.

    The miter is clipped to a bevel once it would reach past 4x the half-width,
    so a very sharp corner produces a blunt end rather than a spike that
    self-intersects (which would break the extruder).
    """
    pts: List[Pt] = []
    for p in points:
        if not pts or (abs(p[0] - pts[-1][0]) > 1e-9
                       or abs(p[1] - pts[-1][1]) > 1e-9):
            pts.append((float(p[0]), float(p[1])))
    if len(pts) < 2:
        raise ValueError("wire() needs at least two distinct points")

    half = w / 2.0

    def unit(a: Pt, b: Pt) -> Pt:
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        return (dx / L, dy / L)

    offsets: List[Pt] = []
    n = len(pts)
    for i in range(n):
        if i == 0:
            d = unit(pts[0], pts[1])
            offsets.append((-d[1] * half, d[0] * half))
        elif i == n - 1:
            d = unit(pts[-2], pts[-1])
            offsets.append((-d[1] * half, d[0] * half))
        else:
            d0 = unit(pts[i - 1], pts[i])
            d1 = unit(pts[i], pts[i + 1])
            n0 = (-d0[1], d0[0])
            n1 = (-d1[1], d1[0])
            mx, my = n0[0] + n1[0], n0[1] + n1[1]
            L = math.hypot(mx, my)
            if L < 1e-12:          # 180-degree reversal: keep the incoming normal
                offsets.append((n0[0] * half, n0[1] * half))
                continue
            mx, my = mx / L, my / L
            cos_half = mx * n0[0] + my * n0[1]
            scale = half / max(cos_half, 0.25)     # bevel clip at 4x half-width
            offsets.append((mx * scale, my * scale))

    left = [(p[0] + o[0], p[1] + o[1]) for p, o in zip(pts, offsets)]
    right = [(p[0] - o[0], p[1] - o[1]) for p, o in zip(pts, offsets)]
    return Polygon(left + list(reversed(right))).normalized()


def circle(R: float, n_seg: int = 64,
           cx: float = 0.0, cy: float = 0.0) -> Polygon:
    """A regular-polygon approximation of a disk, wound CCW."""
    if n_seg < 3:
        raise ValueError("circle() needs at least 3 segments")
    ring = [(cx + R * math.cos(2 * math.pi * i / n_seg),
             cy + R * math.sin(2 * math.pi * i / n_seg))
            for i in range(n_seg)]
    return Polygon(ring)


def annulus(R: float, w: float, n_seg: int = 64,
            cx: float = 0.0, cy: float = 0.0) -> Polygon:
    """A ring of centreline radius R and radial width w.

    The hole samples the SAME angles as the exterior, so the two rings pair up
    and the mesher can cap the band with quads (see mesh._band_caps).  Marked
    band=True to say so.
    """
    if w <= 0 or w >= 2 * R:
        raise ValueError("annulus() needs 0 < w < 2R")
    outer = circle(R + w / 2.0, n_seg, cx, cy).exterior
    inner = circle(R - w / 2.0, n_seg, cx, cy).exterior
    return Polygon(list(outer), [list(reversed(inner))], band=True)


def arc(R: float, w: float, a0_deg: float, a1_deg: float, n_seg: int = 64,
        cx: float = 0.0, cy: float = 0.0) -> Polygon:
    """A partial band from a0 to a1 degrees: a simple ring, no hole."""
    a0 = math.radians(a0_deg)
    a1 = math.radians(a1_deg)
    k = max(2, int(round(n_seg * abs(a1_deg - a0_deg) / 360.0)) + 1)
    ang = [a0 + (a1 - a0) * i / (k - 1) for i in range(k)]
    ro, ri = R + w / 2.0, R - w / 2.0
    outer = [(cx + ro * math.cos(a), cy + ro * math.sin(a)) for a in ang]
    inner = [(cx + ri * math.cos(a), cy + ri * math.sin(a)) for a in ang]
    return Polygon(outer + list(reversed(inner))).normalized()


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
    owner: str = ""          # device-level instance this shape belongs to


def bbox_of(shapes: List[Shape]) -> Tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for sh in shapes:
        x0, y0, x1, y1 = sh.polygon.bbox()
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    if not xs0:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs0), min(ys0), max(xs1), max(ys1)
