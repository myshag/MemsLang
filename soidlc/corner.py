"""Convex-corner undercut and corner-compensation structures for anisotropic
(KOH / TMAH) wet etching.

Bulk micromachining on a (100) wafer is self-terminating on the slow {111}
planes, so *concave* features (the inside corners of an etch pit) stay sharp.
But a *convex* corner -- the outside corner of a mesa, boss or cantilever base
you want to keep -- is attacked by fast intermediate planes (~{411}/{311}) at
the 45 deg bisector and is beveled/undercut by roughly

        U  ~=  ratio * depth          (ratio ~ 1.4 .. 1.8 for KOH)

A *corner compensation structure* is extra, sacrificial mask added at the
convex corner -- a <100>-oriented bar, a square, or a beam -- dimensioned so
the fast-plane undercut eats the sacrifice and only just reaches the true
corner at the target depth, leaving it sharp.

This module simulates the lateral mask undercut directly: a plan-view
(top-down) level-set whose front speed depends on the in-plane direction of
the front normal -- ~zero along <110> (a {111} sidewall forms) and fast along
<100> (the convex-corner planes).  Run it on a mask with and without a
compensation structure and measure whether the protected corner survives.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from . import etch as _etch                       # reuse the compiled core

Rect = Tuple[float, float, float, float]          # (x0, y0, x1, y1)


@dataclass
class UndercutEtch:
    """Lateral mask-undercut model for a timed (100) anisotropic etch.

    ``ratio`` is the convex-corner undercut per unit depth (the {411}/{100}
    ratio); ``depth`` is the target etch depth.  The fast corner therefore
    advances ``ratio * depth`` microns laterally during the etch.
    """
    depth: float = 30.0          # um: target etch depth
    ratio: float = 1.6           # convex-corner lateral undercut per depth
    r_slow: float = 0.03         # <110>-edge undercut, relative to r_peak
    r100: float = 0.55           # <100>-sidewall undercut, relative to r_peak
    notch_w: float = 10.0        # deg: angular width of the slow <110> notch
    dpeak: float = 28.0          # deg: angle of the fastest (<410>) plane
    peak_w: float = 14.0         # deg: angular width of the fast-plane peak
    dx: float = 0.5              # um: grid spacing

    @property
    def r_fast(self) -> float:
        """Peak (convex-corner) lateral undercut for this etch, in um."""
        return self.ratio * self.depth


@dataclass
class UCResult:
    phi: List[float]
    nx: int
    ny: int
    dx: float
    mask: Callable[[float, float], bool]
    recipe: UndercutEtch

    def is_silicon(self, x: float, y: float) -> bool:
        """True if silicon is still present at the top surface at (x, y)."""
        i = min(self.nx - 1, max(0, int(x / self.dx)))
        j = min(self.ny - 1, max(0, int(y / self.dx)))
        return self.phi[i + self.nx * j] < 0.0

    def undercut_along(self, x0: float, y0: float, dirx: float, diry: float,
                       maxd: Optional[float] = None) -> float:
        """How far the front has receded from (x0, y0) along a unit direction:
        the distance until silicon (phi<0) is first found."""
        n = math.hypot(dirx, diry) or 1.0
        ux, uy = dirx / n, diry / n
        maxd = maxd or (self.nx + self.ny) * self.dx
        t = 0.0
        while t < maxd:
            if self.is_silicon(x0 + ux * t, y0 + uy * t):
                return t
            t += 0.5 * self.dx
        return maxd


# ---------------------------------------------------------------------------
# mask construction: a mask is a list of (rotated) rectangles, OR-ed together
# ---------------------------------------------------------------------------
@dataclass
class _RotRect:
    cx: float
    cy: float
    w: float                     # full width  (along local u)
    h: float                     # full height (along local v)
    angle: float = 0.0           # deg, CCW

    def contains(self, x: float, y: float) -> bool:
        a = math.radians(self.angle)
        ca, sa = math.cos(a), math.sin(a)
        dx, dy = x - self.cx, y - self.cy
        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca
        return abs(u) <= self.w / 2 and abs(v) <= self.h / 2


@dataclass
class Mask:
    """A protected-silicon mask as a union of (possibly rotated) rectangles."""
    rects: List[_RotRect] = field(default_factory=list)

    def add_rect(self, x0: float, y0: float, x1: float, y1: float) -> "Mask":
        self.rects.append(_RotRect((x0 + x1) / 2, (y0 + y1) / 2,
                                   abs(x1 - x0), abs(y1 - y0), 0.0))
        return self

    def add(self, r: _RotRect) -> "Mask":
        self.rects.append(r)
        return self

    def __call__(self, x: float, y: float) -> bool:
        return any(r.contains(x, y) for r in self.rects)


# ---------------------------------------------------------------------------
# corner compensation primitives -- extra sacrificial mask at a convex corner
# ---------------------------------------------------------------------------
def square_comp(corner: Tuple[float, float], size: float) -> _RotRect:
    """A <110>-aligned square centred on a convex corner -- the simplest, most
    robust compensation.  Its sidewalls run along <110> (slow {111}, they hold
    their line); only its outer convex corner is beveled by the fast plane, so
    it survives until that bevel reaches the real corner.  Protects when
    ``size > sqrt(2) * undercut``."""
    return _RotRect(corner[0], corner[1], size, size, 0.0)


def bar_comp(corner: Tuple[float, float], outx: int, outy: int,
             length: float, width: float, root: float = 6.0) -> _RotRect:
    """A <100>-oriented beam (rotated 45 deg) projecting diagonally outward
    from a convex corner -- area-efficient but its <100> sidewalls also
    undercut, so it is sacrificial and must be wide enough.  ``root`` overlaps
    it into the structure so the connection is solid (not a point contact)."""
    a = math.radians(45.0)
    ux, uy = outx * math.cos(a), outy * math.sin(a)
    span = length + root
    # centre the beam so its inner end sits ``root`` inside the corner
    cx = corner[0] + ux * (span / 2 - root)
    cy = corner[1] + uy * (span / 2 - root)
    ang = 45.0 if outx * outy > 0 else -45.0
    return _RotRect(cx, cy, span, width, ang)


# ---------------------------------------------------------------------------
def _load():
    lib = _etch._load_lib()
    if lib is None:
        raise RuntimeError("corner-undercut simulation requires the C core")
    if not hasattr(lib, "_uc_set"):
        lib.uc_run.restype = None
        lib.uc_run.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
            ctypes.c_double, ctypes.c_int]
        lib._uc_set = True
    return lib


def simulate(mask: Callable[[float, float], bool], domain_w: float,
             domain_h: float, recipe: Optional[UndercutEtch] = None) -> UCResult:
    """Lateral undercut of ``mask`` over a (domain_w x domain_h) um surface."""
    recipe = recipe or UndercutEtch()
    dx = recipe.dx
    nx = max(8, int(round(domain_w / dx)))
    ny = max(8, int(round(domain_h / dx)))
    n = nx * ny
    phi = (ctypes.c_double * n)()
    for j in range(ny):
        for i in range(nx):
            x, y = (i + 0.5) * dx, (j + 0.5) * dx
            # phi < 0 = silicon under mask, phi > 0 = open / undercut void
            phi[i + nx * j] = -1.0 * dx if mask(x, y) else 1.0 * dx

    lib = _load()
    cfl = 0.3 * dx
    dt = cfl / max(recipe.r_fast, 1e-9)
    nsteps = max(1, int(math.ceil(recipe.r_fast / cfl)))
    lib.uc_run(phi, nx, ny, dx, recipe.r_fast,
               recipe.r100 * recipe.r_fast, recipe.r_slow * recipe.r_fast,
               recipe.notch_w, recipe.dpeak, recipe.peak_w, nsteps, dt, 2)
    return UCResult(list(phi), nx, ny, dx, mask, recipe)


# ---------------------------------------------------------------------------
# plan-view PNG: original mask outline (grey) over the final silicon (blue)
# ---------------------------------------------------------------------------
def render(res: UCResult, path: str, width_px: int = 480,
           marks: Optional[Sequence[Tuple[float, float]]] = None) -> None:
    from .render import write_png
    nx, ny, dx = res.nx, res.ny, res.dx
    W = width_px
    H = max(1, int(W * ny / nx))
    img = bytearray([18, 18, 24] * (W * H))
    for py in range(H):
        y = (py + 0.5) * ny * dx / H
        for px in range(W):
            x = (px + 0.5) * nx * dx / W
            i = min(nx - 1, int(x / dx))
            j = min(ny - 1, int(y / dx))
            sil = res.phi[i + nx * j] < 0.0
            masked = res.mask(x, y)
            if sil:
                col = (70, 120, 190)             # silicon that survived
            elif masked:
                col = (150, 70, 70)              # lost to undercut (was masked)
            else:
                col = (28, 28, 36)               # open field
            o = ((H - 1 - py) * W + px) * 3
            img[o:o + 3] = bytes(col)
    if marks:
        for (mx, my) in marks:
            px = int(mx / (nx * dx) * W)
            py = H - 1 - int(my / (ny * dx) * H)
            for ddx in range(-3, 4):
                for ddy in range(-3, 4):
                    qx, qy = px + ddx, py + ddy
                    if 0 <= qx < W and 0 <= qy < H and abs(ddx) + abs(ddy) <= 4:
                        o = (qy * W + qx) * 3
                        img[o:o + 3] = bytes((240, 220, 90))
    write_png(path, W, H, img)
