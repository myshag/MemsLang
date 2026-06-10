"""Anisotropic wet etch (KOH / TMAH) of single-crystal silicon -- the
bulk-micromachining sibling of the Bosch DRIE module.

Same level-set core (``etch_core``), but the etch velocity is *orientation
dependent*: the rate is a function of the angle between the local surface
normal and the crystal axes.  On a (100) wafer the {111} planes are ~100x
slower than {100} and meet the surface at 54.74 deg, so a mask opening etches
into a self-terminating facetted pit / V-groove bounded by {111} -- the
classic anisotropic-etch geometry (V-grooves, membranes, pyramidal pits).
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import etch as _etch                       # reuse the compiled core

ANG_111_ON_100 = 54.74      # deg: {111}^{100} dihedral angle
ANG_111_ON_110 = 35.26      # deg: {111} on a (110) wafer (V-grooves)


@dataclass
class WetEtch:
    """KOH/TMAH recipe as effective plane rates (um per time unit)."""
    r100: float = 1.0           # {100} etch rate (fast)
    r110: float = 0.6           # {110} etch rate
    r111: float = 0.012         # {111} etch rate (slow; ~R100/80)
    facet_angle: float = ANG_111_ON_100   # wafer/orientation dependent
    notch_w: float = 9.0        # deg width of the slow-plane notch
    selectivity: float = 200.0  # Si:mask (nitride/oxide mask)
    steps: int = 320


@dataclass
class WetResult:
    phi: List[float]
    nx: int
    nz: int
    dx: float
    j0: int
    facet_angle: float = 0.0    # measured sidewall angle from horizontal
    depth: float = 0.0
    self_terminated: bool = False

    def report(self) -> List[str]:
        return [
            "wet    KOH/TMAH level-set: %d x %d cells @ %.3f um" % (
                self.nx, self.nz, self.dx),
            "wet    pit depth %.1f um, sidewall %.1f deg from surface%s" % (
                self.depth, self.facet_angle,
                " (V-groove, self-terminated)" if self.self_terminated else "")]


def _load():
    lib = _etch._load_lib()
    if lib is not None and not hasattr(lib, "_wet_set"):
        lib.wet_run.restype = None
        lib.wet_run.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
        lib._wet_set = True
    return lib


def simulate(openings: List[Tuple[float, float]], domain_w: float,
             depth: float, dx: float = 0.2,
             recipe: Optional[WetEtch] = None,
             mask_thick: float = 1.0) -> WetResult:
    """Etch mask ``openings`` (list of (x0, x1) um) into a (100) wafer."""
    recipe = recipe or WetEtch()
    nx = max(8, int(round(domain_w / dx)))
    nz = max(8, int(round((depth + mask_thick + 4 * dx) / dx)))
    j0 = int(round((mask_thick + 2 * dx) / dx))
    phi = [0.0] * (nx * nz)
    mask = bytearray(nx * nz)
    mt = int(round(mask_thick / dx))

    def is_open(x):
        return any(x0 <= x <= x1 for (x0, x1) in openings)

    for j in range(nz):
        for i in range(nx):
            x = (i + 0.5) * dx
            opn = is_open(x)
            surf = j0 if opn else (j0 - mt)
            phi[i + nx * j] = (surf - j) * dx
            if not opn and (j0 - mt) <= j < j0:
                mask[i + nx * j] = 1

    lib = _load()
    dt = 0.30 * dx
    if lib is not None:
        arr = (ctypes.c_double * len(phi))(*phi)
        mk = (ctypes.c_ubyte * len(mask)).from_buffer_copy(bytes(mask))
        lib.wet_run(arr, mk, nx, nz, dx, recipe.r100, recipe.r111,
                    recipe.r110, recipe.facet_angle, recipe.notch_w,
                    recipe.selectivity, recipe.steps, dt, 2)
        phi = list(arr)
    else:
        raise RuntimeError("wet etch requires the compiled C core")

    res = WetResult(phi, nx, nz, dx, j0)
    _measure(res)
    return res


def _measure(res: WetResult):
    nx, nz, dx, j0, phi = res.nx, res.nz, res.dx, res.j0, res.phi
    ic = nx // 2
    jdeep = j0
    for j in range(j0, nz):
        if phi[ic + nx * j] > 0:
            jdeep = j
    res.depth = (jdeep - j0) * dx
    # sidewall angle: trace the left wall x vs depth, fit a line, convert
    js, xs = [], []
    for j in range(j0 + 2, jdeep - 1):
        wall = None
        for i in range(1, ic):
            if phi[i - 1 + nx * j] < 0 and phi[i + nx * j] > 0:
                wall = i
        if wall is not None:
            js.append(float(j))
            xs.append(wall * dx)
    if len(js) >= 4:
        n = len(js)
        mj, mx = sum(js) / n, sum(xs) / n
        sjj = sum((j - mj) ** 2 for j in js) or 1.0
        # slope = d(x_um)/d(j_cells); one cell of depth is dx um, so the
        # angle from the horizontal surface is atan2(dz_um, dx_wall_um)
        slope = sum((j - mj) * (x - mx) for j, x in zip(js, xs)) / sjj
        res.facet_angle = math.degrees(math.atan2(dx, abs(slope) + 1e-9))
    # self-terminated (V-groove) if the floor has shrunk to ~a point
    jb = min(jdeep, nz - 1)
    floor = sum(1 for i in range(nx) if phi[i + nx * jb] > 0)
    res.self_terminated = floor * dx < 2.0
