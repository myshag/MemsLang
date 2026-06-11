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
        lib.wet_run_3d.restype = None
        lib.wet_run_3d.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_double, ctypes.c_int]
        lib.wet_run_3d_tab.restype = None
        lib.wet_run_3d_tab.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
        lib.wet_run_3d_film.restype = None
        lib.wet_run_3d_film.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_int]
        lib._wet_set = True
    return lib


@dataclass
class WetResult3D:
    phi: List[float]
    nx: int
    ny: int
    nz: int
    dx: float
    j0: int
    film: Optional[List[float]] = None     # remaining film thickness per column

    def film_at(self, x: float, y: float) -> float:
        if self.film is None:
            return 0.0
        ix = min(self.nx - 1, max(0, int(x / self.dx)))
        jy = min(self.ny - 1, max(0, int(y / self.dx)))
        return self.film[ix + self.nx * jy]


def simulate_3d(mask_open, domain_w: float, domain_h: float, depth: float,
                dx: float = 0.5, recipe: Optional[WetEtch] = None,
                mask_thick: float = 1.0, diagram=None,
                orientation: str = "100", misalign_deg: float = 0.0,
                miscut_deg: float = 0.0, miscut_az: float = 0.0,
                film_mask: bool = False) -> WetResult3D:
    """3D anisotropic etch.  ``mask_open(x, y) -> bool`` is the mask opening
    over the (domain_w x domain_h) wafer surface (um).

    By default the closed-form {111}-notch model (``recipe``) drives the etch.
    Pass a ``crystal_rates.RateDiagram`` as ``diagram`` (with a wafer
    ``orientation`` of "100"/"110"/"111") to drive it from a calibrated,
    measured anisotropy map sampled into a lookup table instead.
    ``misalign_deg`` turns the mask off the wafer flat (<110>); ``miscut_deg``/
    ``miscut_az`` tilt the surface off the ideal pole (an off-axis wafer).

    ``film_mask`` switches from the default whole-column protection to a thin
    masking film (thickness ``mask_thick``) of finite ``recipe.selectivity``:
    the film alone is the slow-etching guard.  It protects the silicon top while
    it lasts, thins at the vertical rate / selectivity, and once consumed the
    silicon is exposed and etches (mask failure) -- the result carries the
    remaining film thickness per column (``WetResult3D.film``).  Silicon is also
    undercut laterally from the opening edges (the released-membrane regime);
    note this single-field model does not reproduce the inclined-{111}
    self-limiting that spares a perfectly <110>-aligned edge -- that needs
    explicit facet tracking."""
    recipe = recipe or WetEtch()
    nx = max(8, int(round(domain_w / dx)))
    ny = max(8, int(round(domain_h / dx)))
    nz = max(8, int(round((depth + mask_thick + 4 * dx) / dx)))
    j0 = int(round((mask_thick + 2 * dx) / dx))
    mt = int(round(mask_thick / dx))
    n = nx * ny * nz
    lib = _load()
    if lib is None:
        raise RuntimeError("3D wet etch requires the compiled C core")
    tab = None
    if diagram is not None:
        from . import crystal_rates as _cr
        ntheta, nphi = 90, 180
        tab_list, _rmax = _cr.sample_table(diagram, orientation, ntheta, nphi,
                                           misalign_deg, miscut_deg, miscut_az)
        tab = (ctypes.c_double * len(tab_list))(*tab_list)

    if film_mask:
        if tab is None:
            raise ValueError("film_mask requires a crystal_rates diagram")
        # silicon only: flat top at j0 (phi>0 = etchant/air above, <0 = Si).
        # the mask lives in a separate per-column thickness field mfilm.
        phi = (ctypes.c_double * n)()
        mfilm = (ctypes.c_double * (nx * ny))()
        for jy in range(ny):
            for ix in range(nx):
                opn = mask_open((ix + 0.5) * dx, (jy + 0.5) * dx)
                mfilm[ix + nx * jy] = 0.0 if opn else mask_thick
                for k in range(nz):
                    phi[ix + nx * (jy + ny * k)] = (j0 - k) * dx
        lib.wet_run_3d_film(phi, nx, ny, nz, dx, tab, ntheta, nphi, mfilm,
                            recipe.selectivity, recipe.steps, 0.30 * dx, 2)
        return WetResult3D(list(phi), nx, ny, nz, dx, j0, list(mfilm))

    # solid mask carved into phi (default whole-column protection)
    phi = (ctypes.c_double * n)()
    mask = bytearray(n)
    for k in range(nz):
        for jy in range(ny):
            for ix in range(nx):
                opn = mask_open((ix + 0.5) * dx, (jy + 0.5) * dx)
                surf = j0 if opn else (j0 - mt)
                idx = ix + nx * (jy + ny * k)
                phi[idx] = (surf - k) * dx
                if not opn and k >= (j0 - mt):
                    mask[idx] = 1
    mk = (ctypes.c_ubyte * n).from_buffer_copy(bytes(mask))
    if tab is not None:
        lib.wet_run_3d_tab(phi, mk, nx, ny, nz, dx, tab, ntheta, nphi,
                           recipe.selectivity, recipe.steps, 0.30 * dx, 2)
    else:
        lib.wet_run_3d(phi, mk, nx, ny, nz, dx, recipe.r100, recipe.r111,
                       recipe.notch_w, recipe.selectivity, recipe.steps,
                       0.30 * dx, 2)
    return WetResult3D(list(phi), nx, ny, nz, dx, j0)


def heightmap_mesh(res: WetResult3D):
    """Extract the etch-front depth z(x,y) and build a 3D surface mesh
    (soidlc.mesh.Mesh) of the pit plus the surrounding wafer top."""
    from . import mesh as M
    nx, ny, nz, dx, j0 = res.nx, res.ny, res.nz, res.dx, res.j0
    phi = res.phi
    depth = [[0.0] * nx for _ in range(ny)]
    for jy in range(ny):
        for ix in range(nx):
            kd = j0
            for k in range(j0, nz):
                if phi[ix + nx * (jy + ny * k)] > 0:
                    kd = k
            depth[jy][ix] = (kd - j0) * dx          # um below surface
    out = M.Mesh()
    g = "DEVICE"
    for jy in range(ny - 1):
        for ix in range(nx - 1):
            x0, x1 = ix * dx, (ix + 1) * dx
            y0, y1 = jy * dx, (jy + 1) * dx
            z00 = depth[jy][ix]; z10 = depth[jy][ix + 1]
            z11 = depth[jy + 1][ix + 1]; z01 = depth[jy + 1][ix]
            a = out.add_vertex((x0, y0, -z00)); b = out.add_vertex((x1, y0, -z10))
            c = out.add_vertex((x1, y1, -z11)); d = out.add_vertex((x0, y1, -z01))
            out.add_triangle(a, b, c, g); out.add_triangle(a, c, d, g)
    return out


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
