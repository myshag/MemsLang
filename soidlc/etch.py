"""Level-set Bosch / DRIE etch simulation (tier-2 process model).

Drives the C kernel in ``etch_core.c`` (auto-compiled with the system gcc on
first use; falls back to a slow pure-Python kernel if no compiler is present).
Given a mask cross-section and a Bosch recipe it evolves the silicon etch
front and reports the resulting profile: trench depth, sidewall scalloping,
aspect-ratio-dependent etch (ARDE / RIE-lag), taper, and footing at the buried
oxide.  These feed the manufacturability checks (a trench that does not reach
the BOX leaves its region anchored, not released).
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "etch_core.so")
_SRC = os.path.join(_HERE, "etch_core.c")
_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    need = (not os.path.exists(_SO)
            or os.path.getmtime(_SO) < os.path.getmtime(_SRC))
    if need:
        for cc in ("gcc", "cc", "clang"):
            try:
                subprocess.run(
                    [cc, "-O3", "-fopenmp", "-shared", "-fPIC",
                     _SRC, "-o", _SO, "-lm"],
                    check=True, capture_output=True)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
    if os.path.exists(_SO):
        lib = ctypes.CDLL(_SO)
        lib.bosch_run.restype = None
        lib.bosch_run.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
            ctypes.c_int, ctypes.c_int, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_int]
        _LIB = lib
    return _LIB


# the band/reinit-limited scheme advances the front at ~half of V*dt per
# cycle; this constant makes ``etch_per_cycle`` read as um/cycle
_RATE_CAL = 2.5


# ---------------------------------------------------------------------------
@dataclass
class BoschRecipe:
    etch_per_cycle: float = 0.6     # um of vertical advance per cycle
    cycles: int = 40
    r_ion: float = 1.0              # anisotropic (ion) rate, relative
    r_iso: float = 0.18             # isotropic (radical) rate, relative
    selectivity: float = 75.0       # Si:mask etch ratio
    footing: float = 1.5            # lateral boost at the oxide interface
    scallop_um: float = 0.12        # lateral isotropic bulge per cycle (um)
    passivation: float = 0.92       # sidewall protection (1 = perfect)
    bow: float = 0.0                # depth-increasing sidewall etch: >0 gives
                                    # negative taper / bowing, =0 positive taper
    reinit_stride: int = 2          # reinit every N substeps (drift control)
    arde_angles: int = 28           # rays for the visibility (ARDE) factor


@dataclass
class EtchResult:
    phi: List[float]
    nx: int
    nz: int
    dx: float                       # um/cell
    j_box: int
    openings: List[Tuple[float, float]]   # (x0, x1) of each mask opening, um
    depths: List[float] = field(default_factory=list)        # per opening, um
    widths: List[float] = field(default_factory=list)
    reached_box: List[bool] = field(default_factory=list)
    tapers: List[float] = field(default_factory=list)   # deg, + = narrowing
    scallop_um: float = 0.0
    footing_um: float = 0.0

    def report(self) -> List[str]:
        out = ["etch   Bosch level-set: %d x %d cells @ %.3f um, BOX at %.1f um"
               % (self.nx, self.nz, self.dx, self.j_box * self.dx)]
        for k, (x0, x1) in enumerate(self.openings):
            ar = self.depths[k] / max(self.widths[k], 1e-6)
            flag = "reached BOX" if self.reached_box[k] else \
                "ARDE-LIMITED (did not reach BOX -> region stays anchored)"
            tp = self.tapers[k]
            sign = "positive/narrowing" if tp > 0.3 else \
                ("negative/bowing" if tp < -0.3 else "vertical")
            out.append(
                "etch   opening %d: width %.1f um -> depth %.1f um "
                "(AR %.1f) taper %+.1f deg (%s) %s"
                % (k, self.widths[k], self.depths[k], ar, tp, sign, flag))
        out.append("etch   scalloping ~%.2f um, footing undercut ~%.2f um"
                   % (self.scallop_um, self.footing_um))
        return out


# ---------------------------------------------------------------------------
def simulate(openings: List[Tuple[float, float]], domain_w: float,
             depth: float, dx: float = 0.15,
             recipe: Optional[BoschRecipe] = None,
             mask_thick: float = 2.0, box: bool = True,
             box_at: Optional[float] = None) -> EtchResult:
    """Etch a set of mask ``openings`` (list of (x0, x1) in um) into a wafer.

    ``domain_w`` x ``depth`` is the simulated cross-section (um); the buried
    oxide sits at ``box_at`` (default: just below the target depth).
    """
    recipe = recipe or BoschRecipe()
    box_at = box_at if box_at is not None else depth
    z_extent = max(depth, box_at)
    nx = max(8, int(round(domain_w / dx)))
    nz = max(8, int(round((z_extent + mask_thick + 4 * dx) / dx)))
    j0 = int(round((mask_thick + 2 * dx) / dx))          # wafer top surface
    j_box = min(nz - 1, j0 + int(round(box_at / dx))) if box else nz + 10

    phi = [0.0] * (nx * nz)
    mask = bytearray(nx * nz)

    def is_open(x):
        return any(x0 <= x <= x1 for (x0, x1) in openings)

    for j in range(nz):
        for i in range(nx):
            x = (i + 0.5) * dx
            opn = is_open(x)
            # silicon top at j0; in covered columns a mask sits j0-mt..j0
            surf = j0 if opn else (j0 - int(round(mask_thick / dx)))
            phi[i + nx * j] = (surf - j) * dx          # <0 solid, >0 void
            if not opn and (j0 - int(round(mask_thick / dx))) <= j < j0:
                mask[i + nx * j] = 1

    lib = _load_lib()
    dt = 0.35 * dx
    n_aniso = max(1, int(round(_RATE_CAL * recipe.etch_per_cycle
                               / (recipe.r_ion * dt))))
    # isotropic burst sized to one scallop (absolute lateral bulge per cycle)
    n_iso = max(1, int(round(recipe.scallop_um
                             / (recipe.r_iso * dt + 1e-9))))
    vis_maxlen = (depth + mask_thick) * 1.2

    if lib is not None:
        arr = (ctypes.c_double * len(phi))(*phi)
        mk = (ctypes.c_ubyte * len(mask)).from_buffer_copy(bytes(mask))
        lib.bosch_run(arr, mk, nx, nz, dx, j_box,
                      recipe.r_ion, recipe.r_iso, recipe.selectivity,
                      recipe.footing, recipe.cycles, n_aniso, n_iso, dt,
                      recipe.reinit_stride, recipe.arde_angles, vis_maxlen,
                      recipe.passivation, recipe.bow, j0)
        phi = list(arr)
    else:
        phi = _py_fallback(phi, mask, nx, nz, dx, j_box, recipe,
                           n_aniso, n_iso, dt)

    res = EtchResult(phi, nx, nz, dx, j_box, openings)
    _measure(res, j0, mask_thick)
    return res


def _measure(res: EtchResult, j0: int, mask_thick: float):
    nx, nz, dx = res.nx, res.nz, res.dx
    phi = res.phi
    for (x0, x1) in res.openings:
        ic = int(round((0.5 * (x0 + x1)) / dx))
        ic = min(max(ic, 0), nx - 1)
        # deepest solid->void transition in this column
        jdeep = j0
        for j in range(j0, nz):
            if phi[ic + nx * j] > 0:
                jdeep = j
        depth = (jdeep - j0) * dx
        # measured width at mid-depth, searched only near this opening
        jm = j0 + int((jdeep - j0) * 0.5)
        margin = int((x1 - x0) / dx)
        ilo = max(0, int(x0 / dx) - margin)
        ihi = min(nx, int(x1 / dx) + margin)
        left = right = None
        for i in range(ilo, ihi):
            if phi[i + nx * jm] > 0:
                if left is None:
                    left = i
                right = i
        width = ((right - left) * dx) if (left is not None) else (x1 - x0)
        res.depths.append(depth)
        res.widths.append(max(width, x1 - x0))
        res.reached_box.append(jdeep >= res.j_box - 1)
        res.tapers.append(_taper(res, j0, ic, jdeep))
    # scallop and footing measured on the deepest (best-resolved) trench
    kd = max(range(len(res.depths)), key=lambda k: res.depths[k])
    res.scallop_um = _scallop(res, j0, kd)
    res.footing_um = _footing(res, j0, kd)


def _row_width(res, ic, j):
    """Void width of the trench containing column ic at row j (um)."""
    nx, nz, dx, phi = res.nx, res.nz, res.dx, res.phi
    if phi[ic + nx * j] <= 0:
        return 0.0
    li = ic
    while li > 0 and phi[li - 1 + nx * j] > 0:
        li -= 1
    ri = ic
    while ri < nx - 1 and phi[ri + 1 + nx * j] > 0:
        ri += 1
    return (ri - li + 1) * dx


def _taper(res, j0, ic, jdeep):
    """Sidewall taper angle from vertical (deg): + = narrowing downward
    (positive taper), - = widening / re-entrant (negative taper / bowing)."""
    if jdeep <= j0 + 5:
        return 0.0
    jt = j0 + max(2, int(0.15 * (jdeep - j0)))
    jb = j0 + int(0.85 * (jdeep - j0))
    wt = _row_width(res, ic, jt)
    wb = _row_width(res, ic, jb)
    if wt <= 0 or wb <= 0:
        return 0.0
    dz = (jb - jt) * res.dx
    return math.degrees(math.atan(0.5 * (wt - wb) / dz))


def _wall_trace(res, j0, k):
    """Trace the left sidewall x-position of opening ``k`` versus depth,
    windowed to that opening so neighbouring trenches are not picked up."""
    nx, nz, dx, phi = res.nx, res.nz, res.dx, res.phi
    x0, x1 = res.openings[k]
    ic = min(max(int(round(0.5 * (x0 + x1) / dx)), 0), nx - 1)
    ilo = max(1, int(x0 / dx) - int((x1 - x0) / dx) - 2)
    js, xs = [], []
    for j in range(j0 + 2, nz - 1):
        if phi[ic + nx * j] <= 0:          # below this trench's bottom
            break
        wall = None                         # first solid->void going right
        for i in range(ilo, ic):
            if phi[i - 1 + nx * j] < 0 and phi[i + nx * j] > 0:
                wall = i
        if wall is not None:
            js.append(float(j))
            xs.append(wall * dx)
    return js, xs


def _scallop(res, j0, k):
    """Sidewall ripple = RMS of the wall position after removing the linear
    taper (so taper/bow is not counted as scalloping)."""
    js, xs = _wall_trace(res, j0, k)
    n = len(xs)
    if n < 6:
        return 0.0
    mj, mx = sum(js) / n, sum(xs) / n
    sjj = sum((j - mj) ** 2 for j in js) or 1.0
    b = sum((j - mj) * (x - mx) for j, x in zip(js, xs)) / sjj
    a = mx - b * mj
    rms = math.sqrt(sum((x - (a + b * j)) ** 2 for j, x in zip(js, xs)) / n)
    return 2.0 * rms


def _footing(res, j0, k):
    """Lateral undercut at the oxide = how much further the wall is etched
    just above the BOX compared with mid-depth."""
    js, xs = _wall_trace(res, j0, k)
    if len(xs) < 6 or not res.reached_box[k]:
        return 0.0
    nmid = len(xs) // 2
    x_mid = sum(xs[max(0, nmid - 2):nmid + 2]) / len(xs[max(0, nmid - 2):nmid + 2])
    x_bot = min(xs[-4:])                    # leftmost (most undercut) near BOX
    return max(0.0, x_mid - x_bot)


def _py_fallback(phi, mask, nx, nz, dx, j_box, recipe, n_aniso, n_iso, dt):
    # minimal, slow reference kernel (used only when no C compiler exists)
    band = 2.5 * dx
    for _c in range(recipe.cycles):
        for mode, ns in ((0, n_aniso), (1, n_iso)):
            for _s in range(ns):
                new = phi[:]
                for j in range(1, nz - 1):
                    for i in range(1, nx - 1):
                        k = i + nx * j
                        if abs(phi[k]) > band or j >= j_box:
                            continue
                        gx = (phi[k + 1] - phi[k - 1]) * 0.5 / dx
                        gz = (phi[k + nx] - phi[k - nx]) * 0.5 / dx
                        gn = math.hypot(gx, gz) + 1e-12
                        if mode == 0:
                            rate = recipe.r_ion * max(0.0, -gz / gn)
                        else:
                            rate = recipe.r_iso * 0.6
                        if mask[k]:
                            rate /= recipe.selectivity
                        dxm = max(phi[k] - phi[k - 1], 0.0) / dx
                        dxp = min(phi[k + 1] - phi[k], 0.0) / dx
                        dzm = max(phi[k] - phi[k - nx], 0.0) / dx
                        dzp = min(phi[k + nx] - phi[k], 0.0) / dx
                        g = math.sqrt(dxm * dxm + dxp * dxp
                                      + dzm * dzm + dzp * dzp)
                        new[k] = phi[k] + dt * rate * g
                phi = new
    return phi
