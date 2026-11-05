"""Level-set Bosch / DRIE etch simulation (tier-2 process model).

Drives the C kernel (shared ``levelset.c`` core + ``bosch.c`` velocity field,
compiled together into ``etch_core.so`` with the system gcc on first use; falls
back to a slow pure-Python kernel if no compiler is present).
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
# shared level-set core + per-process velocity fields, one shared library
_SRCS = [os.path.join(_HERE, f)
         for f in ("levelset.c", "bosch.c", "wet.c", "corner.c", "drie.c")]
_HDR = os.path.join(_HERE, "levelset.h")
_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    deps = _SRCS + [_HDR]
    need = (not os.path.exists(_SO)
            or os.path.getmtime(_SO) < max(os.path.getmtime(s) for s in deps))
    if need:
        for cc in ("gcc", "cc", "clang"):
            try:
                subprocess.run(
                    [cc, "-O3", "-fopenmp", "-shared", "-fPIC",
                     *_SRCS, "-o", _SO, "-lm"],
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
        lib.drie_run.restype = None
        lib.drie_run.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
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
class SurfaceKinetics:
    """Surface parameters of the flux-driven Bosch model (``drie.c``), SI.

    Silicon is etched through adsorbed F as on Ferma's wafer (Belen, Gomez,
    Kiehlbauch, Cooperberg & Aydil, J. Vac. Sci. Technol. A 23, 99 (2005)),
    so an open flat wafer etches at the rate Ferma reports.  The fluorocarbon
    film and the mask follow Kokkoris, Tserepi, Boudouvis & Gogolides, J. Vac.
    Sci. Technol. A 22, 1896 (2004) (the set Ertl & Selberherr, Microelectron.
    Eng. 87, 20 (2010) and ViennaPS's Bosch example use): volumes per
    arriving particle and sticking probabilities.  Their ion yields are taken
    at ``e_ref`` and scaled with sqrt(E) - sqrt(``e_th_ion``) -- both energies
    are assumptions, not from that paper."""
    f_sticking: float = 0.7          # F on the free silicon sites
    # F m^-2 s^-1 removed at full coverage (SiF4): ten times Belen et al.'s,
    # as in Ferma's built-in "sf6" (Tinck et al., Plasma Process. Polym. 14,
    # e1700018 (2017): unbiased silicon at 123 nm/s in SF6)
    thermal: float = 3e22
    ion_a: float = 7.0               # Si per ion per sqrt(eV) at full coverage (SiF2)
    ion_eth: float = 15.0            # eV, silicon's threshold
    alpha_poly: float = 125e-30      # m^3 of film per ion (at e_ref)
    beta_poly: float = 0.03e-30      # m^3 of film per F
    alpha_mask: float = 13.5e-30     # m^3 of mask per ion (at e_ref)
    beta_mask: float = 0.045e-30     # m^3 of mask per F
    e_ref: float = 100.0             # eV at which alpha_poly, alpha_mask hold
    e_th_ion: float = 15.0           # eV, the film's and the mask's threshold
    beta_dep: float = 0.5e-30        # m^3 of film per CFx (passivation step)
    alpha_dep: float = 10e-30        # m^3 of film per ion (passivation step)
    stick_dep: float = 0.1           # CFx lost per hit (passivation step)
    stick_poly: float = 0.1          # F lost on the film per hit
    stick_mask: float = 0.2          # F lost on the mask (and the stop layer)
    ion_sigma_deg: float = 2.0       # spread of the ions' angles
    n_si: float = 4.99e28            # silicon atoms per m^3


@dataclass
class BoschFluxes:
    """What the plasma delivers to an open flat wafer in each step of the
    cycle (SI units, energies in eV), and the cycle's timing -- the input of
    the flux-driven model, e.g. from :meth:`BoschProcess.fluxes`."""
    f_flux: float                    # F, etch step, m^-2 s^-1
    ion_flux: float                  # positive ions, etch step
    ion_energy: float                # their mean energy at the wafer, eV
    film_flux: float                 # film-forming CFx, passivation step
    pass_ion_flux: float             # positive ions, passivation step
    t_etch: float = 7.0              # s
    t_pass: float = 5.0              # s
    cycles: int = 40
    surface: SurfaceKinetics = field(default_factory=SurfaceKinetics)
    # the energy in silicon's ion yield (eV), when it differs from the mean
    # (the yield averaged over the ions' energy distributions); None: the mean
    si_energy: Optional[float] = None
    rays: int = 64                   # neutral rays per surface point
    ion_angles: int = 11             # ion directions per surface point
    cfl: float = 0.3                 # front advance per time step, cells
    reinit_stride: int = 4

    @classmethod
    def from_plasmas(cls, etch, passivation, **kw) -> "BoschFluxes":
        """The fluxes of a recipe from its two steps' plasmas
        (:class:`soidlc.plasma.EtchPlasma`, :class:`~soidlc.plasma.PassivationPlasma`).

        The F flux and silicon's ion term are those that give the plasma's
        own etch rate and ion-driven share of the open wafer: the plasma model
        balances the coverage with the F density at the wafer, depleted by the
        etching, not with the F flux it reports -- so an open flat surface of
        the feature etches exactly as the plasma's wafer does."""
        s = kw.get("surface") or SurfaceKinetics()
        rate, share = etch.etch_rate, etch.ion_share
        f_flux, si_energy = etch.f_flux, None
        if rate > 0 and 0.0 <= share < 1.0:
            theta = 4.0 * (1.0 - share) * rate * s.n_si / s.thermal
            if 0.0 < theta < 1.0:
                ion_term = share * rate * s.n_si / theta       # Y Gi, Si/m^2/s
                f_flux = theta * (s.thermal + 2.0 * ion_term) / (
                    s.f_sticking * (1.0 - theta))
                if etch.ion_flux > 0:
                    si_energy = (ion_term / (s.ion_a * etch.ion_flux)
                                 + math.sqrt(s.ion_eth)) ** 2
        return cls(f_flux=f_flux, ion_flux=etch.ion_flux,
                   ion_energy=etch.ion_energy,
                   film_flux=passivation.film_flux,
                   pass_ion_flux=passivation.ion_flux, si_energy=si_energy,
                   **kw)

    def flat_etch_rate(self) -> float:
        """Silicon etch rate of the open flat wafer, m/s (Ferma's)."""
        s = self.surface
        e = self.si_energy if self.si_energy is not None else self.ion_energy
        y = (s.ion_a * (math.sqrt(e) - math.sqrt(s.ion_eth))
             if e > s.ion_eth else 0.0)
        sg = s.f_sticking * self.f_flux
        theta = sg / (sg + s.thermal + 2.0 * y * self.ion_flux)
        return theta * (0.25 * s.thermal + y * self.ion_flux) / s.n_si

    def flat_film(self) -> float:
        """Film one passivation step leaves on the open flat wafer, m."""
        s = self.surface
        return (s.beta_dep * self.film_flux
                + s.alpha_dep * self.pass_ion_flux) * self.t_pass

    def flat_strip(self) -> float:
        """Rate at which the etch step wears the film off the open flat
        wafer, m/s (the ions and F)."""
        s = self.surface
        y = (s.alpha_poly * (math.sqrt(self.ion_energy) - math.sqrt(s.e_th_ion))
             / (math.sqrt(s.e_ref) - math.sqrt(s.e_th_ion))
             if self.ion_energy > s.e_th_ion else 0.0)
        return y * self.ion_flux + s.beta_poly * self.f_flux

    def breakthrough(self) -> float:
        """Time the etch step takes to clear one passivation step's film off
        the open flat wafer, s (inf: never)."""
        strip = self.flat_strip()
        return self.flat_film() / strip if strip > 0 else math.inf

    def calibrated(self, breakthrough: float) -> "BoschFluxes":
        """These fluxes with the film's deposition (``beta_dep``,
        ``alpha_dep``) scaled so that the open flat wafer's film clears in
        ``breakthrough`` seconds of the etch step -- the one constant to take
        from the tool (the published volumes per particle come from another
        reactor; the film's growth is the least transferable of them)."""
        from dataclasses import replace
        k = breakthrough / self.breakthrough()
        s = replace(self.surface, beta_dep=self.surface.beta_dep * k,
                    alpha_dep=self.surface.alpha_dep * k)
        return replace(self, surface=s)

    def _params(self):
        s = self.surface
        return [self.cycles, self.t_etch, self.t_pass, self.f_flux,
                self.ion_flux, self.ion_energy, self.film_flux,
                self.pass_ion_flux, s.f_sticking, s.thermal, s.ion_a,
                s.ion_eth, s.alpha_poly, s.beta_poly, s.alpha_mask,
                s.beta_mask, s.e_ref, s.e_th_ion, s.beta_dep, s.alpha_dep,
                s.stick_dep, s.stick_poly, s.stick_mask, s.ion_sigma_deg,
                self.rays, self.ion_angles, s.n_si, self.cfl,
                self.reinit_stride,
                self.si_energy if self.si_energy is not None
                else self.ion_energy]


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
    warnings: List[str] = field(default_factory=list)   # process warnings
    # the flux-driven model's: the open wafer's etch rate (m/s) and F
    # coverage, the film after the first passivation step (m), when the first
    # etch step clears it (s; -1: never), flux evaluations, time steps
    fluxes: Optional[BoschFluxes] = None
    diagnostics: dict = field(default_factory=dict)

    def report(self) -> List[str]:
        out = ["etch   Bosch level-set: %d x %d cells @ %.3f um, BOX at %.1f um"
               % (self.nx, self.nz, self.dx, self.j_box * self.dx)]
        if self.fluxes is not None:
            d = self.diagnostics
            out.append(
                "etch   from the plasma's fluxes: open Si %.0f nm/min "
                "(F coverage %.2f), film %.1f nm per passivation step, "
                "cleared from the bottom after %s of %.1f s"
                % (d["etch_rate"] * 6e10, d["coverage"], d["film"] * 1e9,
                   ("%.2f s" % d["clear_time"]) if d["clear_time"] >= 0
                   else "never", self.fluxes.t_etch))
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
             box_at: Optional[float] = None,
             process: Optional[object] = None,
             fluxes: Optional[BoschFluxes] = None) -> EtchResult:
    """Etch a set of mask ``openings`` (list of (x0, x1) in um) into a wafer.

    ``domain_w`` x ``depth`` is the simulated cross-section (um); the buried
    oxide sits at ``box_at`` (default: just below the target depth).  Pass a
    :class:`~soidlc.recipe_physics.BoschProcess` as ``process`` to drive the
    etch from real machine parameters (step times, flows, pressure, powers)
    instead of the effective ``recipe`` knobs, or the plasma's fluxes
    (:class:`BoschFluxes`, e.g. ``BoschProcess(plasma=...).fluxes()``) as
    ``fluxes`` for the flux-driven model: the particles' transport into the
    feature, the passivation film and the surface kinetics in place of the
    knobs.
    """
    proc_warnings: List[str] = []
    if process is not None and fluxes is None:
        recipe, proc_warnings = process.to_recipe()
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
    if fluxes is not None:
        if lib is None:
            raise RuntimeError("etch: the flux-driven model needs the C kernel "
                               "(no C compiler found)")
        arr = (ctypes.c_double * len(phi))(*phi)
        mk = (ctypes.c_ubyte * len(mask)).from_buffer_copy(bytes(mask))
        prm = fluxes._params()
        par = (ctypes.c_double * len(prm))(*prm)
        out = (ctypes.c_double * 8)()
        lib.drie_run(arr, mk, nx, nz, dx, j_box, par, out)
        res = EtchResult(list(arr), nx, nz, dx, j_box, openings)
        res.fluxes = fluxes
        res.diagnostics = dict(etch_rate=out[0], coverage=out[1], film=out[2],
                               clear_time=out[3], flux_evaluations=int(out[4]),
                               steps=int(out[5]))
        _measure(res, j0, mask_thick)
        return res
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
    res.warnings = proc_warnings
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
        # measured width at mid-depth: the void around this opening's centre
        # (a neighbouring trench within reach is not part of it)
        jm = j0 + int((jdeep - j0) * 0.5)
        width = _row_width(res, ic, jm) or (x1 - x0)
        res.depths.append(depth)
        res.widths.append(max(width, x1 - x0))
        res.reached_box.append(jdeep >= res.j_box - 1)
        res.tapers.append(_taper(res, j0, ic, jdeep))
    # scallop and footing measured on the deepest (best-resolved) trench
    kd = max(range(len(res.depths)), key=lambda k: res.depths[k])
    res.scallop_um = _scallop(res, j0, kd,
                              res.fluxes.cycles if res.fluxes is not None else None)
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


def _scallop(res, j0, k, cycles=None):
    """Sidewall ripple of the wall after removing its linear taper (so
    taper/bow is not counted as scalloping), over the middle 70 % of the
    depth: the undercut under the mask at the top and the bottom's rounding
    are not scallops.  With the number of cycles known (the flux-driven
    model), the median of the peak-to-valley within each cycle's advance --
    a scallop's depth; else twice the RMS of the ripple."""
    js, xs = _wall_trace(res, j0, k)
    if js:
        top, bottom = js[0], js[-1]
        keep = [(j, x) for j, x in zip(js, xs)
                if top + 0.15 * (bottom - top) <= j <= bottom - 0.15 * (bottom - top)]
        js, xs = [j for j, _ in keep], [x for _, x in keep]
    n = len(xs)
    if n < 6:
        return 0.0
    mj, mx = sum(js) / n, sum(xs) / n
    sjj = sum((j - mj) ** 2 for j in js) or 1.0
    b = sum((j - mj) * (x - mx) for j, x in zip(js, xs)) / sjj
    a = mx - b * mj
    resid = [x - (a + b * j) for j, x in zip(js, xs)]
    if cycles:
        period = res.depths[k] / res.dx / cycles   # rows per cycle
        if period >= 2:
            pp, start = [], js[0]
            while start + period <= js[-1] + 1e-9:
                w = [r for j, r in zip(js, resid) if start <= j < start + period]
                if len(w) > 1:
                    pp.append(max(w) - min(w))
                start += period
            if pp:
                return sorted(pp)[len(pp) // 2]
    rms = math.sqrt(sum(r * r for r in resid) / n)
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
