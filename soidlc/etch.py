"""Level-set Bosch / DRIE etch simulation from the plasma's fluxes.

Drives the C kernel (the shared ``levelset.c`` core and ``drie.c``, compiled
with the other processes' velocity fields into ``etch_core.so`` with the system
compiler on first use).  Given a mask cross-section and what the two steps'
plasmas deliver to the wafer (:class:`BoschFluxes`, e.g. from
:meth:`soidlc.recipe_physics.BoschProcess.fluxes`), it transports the
particles into the feature, grows and wears the passivation film and etches
the silicon, cycle by cycle, and reports the resulting profile: trench depth,
sidewall scalloping, aspect-ratio-dependent etch (ARDE / RIE-lag), taper, and
footing at the buried oxide (a trench that does not reach the BOX leaves its
region anchored, not released).
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import film
from .film import FilmPrecursor

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "etch_core.so")
# shared level-set core + per-process velocity fields, one shared library
_SRCS = [os.path.join(_HERE, f)
         for f in ("levelset.c", "wet.c", "corner.c", "drie.c")]
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
        lib.drie_run.restype = None
        lib.drie_run.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        _LIB = lib
    return _LIB


# ---------------------------------------------------------------------------
@dataclass
class SurfaceKinetics:
    """Surface parameters of the flux-driven Bosch model (``drie.c``), SI.

    Silicon is etched through adsorbed F as on Ferma's wafer (Belen, Gomez,
    Kiehlbauch, Cooperberg & Aydil, J. Vac. Sci. Technol. A 23, 99 (2005)),
    so an open flat wafer etches at the rate Ferma reports.  The fluorocarbon
    film and the mask follow Kokkoris, Tserepi, Boudouvis & Gogolides, J. Vac.
    Sci. Technol. A 22, 1896 (2004) (the set Ertl & Selberherr, Microelectron.
    Eng. 87, 20 (2010) and ViennaPS's Bosch example use): volumes per
    arriving particle and sticking probabilities.  The film's radicals each
    join it with their own probability (:mod:`soidlc.film`); the ions add
    ``alpha_dep``.  Their ion yields are taken
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
    alpha_dep: float = 10e-30        # m^3 of film per ion (passivation step)
    stick_poly: float = 0.1          # F lost on the film per hit
    stick_mask: float = 0.2          # F lost on the mask (and the stop layer)
    ion_sigma_deg: float = 2.0       # spread of the ions' angles
    n_si: float = 4.99e28            # silicon atoms per m^3
    # silicon's linear law in place of Belen's, m^3 per ion and per F
    # arriving (F sticking with f_sticking): Kokkoris et al.'s, as Ertl &
    # Selberherr and ViennaPS's Bosch example take it (270e-30, 0.9e-30 with
    # f_sticking 0.2); 0: Belen's
    si_alpha: float = 0.0
    si_beta: float = 0.0
    # the film's and the mask's yields per ion against the ion's angle to the
    # surface's normal theta: (1 + B sin^2 theta) cos theta, a sputtering
    # yield's form (Yamamura), up to 3.9 times the normal one near 52 degrees
    # (B 8.5) and 0 at grazing incidence, as ViennaPS's PlasmaEtching takes
    # it (polymer 8.5, mask 9.3); negative: the same at every angle.  Off:
    # with ViennaPS's B the scallops of Zhang et al.'s recipe come out 103 nm
    # (measured 70 - 80) and Craigie et al.'s angles as without (the README)
    poly_angular: float = -1.0
    mask_angular: float = -1.0

    def film_yield(self, energy: float) -> float:
        """The film removed per ion of ``energy`` eV, m^3 (``alpha_poly`` at
        ``e_ref``, as sqrt(E) - sqrt(``e_th_ion``); 0 below the threshold)."""
        if energy <= self.e_th_ion:
            return 0.0
        return (self.alpha_poly
                * (math.sqrt(energy) - math.sqrt(self.e_th_ion))
                / (math.sqrt(self.e_ref) - math.sqrt(self.e_th_ion)))


@dataclass
class BoschFluxes:
    """What the plasma delivers to an open flat wafer in each step of the
    cycle (SI units, energies in eV), and the cycle's timing -- the input of
    the flux-driven model, e.g. from :meth:`BoschProcess.fluxes`."""
    f_flux: float                    # F, etch step, m^-2 s^-1
    ion_flux: float                  # positive ions, etch step
    ion_energy: float                # their mean energy at the wafer, eV
    # the radicals that build the film in the passivation step (soidlc.film)
    precursors: Tuple[FilmPrecursor, ...]
    pass_ion_flux: float             # positive ions, passivation step
    # their mean energy at the wafer, eV: a bias on the passivation step
    # makes its ions wear the film down (SurfaceKinetics.film_yield); 0 (or
    # an unbiased wafer's few eV, below the film's threshold): they do not
    pass_ion_energy: float = 0.0
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
                   precursors=film.precursors(passivation.radicals),
                   pass_ion_flux=passivation.ion_flux,
                   pass_ion_energy=passivation.ion_energy, si_energy=si_energy,
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

    def flat_growth(self) -> float:
        """The film the radicals and the ions add to the open flat wafer in
        the passivation step, m/s (before the ions' removal)."""
        return (film.growth(self.precursors)
                + self.surface.alpha_dep * self.pass_ion_flux)

    def pass_removal(self) -> float:
        """The film the passivation step's ions remove from the open flat
        wafer, m/s (a bias on that step)."""
        return self.surface.film_yield(self.pass_ion_energy) * self.pass_ion_flux

    def flat_film(self) -> float:
        """Film one passivation step leaves on the open flat wafer, m."""
        return max(0.0, self.flat_growth() - self.pass_removal()) * self.t_pass

    def flat_strip(self) -> float:
        """Rate at which the etch step wears the film off the open flat
        wafer, m/s (the ions and F)."""
        s = self.surface
        return (s.film_yield(self.ion_energy) * self.ion_flux
                + s.beta_poly * self.f_flux)

    def breakthrough(self) -> float:
        """Time the etch step takes to clear one passivation step's film off
        the open flat wafer, s (inf: never)."""
        strip = self.flat_strip()
        return self.flat_film() / strip if strip > 0 else math.inf

    def calibrated(self, breakthrough: float) -> "BoschFluxes":
        """These fluxes with the film's growth (every radical's and the ions'
        film, ``alpha_dep``) scaled so that the open flat wafer's film clears
        in ``breakthrough`` seconds of the etch step -- the one constant to
        take from the tool (the film's growth is the least transferable
        number: the radicals' fluxes and the film's density as much as the
        sticking).  The passivation's ions' removal is not scaled: it is
        their yield, the same as the etch step's.  The scale
        (:meth:`growth_scale`) says how far the plasma's fluxes and the
        measured sticking are from the tool's."""
        want = breakthrough * self.flat_strip() / self.t_pass
        return self.growth_scaled((want + self.pass_removal())
                                  / self.flat_growth())

    def growth_scaled(self, k: float) -> "BoschFluxes":
        """These fluxes with every radical's film and the ions' scaled by k
        (e.g. another recipe's :meth:`growth_scale` on the same tool)."""
        from dataclasses import replace
        s = replace(self.surface, alpha_dep=self.surface.alpha_dep * k)
        pre = tuple(replace(p, volume=p.volume * k) for p in self.precursors)
        return replace(self, surface=s, precursors=pre)

    def growth_scale(self, reference: "BoschFluxes") -> float:
        """The film's growth (:meth:`flat_growth`) relative to
        ``reference``'s (e.g. these fluxes before :meth:`calibrated`)."""
        return self.flat_growth() / reference.flat_growth()

    def _params(self):
        s = self.surface
        pre = [x for p in self.precursors
               for x in (p.flux, p.sticking, p.volume)]
        return [self.cycles, self.t_etch, self.t_pass, self.f_flux,
                self.ion_flux, self.ion_energy, self.pass_ion_flux,
                s.f_sticking, s.thermal, s.ion_a, s.ion_eth, s.alpha_poly,
                s.beta_poly, s.alpha_mask, s.beta_mask, s.e_ref, s.e_th_ion,
                s.alpha_dep, s.stick_poly, s.stick_mask, s.ion_sigma_deg,
                self.rays, self.ion_angles, s.n_si, self.cfl,
                self.reinit_stride,
                self.si_energy if self.si_energy is not None
                else self.ion_energy, s.si_alpha, s.si_beta,
                self.pass_ion_energy, s.poly_angular, s.mask_angular,
                len(self.precursors)] + pre


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
    fluxes: Optional[BoschFluxes] = None
    # the open wafer's etch rate (m/s) and F coverage, the film after the
    # first passivation step (m), when the first etch step clears it (s; -1:
    # never), flux evaluations, time steps
    diagnostics: dict = field(default_factory=dict)

    def report(self) -> List[str]:
        d = self.diagnostics
        out = ["etch   Bosch level-set: %d x %d cells @ %.3f um, BOX at %.1f um"
               % (self.nx, self.nz, self.dx, self.j_box * self.dx),
               "etch   from the plasma's fluxes: open Si %.0f nm/min "
               "(F coverage %.2f), film %.1f nm per passivation step, "
               "cleared from the bottom after %s of %.1f s"
               % (d["etch_rate"] * 6e10, d["coverage"], d["film"] * 1e9,
                  ("%.2f s" % d["clear_time"]) if d["clear_time"] >= 0
                  else "never", self.fluxes.t_etch)]
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
             depth: float, dx: float = 0.15, *, fluxes: BoschFluxes,
             mask_thick: float = 2.0, box: bool = True,
             box_at: Optional[float] = None) -> EtchResult:
    """Etch a set of mask ``openings`` (list of (x0, x1) in um) into a wafer
    with what the plasma delivers (``fluxes``, e.g.
    ``BoschProcess(plasma=...).fluxes()``, calibrated to the tool): the
    particles' transport into the feature, the passivation film and the
    surface kinetics, cycle by cycle.

    ``domain_w`` x ``depth`` is the simulated cross-section (um), under a mask
    ``mask_thick`` thick; the buried oxide sits at ``box_at`` (default: just
    below the target depth).
    """
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
    if lib is None:
        raise RuntimeError("etch: the Bosch model needs the C kernel "
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
    res.scallop_um = _scallop(res, j0, kd, res.fluxes.cycles)
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


def _scallop(res, j0, k, cycles):
    """Sidewall ripple of the wall after removing its linear taper (so
    taper/bow is not counted as scalloping), over the middle 70 % of the
    depth: the undercut under the mask at the top and the bottom's rounding
    are not scallops.  The median of the peak-to-valley within each cycle's
    advance -- a scallop's depth; twice the RMS of the ripple where a cycle
    advances less than two cells."""
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
