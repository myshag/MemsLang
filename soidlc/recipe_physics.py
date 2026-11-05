"""Process-physics layer: real Bosch recipe parameters -> effective knobs.

The level-set kernel (``etch_core``) is driven by *effective* quantities
(etch-per-cycle, isotropic/anisotropic balance, passivation, ...).  Those are
outputs of the process; the inputs an engineer actually sets on the tool are
step times, gas flows, chamber pressure and RF powers.  This module maps the
latter to the former through semi-empirical relations with named, calibratable
coefficients -- or, given a ``plasma`` (:class:`soidlc.plasma.FermaPlasma`),
through the plasmas of the two steps that Ferma's ICP solver computes for the
recipe: the etch step's silicon etch rate, its thermal/ion-driven split, F flux
and ion energy, the passivation step's film-forming CFx flux, each relative to
the reference recipe's, in place of the power laws in coil power, bias, flows
and pressure (the calibration constants and the warnings stay).

Honest scope: the *direction* of every dependency is physically established
and reliable (more bias -> deeper, lower selectivity, more footing; more
C4F8/t_pass -> more positive taper, eventually grass; longer etch step ->
bigger scallops).  The *absolute* constants are tool/chemistry specific and
must be calibrated (see the calibration discussion) -- the defaults below are
normalised so a typical reference recipe reproduces the package defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import film
from .etch import BoschFluxes, BoschRecipe

# reference recipe (a typical SPTS/STS-class Bosch process); the calibration
# constants are chosen so this maps onto the BoschRecipe defaults
_T_ETCH0 = 7.0      # s
_T_PASS0 = 5.0      # s
_SF6_0 = 130.0      # sccm
_C4F8_0 = 100.0     # sccm
_PRES0 = 30.0       # mTorr
_COIL0 = 600.0      # W (ICP / coil)
_BIAS0 = 15.0       # W (platen / bias)


@dataclass
class Calibration:
    """Semi-empirical coefficients (fit these to literature / a MPW run)."""
    vert_rate0: float = 0.114    # um/s vertical etch rate at the reference
    riso0: float = 0.18          # isotropic/anisotropic ratio at reference
    etch_pass_ratio: float = 0.111   # sidewall etch vs deposit at reference
    scallop0: float = 0.12       # um scallop per cycle at reference
    sel0: float = 75.0           # Si:mask selectivity at reference
    k_sel: float = 0.5           # selectivity drop with bias
    foot0: float = 1.5           # footing at reference bias
    bow0: float = 2.5            # bowing gain (pressure x under-passivation)
    grass_passiv: float = 0.96   # passivation above this risks micro-masking
    k_load: float = 0.0          # loading: rate /(1 + k_load*open_fraction)


@dataclass
class BoschProcess:
    """A real Bosch recipe in machine units."""
    t_etch: float = _T_ETCH0     # SF6 etch step, s
    t_pass: float = _T_PASS0     # C4F8 passivation step, s
    sf6: float = _SF6_0          # SF6 flow, sccm
    c4f8: float = _C4F8_0        # C4F8 flow, sccm
    pressure: float = _PRES0     # chamber pressure, mTorr
    coil_w: float = _COIL0       # ICP / coil power, W
    bias_w: float = _BIAS0       # platen / bias power, W
    cycles: int = 40
    open_fraction: float = 0.1   # fraction of wafer open (loading)
    cal: Calibration = field(default_factory=Calibration)
    # chamber pressure of the passivation step, mTorr, where it differs
    pass_pressure: Optional[float] = None
    # platen / bias power of the passivation step, W (0: none): its ions wear
    # the film down where they reach it (the flux-driven model)
    pass_bias_w: float = 0.0
    # each step's bias as the tool measured it (its DC self-bias, V), in
    # place of the power (the plasma model's voltage for a power rests on the
    # electrode's size and the ion flux)
    bias_v: Optional[float] = None
    pass_bias_v: Optional[float] = None
    # a soidlc.plasma.FermaPlasma (or anything with its etch/passivation
    # methods): the drive terms from the steps' plasmas, not the power laws
    plasma: Optional[object] = field(default=None, repr=False, compare=False)

    # ---- normalised drive variables -------------------------------------
    def _norm(self):
        c = self.coil_w / _COIL0
        b = self.bias_w / _BIAS0
        p = self.pressure / _PRES0
        s = self.sf6 / _SF6_0
        f = self.c4f8 / _C4F8_0
        te = self.t_etch / _T_ETCH0
        tp = self.t_pass / _T_PASS0
        return c, b, p, s, f, te, tp

    def plasmas(self):
        """The etch and passivation plasmas of this recipe and of the
        reference recipe (needs ``plasma``)."""
        pl = self.plasma
        return (self._etch_plasma(), self._passivation_plasma(),
                pl.etch(_COIL0, _PRES0, _SF6_0, _BIAS0),
                pl.passivation(_COIL0, _PRES0, _C4F8_0))

    def _etch_plasma(self):
        return self.plasma.etch(self.coil_w, self.pressure, self.sf6,
                                self.bias_w, self.bias_v or 0.0)

    def _passivation_plasma(self):
        return self.plasma.passivation(self.coil_w, self._pass_pressure(),
                                       self.c4f8, self.pass_bias_w,
                                       self.pass_bias_v or 0.0)

    def fluxes(self, **kw):
        """The recipe's two plasmas as the input of the flux-driven model
        (:class:`soidlc.etch.BoschFluxes`; needs ``plasma``): pass it to
        ``etch.simulate(..., fluxes=...)``.  ``kw`` goes to BoschFluxes
        (``surface``, ``rays``, ...)."""
        if self.plasma is None:
            raise RuntimeError("recipe: the flux-driven model needs a plasma "
                               "(BoschProcess(plasma=FermaPlasma()))")
        return BoschFluxes.from_plasmas(
            self._etch_plasma(), self._passivation_plasma(),
            t_etch=self.t_etch, t_pass=self.t_pass, cycles=self.cycles, **kw)

    def _pass_pressure(self) -> float:
        return self.pressure if self.pass_pressure is None else self.pass_pressure

    def _drivers(self) -> Dict[str, float]:
        """The drive terms of the recipe relative to the reference recipe's:
        the vertical (ion-assisted) etch rate, the isotropic/anisotropic
        balance, the film deposited per passivation step, the sidewall film
        etched per etch step and the ions' energy."""
        c, b, p, s, f, te, tp = self._norm()
        if self.plasma is None:
            # coil sets the flux, bias the yield; pressure makes the etch
            # isotropic, bias anisotropic; the bias power stands for the
            # ions' energy
            ion_yield = 0.4 + 0.6 * b
            return dict(rate=c * ion_yield, iso=p * s / ion_yield,
                        deposit=f * tp * c, sidewall=s * te, energy=b)
        e, d, e0, d0 = self.plasmas()

        def iso(x):  # thermal (isotropic F) over ion-driven (directional)
            return (1.0 - x.ion_share) / x.ion_share

        def deposit(x, y):  # the open wafer's film per second of passivation
            return BoschFluxes.from_plasmas(x, y, t_pass=1.0).flat_film()
        return dict(rate=e.etch_rate / e0.etch_rate, iso=iso(e) / iso(e0),
                    deposit=deposit(e, d) / deposit(e0, d0) * tp,
                    sidewall=e.f_flux / e0.f_flux * te,
                    energy=e.ion_energy / e0.ion_energy)

    def to_recipe(self) -> Tuple[BoschRecipe, List[str]]:
        """Map the recipe to effective :class:`BoschRecipe` knobs.

        Returns the recipe and a list of process warnings (grass, bowing,
        mask-erosion risks)."""
        k = self.cal
        p = self.pressure / _PRES0
        drv = self._drivers()

        # vertical (ion-assisted) etch rate; loading slows it as more area is
        # opened
        load = 1.0 / (1.0 + k.k_load * self.open_fraction)
        vert_rate = k.vert_rate0 * drv["rate"] * load              # um/s
        etch_per_cycle = vert_rate * self.t_etch

        # isotropic/anisotropic balance
        r_iso = k.riso0 * drv["iso"]

        # passivation = deposit / (deposit + sidewall etch)
        deposit, sidewall = drv["deposit"], drv["sidewall"]
        passivation = deposit / (deposit + k.etch_pass_ratio * sidewall)
        passivation = min(0.999, max(0.0, passivation))

        # scallop grows with the per-cycle vertical advance (longer/faster
        # etch step bulges the sidewall more before re-passivation)
        scallop_um = k.scallop0 * etch_per_cycle / (k.vert_rate0 * _T_ETCH0)

        # mask selectivity falls with ion energy; footing/notching rises
        selectivity = k.sel0 / (1.0 + k.k_sel * (drv["energy"] - 1.0))
        selectivity = max(5.0, selectivity)
        footing = k.foot0 * drv["energy"]

        # bowing: ion scattering at higher pressure, worse when under-passivated
        bow = k.bow0 * max(0.0, p - 1.0) * (1.0 - passivation)

        recipe = BoschRecipe(
            etch_per_cycle=etch_per_cycle, cycles=self.cycles,
            r_ion=1.0, r_iso=r_iso, selectivity=selectivity,
            footing=footing, scallop_um=scallop_um,
            passivation=passivation, bow=bow)
        return recipe, self._warn(passivation, selectivity, bow)

    def _warn(self, passivation, selectivity, bow) -> List[str]:
        w: List[str] = []
        if passivation > self.cal.grass_passiv:
            w.append(
                "recipe: passivation balance %.2f > %.2f -> micro-masking / "
                "grass (black silicon) risk; lengthen t_etch or raise SF6"
                % (passivation, self.cal.grass_passiv))
        if passivation < 0.30:
            w.append(
                "recipe: passivation %.2f very low -> severe bowing/undercut; "
                "lengthen t_pass or raise C4F8" % passivation)
        if selectivity < 20.0:
            w.append(
                "recipe: Si:mask selectivity ~%.0f is low (high bias) -> mask "
                "erosion / faceting" % selectivity)
        if bow > 4.0:
            w.append("recipe: strong re-entrant bowing predicted (%.1f); "
                     "lower pressure or raise passivation" % bow)
        return w

    def summary(self) -> List[str]:
        r, w = self.to_recipe()
        out = [
            "recipe: t_etch=%.1f s t_pass=%.1f s SF6=%.0f C4F8=%.0f sccm "
            "P=%.0f mTorr coil=%.0f W bias=%s, in the passivation %s"
            % (self.t_etch, self.t_pass, self.sf6, self.c4f8,
               self.pressure, self.coil_w, _bias_text(self.bias_w, self.bias_v),
               _bias_text(self.pass_bias_w, self.pass_bias_v)),
            "recipe: -> etch/cycle=%.2f um, r_iso=%.3f, passivation=%.2f, "
            "scallop=%.2f um, selectivity=%.0f, footing=%.1f, bow=%.2f"
            % (r.etch_per_cycle, r.r_iso, r.passivation, r.scallop_um,
               r.selectivity, r.footing, r.bow)]
        if self.plasma is not None:
            e, d, _, _ = self.plasmas()
            out.append(
                "plasma: etch step ne=%.3g m^-3 Te=%.2f eV, F %.3g and ions "
                "%.3g m^-2 s^-1 at %.0f eV, Si %.0f nm/min (%.0f%% ion-driven)"
                % (e.ne, e.te, e.f_flux, e.ion_flux, e.ion_energy,
                   e.etch_rate * 6e10, 100 * e.ion_share))
            out.append(
                "plasma: passivation step ne=%.3g m^-3 Te=%.2f eV, film from "
                "its radicals %.3g nm/s (%s), ions %.3g m^-2 s^-1%s" % (
                    d.ne, d.te, 1e9 * film.growth(film.precursors(d.radicals)),
                    ", ".join("%s %.3g" % (p.name, p.flux)
                              for p in film.precursors(d.radicals)),
                    d.ion_flux, (" at %.0f eV" % d.ion_energy)
                    if d.ion_energy > 0 else ""))
        out += ["recipe: WARNING: " + m for m in w]
        return out


def _bias_text(power_w: float, voltage_v: Optional[float]) -> str:
    if voltage_v:
        return "%.0f V" % voltage_v
    return "%.0f W" % power_w if power_w > 0 else "none"


# A published Bosch recipe with its measured result: Zhang, Wu, Sun, Shen, Lan,
# Guo, Shen & Wang, "Inductively Coupled Plasma Dry Etching of Silicon Deep
# Trenches with Extremely Vertical Smooth Sidewalls Used in Micro-Optical
# Gyroscopes", Micromachines 14, 846 (2023), table 1 and section 3.1: an Oxford
# Plasmalab System 100 ICP180, the electrode at 10 C, AZ4620 photoresist; the
# etch step also feeds 5 sccm of C4F8 and the passivation step 5 sccm of SF6
# and 10 W of bias (left out here: pass_bias_w=10 puts it in; the README).
# Its plasma: FermaPlasma with plasma.ICP180_ETCH and
# plasma.ICP180_PASSIVATION.
ZHANG2023 = dict(t_etch=8.0, t_pass=10.0, sf6=100.0, c4f8=100.0,
                 pressure=40.0, pass_pressure=30.0, coil_w=700.0, bias_w=25.0,
                 cycles=100)
# what they measured: a spiral trench 6 um wide (37.34 m long on a 6-inch
# wafer, 1.3 % of it) under 6 um of AZ4620, 30.72 um deep after 100 cycles
# (30 min: 1.02 um/min, 0.307 um per cycle), scallops 70 - 80 nm
ZHANG2023_MEASURED = dict(width_um=6.0, mask_um=6.0, open_fraction=0.013,
                          cycles=100, depth_um=30.72, scallop_nm=(70.0, 80.0),
                          bottom_width_um=5.9)  # fig. 4b, read off the SEM

# Another, with the walls' angle against the passivation: Craigie, Sheehan,
# Johnson, Burkett, Moll & Knowlton, "Polymer thickness effects on Bosch etch
# profiles", J. Vac. Sci. Technol. B 20, 2229 (2002), table I: an Oxford
# Plasmalab System100 (its source, 180 or 380, is not given), 15 C, 50 um
# trenches under 10 um of AZ P4620, 45 min of 15 s etch steps (SF6 85 sccm,
# 30 mTorr, 450 W, 30 W of bias: 150 V measured) and passivation steps (C4F8
# at 35 mTorr, 450 W and 15 W of bias: 110 V; left out here, pass_bias_v=110
# puts it in; the README).  Ferma's sheath over the 150 mm wafer takes 15 W at
# 150 V in the etch step and 2.5 W at 110 V in the passivation: the tool's
# powers spread over a larger electrode, and the voltages are the measured
# ones.  Its plasma: FermaPlasma with plasma.ICP180_ETCH (f_open 10 %: an
# assumption, the pattern is not given) and plasma.ICP180_PASSIVATION.
CRAIGIE2002 = dict(t_etch=15.0, sf6=85.0, pressure=30.0, pass_pressure=35.0,
                   coil_w=450.0, bias_w=30.0, minutes=45.0, width_um=50.0,
                   mask_um=10.0, open_fraction=0.10)
# the walls' angle (degrees inside the silicon from the wafer's surface, above
# 90 narrowing downwards), read off figs. 8 (60 sccm of C4F8; the passivation
# step's seconds) and 7 (10 s; the C4F8 sccm)
CRAIGIE2002_MEASURED = dict(
    angle_vs_t_pass={5: 88.95, 6: 88.86, 7: 89.37, 8: 89.43, 9: 89.44, 10: 89.75,
                     11: 89.64, 12: 90.53, 15: 90.63},
    angle_vs_c4f8={5: 83.5, 10: 85.0, 15: 87.0, 20: 89.2, 30: 88.9, 50: 90.0,
                   60: 89.75, 70: 89.9, 80: 89.85})


def zhang2023(**kw) -> BoschProcess:
    """The recipe of Zhang et al. 2023 (``ZHANG2023``) as a BoschProcess;
    ``kw`` overrides its fields (``plasma=FermaPlasma(...)``, ``cycles``)."""
    return BoschProcess(**{**ZHANG2023, **kw})
