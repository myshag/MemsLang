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

from .etch import BoschRecipe

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
        return (pl.etch(self.coil_w, self.pressure, self.sf6, self.bias_w),
                pl.passivation(self.coil_w, self.pressure, self.c4f8),
                pl.etch(_COIL0, _PRES0, _SF6_0, _BIAS0),
                pl.passivation(_COIL0, _PRES0, _C4F8_0))

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
        return dict(rate=e.etch_rate / e0.etch_rate, iso=iso(e) / iso(e0),
                    deposit=d.film_flux / d0.film_flux * tp,
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
            "P=%.0f mTorr coil=%.0f W bias=%.0f W"
            % (self.t_etch, self.t_pass, self.sf6, self.c4f8,
               self.pressure, self.coil_w, self.bias_w),
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
                "plasma: passivation step ne=%.3g m^-3 Te=%.2f eV, CFx "
                "%.3g m^-2 s^-1" % (d.ne, d.te, d.film_flux))
        out += ["recipe: WARNING: " + m for m in w]
        return out
