"""A Bosch recipe in machine units and the plasma it makes.

The inputs an engineer sets on the tool are step times, gas flows, chamber
pressures and RF powers (or the bias voltages the tool reports).  Given a
``plasma`` (:class:`soidlc.plasma.FermaPlasma`), :meth:`BoschProcess.fluxes`
runs each step of the cycle as a steady discharge in Ferma's ICP solver and
turns the two plasmas into what the feature-scale model takes
(:class:`soidlc.etch.BoschFluxes`): the F, ion and film-radical fluxes onto the
wafer and the ions' energies.  The film's growth is the tool's one constant
(:meth:`~soidlc.etch.BoschFluxes.calibrated`).  Below are published recipes with
their measured results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import film
from .etch import BoschFluxes

# a typical SPTS/STS-class Bosch recipe: the defaults
_T_ETCH0 = 7.0      # s
_T_PASS0 = 5.0      # s
_SF6_0 = 130.0      # sccm
_C4F8_0 = 100.0     # sccm
_PRES0 = 30.0       # mTorr
_COIL0 = 600.0      # W (ICP / coil)
_BIAS0 = 15.0       # W (platen / bias)


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
    # chamber pressure of the passivation step, mTorr, where it differs
    pass_pressure: Optional[float] = None
    # platen / bias power of the passivation step, W (0: none): its ions wear
    # the film down where they reach it
    pass_bias_w: float = 0.0
    # each step's bias as the tool measured it (its DC self-bias, V), in
    # place of the power (the plasma model's voltage for a power rests on the
    # electrode's size and the ion flux)
    bias_v: Optional[float] = None
    pass_bias_v: Optional[float] = None
    # a soidlc.plasma.FermaPlasma (or anything with its etch/passivation
    # methods): the steps' plasmas
    plasma: Optional[object] = field(default=None, repr=False, compare=False)

    def _etch_plasma(self):
        return self.plasma.etch(self.coil_w, self.pressure, self.sf6,
                                self.bias_w, self.bias_v or 0.0)

    def _passivation_plasma(self):
        return self.plasma.passivation(self.coil_w, self._pass_pressure(),
                                       self.c4f8, self.pass_bias_w,
                                       self.pass_bias_v or 0.0)

    def fluxes(self, **kw):
        """The recipe's two plasmas as the input of the feature-scale model
        (:class:`soidlc.etch.BoschFluxes`; needs ``plasma``): calibrate it to
        the tool and pass it to ``etch.simulate(..., fluxes=...)``.  ``kw``
        goes to BoschFluxes (``surface``, ``rays``, ...)."""
        if self.plasma is None:
            raise RuntimeError("recipe: the Bosch model needs a plasma "
                               "(BoschProcess(plasma=FermaPlasma()))")
        return BoschFluxes.from_plasmas(
            self._etch_plasma(), self._passivation_plasma(),
            t_etch=self.t_etch, t_pass=self.t_pass, cycles=self.cycles, **kw)

    def _pass_pressure(self) -> float:
        return self.pressure if self.pass_pressure is None else self.pass_pressure

    def summary(self) -> List[str]:
        """The recipe and, with a ``plasma``, its two steps' plasmas."""
        out = [
            "recipe: t_etch=%.1f s t_pass=%.1f s SF6=%.0f C4F8=%.0f sccm "
            "P=%.0f mTorr coil=%.0f W bias=%s, in the passivation %s, %d cycles"
            % (self.t_etch, self.t_pass, self.sf6, self.c4f8,
               self.pressure, self.coil_w, _bias_text(self.bias_w, self.bias_v),
               _bias_text(self.pass_bias_w, self.pass_bias_v), self.cycles)]
        if self.plasma is not None:
            e, d = self._etch_plasma(), self._passivation_plasma()
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
