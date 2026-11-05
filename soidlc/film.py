"""The fluorocarbon film of the Bosch passivation step: the radicals that build
it, each with the probability that one arriving at the film joins it.

The plasma (Ferma) gives each species' flux onto the wafer; how readily each
joins the film is surface chemistry, from measurements where there are any.
They differ by orders of magnitude: CF2, the largest flux of a C4F8 plasma,
hardly joins a film without ions (and an ion-bombarded film gives it off),
while CF and C join at about every second hit.  A species that sticks readily
coats the top of a feature; one that rarely does reaches its depth.  Saturated
molecules (C4F8, C2F6, CF4) and F do not build the film.  The ions' part (the
film they activate) is :attr:`SurfaceKinetics.alpha_dep` in ``etch``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

# the film's density, kg/m^3: an assumption (PTFE is 2.2e3); the passivation's
# calibration (``BoschFluxes.calibrated``) scales the film's growth anyway
FILM_DENSITY = 2.0e3

_AMU = 1.66053907e-27
_MASS = {"C": 12.011, "F": 18.998}

# probability that a radical arriving at a fluorocarbon film without ions
# (a feature's walls) joins it
STICKING: Dict[str, float] = {
    # with the ions removed, CF2 scatters off films from C4F8 plasmas with S =
    # 0.9 - 1.3 (IRIS: Martin & Fisher, J. Vac. Sci. Technol. A 22, 2168
    # (2004)); with ions the film gives it off (S > 1). 1.5e-3 on a film no
    # ion reaches in Huang et al.'s model (J. Vac. Sci. Technol. A 37, 031304
    # (2019)), ten times that where the ions activate it: the 0.014 per hit
    # on a reactor's walls (Hikosaka, Toyoda & Sugai, Jpn. J. Appl. Phys. 32,
    # L353 (1993); Ferma's c4f8 walls)
    "CF2": 1.5e-3,
    # CF3 and C3F5 have one dangling bond: they end a chain but cannot join
    # the film's network, whose precursors are CF, CF2, C2F3 and C2F4 (Zhang
    # & Kushner, J. Vac. Sci. Technol. A 19, 524 (2001)); only the sites the
    # ions activate take them (0.012 per hit on a reactor's walls, Hikosaka
    # et al. 1993; Ferma's c4f8 walls), and an ion-bombarded film gives CF3
    # off (Kimura, Coburn & Graves, J. Vac. Sci. Technol. A 22, 2508 (2004))
    "CF3": 0.0,
    # R = 1 - S of 0.4 - 0.65 with the ions blocked, C3F8 and C3F6 plasmas
    # (IRIS: Cuddy, Blechle & Fisher, Int. J. Mass Spectrom. 330-332, 46
    # (2012))
    "CF": 0.5,
    # not measured; the carbon-rich precursors' 0.5 in Izawa et al.'s fit to
    # contact holes' sidewall deposition (Jpn. J. Appl. Phys. 46, 7870 (2007))
    "C": 0.5,
    # not measured; 1e-3 on a film without ions in Huang et al.'s model of an
    # Ar/C4F8/O2 plasma (J. Vac. Sci. Technol. A 37, 031304 (2019))
    "C2F4": 1e-3,
    "C3F5": 0.0,
}


@dataclass(frozen=True)
class FilmPrecursor:
    """A radical that builds the film, as it reaches the open flat wafer."""
    name: str
    flux: float          # onto the open flat wafer, m^-2 s^-1
    sticking: float      # probability of joining the film per hit
    volume: float        # film per radical joined, m^3


def volume(name: str) -> float:
    """Film volume one radical (e.g. "C3F5") adds when it joins, m^3."""
    mass = 0.0
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", name):
        mass += _MASS[el] * (int(n) if n else 1)
    return mass * _AMU / FILM_DENSITY


def precursors(radicals: Dict[str, float]) -> Tuple[FilmPrecursor, ...]:
    """The film's precursors among the fluxes onto the wafer (name -> m^-2
    s^-1, e.g. :attr:`soidlc.plasma.PassivationPlasma.radicals`)."""
    return tuple(FilmPrecursor(k, radicals[k], STICKING[k], volume(k))
                 for k in STICKING if radicals.get(k, 0.0) > 0.0)


def growth(pre: Tuple[FilmPrecursor, ...]) -> float:
    """The film's growth on the open flat wafer from the radicals, m/s."""
    return sum(p.flux * p.sticking * p.volume for p in pre)
