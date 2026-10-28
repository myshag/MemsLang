"""Plasma physics of a Bosch recipe from Ferma's ICP solver (``ferma`` + ``ferma_icp``).

Each step of the Bosch cycle is a steady discharge of a planar-coil ICP reactor
(the plasma settles in milliseconds, a step lasts seconds):

* the etch step (``reactors/bosch_etch.ferma``): SF6 with RF bias on the wafer's
  electrode -- the gas chemistry of Ferma's built-in mechanism ``sf6``, the
  transport of the neutrals to the 150 mm silicon wafer, which F atoms etch
  thermally and under the ions, and the sheath over it, which sets the ions'
  energies;
* the passivation step (``reactors/bosch_passivation.ferma``): c-C4F8 without
  bias -- the built-in mechanism ``c4f8``, whose CF, CF2 and CF3 radicals reach
  the wafer and deposit the fluorocarbon film.

``ferma build -p ...`` sets the recipe's coil power, pressure, flow and bias
in the reactor model, ``ferma mesh`` meshes it and ``ferma_icp`` solves the
coil's RF field with the global chemistry model; its JSON summary gives the
quantities below.  A run takes from half a minute (passivation) to three
minutes (etch, the sheath's particle model), so results are cached in memory
and, with a ``cache_dir``, on disk by the reactor model and the parameters.

The binaries come from the arguments, else the environment variables
``FERMA`` and ``FERMA_ICP``, else ``PATH``; :meth:`FermaPlasma.available`
tells whether both are there.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_REACTORS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reactors")
ETCH_REACTOR = os.path.join(_REACTORS, "bosch_etch.ferma")
PASSIVATION_REACTOR = os.path.join(_REACTORS, "bosch_passivation.ferma")

# the fluorocarbon radicals that build the passivation film
_FILM_RADICALS = ("CF", "CF2", "CF3")


@dataclass(frozen=True)
class EtchPlasma:
    """What the etch step's plasma gives the wafer (SI units, energies in eV)."""
    etch_rate: float      # Si etch rate of the open wafer, mean over it, m/s
    ion_share: float      # share of that etching driven by the ions (the rest: thermal F)
    f_flux: float         # F atoms onto the wafer, m^-2 s^-1
    ion_flux: float       # positive ions onto the wafer, m^-2 s^-1
    ion_energy: float     # their mean energy at the wafer (flux-weighted), eV
    ne: float             # electron density, m^-3
    te: float             # electron temperature, eV


@dataclass(frozen=True)
class PassivationPlasma:
    """What the passivation step's plasma gives the wafer (SI units)."""
    film_flux: float      # CF + CF2 + CF3 onto the wafer, m^-2 s^-1
    ion_flux: float       # positive ions onto the wafer, m^-2 s^-1
    ne: float             # electron density, m^-3
    te: float             # electron temperature, eV


def _species(summary: dict) -> Dict[str, float]:
    """Flux of every gas species onto the wafer in a ferma_icp summary."""
    return {s["name"]: s["flux"] for s in summary.get("species", [])
            if "flux" in s and s["flux"] is not None}


def _ion_flux(fluxes: Dict[str, float]) -> float:
    return sum(v for k, v in fluxes.items() if k.endswith("+"))


def etch_plasma(summary: dict) -> EtchPlasma:
    """:class:`EtchPlasma` from the ferma_icp summary of the etch step."""
    if not summary.get("converged"):
        raise RuntimeError("plasma: the etch step's discharge did not converge")
    wafer = summary.get("wafer") or {}
    if not wafer.get("reacts"):
        raise RuntimeError("plasma: the etch step's wafer has no etch chemistry")
    fluxes = _species(summary)
    energy, weight = 0.0, 0.0
    for ion in (summary.get("bias") or {}).get("ions", []):
        w = fluxes.get(ion["name"], 0.0)
        energy += w * ion["mean"]
        weight += w
    if weight <= 0:
        raise RuntimeError("plasma: the etch step has no biased ions at the wafer")
    return EtchPlasma(
        etch_rate=wafer["etch_rate"], ion_share=wafer["ion_share"],
        f_flux=fluxes.get("F", 0.0), ion_flux=_ion_flux(fluxes),
        ion_energy=energy / weight, ne=summary["ne_max"],
        te=summary["Te_mean"])


def passivation_plasma(summary: dict) -> PassivationPlasma:
    """:class:`PassivationPlasma` from the ferma_icp summary of the passivation
    step."""
    if not summary.get("converged"):
        raise RuntimeError(
            "plasma: the passivation step's discharge did not converge")
    fluxes = _species(summary)
    return PassivationPlasma(
        film_flux=sum(fluxes.get(k, 0.0) for k in _FILM_RADICALS),
        ion_flux=_ion_flux(fluxes), ne=summary["ne_max"],
        te=summary["Te_mean"])


class FermaPlasma:
    """Runs the two steps of a Bosch recipe through Ferma's ICP solver."""

    def __init__(self, ferma: Optional[str] = None,
                 ferma_icp: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 etch_reactor: str = ETCH_REACTOR,
                 passivation_reactor: str = PASSIVATION_REACTOR):
        self.ferma = ferma or os.environ.get("FERMA") or shutil.which("ferma")
        self.ferma_icp = (ferma_icp or os.environ.get("FERMA_ICP")
                          or shutil.which("ferma_icp"))
        self.cache_dir = cache_dir
        self.etch_reactor = etch_reactor
        self.passivation_reactor = passivation_reactor
        self._memo: Dict[Tuple, dict] = {}

    def available(self) -> bool:
        return bool(self.ferma and self.ferma_icp
                    and os.path.exists(self.ferma)
                    and os.path.exists(self.ferma_icp))

    def etch(self, coil_w: float, pressure_mtorr: float, sf6_sccm: float,
             bias_w: float) -> EtchPlasma:
        return etch_plasma(self._run(self.etch_reactor, (
            ("P_rf", "%.6g W" % coil_w), ("p", "%.6g mTorr" % pressure_mtorr),
            ("q_sf6", "%.6g sccm" % sf6_sccm), ("P_b", "%.6g W" % bias_w))))

    def passivation(self, coil_w: float, pressure_mtorr: float,
                    c4f8_sccm: float) -> PassivationPlasma:
        return passivation_plasma(self._run(self.passivation_reactor, (
            ("P_rf", "%.6g W" % coil_w), ("p", "%.6g mTorr" % pressure_mtorr),
            ("q_c4f8", "%.6g sccm" % c4f8_sccm))))

    # ------------------------------------------------------------------
    def _run(self, reactor: str, params: Tuple[Tuple[str, str], ...]) -> dict:
        with open(reactor, "rb") as f:
            model = f.read()
        key = hashlib.sha1(model + repr(params).encode()).hexdigest()[:16]
        if key in self._memo:
            return self._memo[key]
        cached = (os.path.join(self.cache_dir, key + "_icp.json")
                  if self.cache_dir else None)
        if cached and os.path.exists(cached):
            with open(cached) as f:
                summary = json.load(f)
        else:
            summary = self._solve(reactor, params)
            if cached:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cached, "w") as f:
                    json.dump(summary, f)
        self._memo[key] = summary
        return summary

    def _solve(self, reactor: str, params: Tuple[Tuple[str, str], ...]) -> dict:
        if not self.available():
            raise RuntimeError(
                "plasma: ferma and ferma_icp are needed (set FERMA and "
                "FERMA_ICP or put them on PATH)")
        stem = os.path.splitext(os.path.basename(reactor))[0]
        with tempfile.TemporaryDirectory() as work:
            ir = os.path.join(work, stem + ".json")
            msh = os.path.join(work, stem + ".msh")
            overrides = [a for k, v in params for a in ("-p", "%s=%s" % (k, v))]
            for cmd in ([self.ferma, "build", reactor, "-o", ir, *overrides],
                        [self.ferma, "mesh", reactor, "-o", msh, *overrides],
                        [self.ferma_icp, ir, msh, "-o",
                         os.path.join(work, stem)]):
                run = subprocess.run(cmd, capture_output=True, text=True)
                if run.returncode != 0:
                    raise RuntimeError("plasma: %s failed:\n%s%s" % (
                        os.path.basename(cmd[0]), run.stdout[-2000:],
                        run.stderr[-2000:]))
            with open(os.path.join(work, stem + "_icp.json")) as f:
                return json.load(f)
