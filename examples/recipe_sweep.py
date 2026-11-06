#!/usr/bin/env python3
"""Sweep the passivation step's length of a real Bosch recipe and watch the
profile respond.

Run:  python examples/recipe_sweep.py
Zhang et al.'s recipe (recipe_physics.ZHANG2023; its plasmas Ferma's, from
tests/data, so no Ferma binaries are needed), the film's growth calibrated to
their depth at their 10 s of passivation and kept for the others: a longer
passivation step leaves more film, the bottom clears later in each etch step,
the trench etches less deep and its walls turn from bowing to narrowing.  A
step's plasma does not depend on its length; with FermaPlasma, sweeping a
flow, a pressure or a power reruns the discharge (BoschProcess.fluxes).
"""
import json
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from soidlc import etch, plasma                                     # noqa: E402
from soidlc.recipe_physics import ZHANG2023                          # noqa: E402


def summary(step):
    with open(os.path.join(ROOT, "tests", "data", step + "_icp.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    ref = etch.BoschFluxes.from_plasmas(
        plasma.etch_plasma(summary("icp180_etch")),
        plasma.passivation_plasma(summary("icp180_passivation")),
        t_etch=ZHANG2023["t_etch"], t_pass=ZHANG2023["t_pass"], cycles=30)
    k = ref.calibrated(7.7).growth_scale(ref)
    print("t_pass  breakthrough  depth     taper")
    for t_pass in (6.0, 8.0, 10.0, 12.0):
        f = replace(ref, t_pass=t_pass).growth_scaled(k)
        r = etch.simulate([(10, 16)], domain_w=26, depth=22, dx=0.1, fluxes=f,
                          mask_thick=6.0)
        print("%4.0f s   %5.1f s      %5.1f um  %+5.1f deg"
              % (t_pass, f.breakthrough(), r.depths[0], r.tapers[0]))
