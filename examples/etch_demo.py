#!/usr/bin/env python3
"""Bosch DRIE level-set demo: an ARDE ladder etched toward a buried oxide.

The plasma is Zhang et al.'s recipe (recipe_physics.ZHANG2023) as Ferma's ICP
solver gives it (its summaries in tests/data, so no Ferma binaries are needed),
the film's growth calibrated to their depth (a breakthrough of 7.7 s).  The
narrowest opening lags (fewer of the plasma's ions and F reach its bottom);
the widest gets nearly the open wafer's film, which takes 7.7 s of the 8 s
step to clear, and lags too.
Run:  python examples/etch_demo.py   (writes out/etch_arde.png)"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from soidlc import etch, plasma                                     # noqa: E402
from soidlc.recipe_physics import ZHANG2023                          # noqa: E402
from soidlc.render import write_png                                  # noqa: E402


def summary(step):
    with open(os.path.join(ROOT, "tests", "data", step + "_icp.json")) as f:
        return json.load(f)


def render(res, path, scale=4):
    nx, nz = res.nx, res.nz
    W, H = nx * scale, nz * scale
    img = bytearray(3 * W * H)
    SI, VOID, BOX = (70, 90, 120), (20, 20, 26), (150, 120, 60)
    for j in range(nz):
        for i in range(nx):
            p = res.phi[i + nx * j]
            c = BOX if j >= res.j_box else (VOID if p > 0 else SI)
            for dy in range(scale):
                for dx2 in range(scale):
                    o = ((j * scale + dy) * W + (i * scale + dx2)) * 3
                    img[o:o + 3] = bytes(c)
    write_png(path, W, H, img)


if __name__ == "__main__":
    f = etch.BoschFluxes.from_plasmas(
        plasma.etch_plasma(summary("icp180_etch")),
        plasma.passivation_plasma(summary("icp180_passivation")),
        t_etch=ZHANG2023["t_etch"], t_pass=ZHANG2023["t_pass"],
        cycles=40).calibrated(7.7)
    openings = [(4, 6), (14, 18), (28, 36), (50, 66)]
    r = etch.simulate(openings, domain_w=72, depth=14, dx=0.2, fluxes=f,
                      mask_thick=6.0)
    for line in r.report():
        print(line)
    os.makedirs("out", exist_ok=True)
    render(r, "out/etch_arde.png")
    print("wrote out/etch_arde.png")
