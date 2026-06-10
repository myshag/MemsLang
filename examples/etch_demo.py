#!/usr/bin/env python3
"""Bosch DRIE level-set demo: an ARDE ladder etched toward a buried oxide.
Run:  python examples/etch_demo.py   (writes out/etch_arde.png)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import etch
from soidlc.render import write_png

def render(res, path, scale=4):
    nx, nz, dx = res.nx, res.nz, res.dx
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
    openings = [(4, 7), (14, 22), (34, 49), (64, 94)]
    r = etch.simulate(openings, domain_w=100, depth=24, dx=0.2,
                      recipe=etch.BoschRecipe(cycles=74, etch_per_cycle=0.6,
                                              footing=3.0))
    for l in r.report():
        print(l)
    os.makedirs("out", exist_ok=True)
    render(r, "out/etch_arde.png")
    print("wrote out/etch_arde.png")
