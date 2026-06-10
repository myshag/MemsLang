#!/usr/bin/env python3
"""Anisotropic KOH wet etch of (100) silicon: a wide opening gives a
truncated pyramid (flat {100} floor, 54.74 deg {111} walls); a narrow one
self-terminates into a V-groove.  Run: python examples/koh_demo.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import koh
from soidlc.render import write_png

def render(res, path, scale=6):
    nx, nz, dx = res.nx, res.nz, res.dx
    W, H = nx*scale, nz*scale
    img = bytearray([24]*(3*W*H))
    SI, VOID = (70,90,120), (18,18,24)
    for j in range(nz):
        for i in range(nx):
            c = VOID if res.phi[i+nx*j] > 0 else SI
            for dy in range(scale):
                for dx2 in range(scale):
                    o = ((j*scale+dy)*W+(i*scale+dx2))*3; img[o:o+3] = bytes(c)
    write_png(path, W, H, img)

if __name__ == "__main__":
    rec = koh.WetEtch(r100=1.0, r110=1.0, r111=0.008, notch_w=8, steps=210)
    rw = koh.simulate([(16, 44)], domain_w=60, depth=26, dx=0.2, recipe=rec)
    for l in rw.report(): print(l)
    os.makedirs("out", exist_ok=True); render(rw, "out/koh_pit.png")
    rv = koh.simulate([(24, 36)], domain_w=60, depth=26, dx=0.2,
                      recipe=koh.WetEtch(r100=1.0,r110=1.0,r111=0.008,
                                         notch_w=8,steps=300))
    for l in rv.report(): print(l)
    render(rv, "out/koh_vgroove.png"); print("wrote out/koh_*.png")
