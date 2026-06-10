#!/usr/bin/env python3
"""3D anisotropic KOH etch of (100) silicon: a square mask opening forms the
classic inverted pyramid (flat {100} floor + four {111} sidewalls); a slot
forms a V-groove channel.  Writes PNG renders + an STL of the cavity.
Run: python examples/koh3d_demo.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import koh
from soidlc.render import render_mesh
from soidlc.exporters import write_stl

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    rec = koh.WetEtch(r100=1.0, r111=0.008, notch_w=8, steps=170)
    res = koh.simulate_3d(lambda x, y: 12 <= x <= 48 and 12 <= y <= 48,
                          60, 60, 30, dx=0.4, recipe=rec)
    m = koh.heightmap_mesh(res)
    render_mesh(m, {"DEVICE": (0.42, 0.55, 0.72)}, "out/koh3d_pyramid.png")
    write_stl(m, "out/koh3d_pyramid.stl")
    print("inverted pyramid: %dx%dx%d grid, %d tris -> out/koh3d_pyramid.png,"
          " .stl" % (res.nx, res.ny, res.nz, len(m.triangles)))
