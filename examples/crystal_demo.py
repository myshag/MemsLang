"""Calibrated crystal etch-rate diagrams for wet etching of silicon.

Shows the data-driven anisotropy: measured low-index plane rates for named
KOH/TMAH conditions, reconstructed into a continuous R(n) diagram, rendered as
a stereographic map, and used to drive the 3D level-set etch.

Run:  python3 examples/crystal_demo.py
Writes docs/crystal_rate_koh.png, docs/crystal_rate_tmah.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import crystal_rates as CR                      # noqa: E402
from soidlc import koh                                      # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUT, exist_ok=True)

PLANES = [(1, 0, 0), (1, 1, 0), (3, 1, 1), (2, 1, 1), (1, 1, 1)]

for cond in ("KOH_30_70", "TMAH_25_80"):
    d = CR.RateDiagram.from_condition(cond)
    print("\n%s   (anchors -> %d symmetry-equivalent normals)"
          % (d.name, len(d.points)))
    for hkl in PLANES:
        print("   R%-9s = %7.4f um/min   (%.2fx R100)"
              % (str(hkl), d.rate(hkl), d.rate(hkl) / d.rate((1, 0, 0))))
    print("   anisotropy R100/R111 = %.0f" % (d.rate((1, 0, 0)) / d.rate((1, 1, 1))))

# stereographic rate-diagram PNGs
CR.render_diagram(CR.RateDiagram.from_condition("KOH_30_70"),
                  os.path.join(OUT, "crystal_rate_koh.png"))
CR.render_diagram(CR.RateDiagram.from_condition("TMAH_25_80"),
                  os.path.join(OUT, "crystal_rate_tmah.png"))


def depth(res, x, y):
    nx, ny, nz, dx, j0 = res.nx, res.ny, res.nz, res.dx, res.j0
    ix, jy = int(x / dx), int(y / dx)
    kd = j0
    for k in range(j0, nz):
        if res.phi[ix + nx * (jy + ny * k)] > 0:
            kd = k
    return (kd - j0) * dx


# drive the 3D etch from the calibrated table, on two wafer orientations
d = CR.RateDiagram.from_condition("KOH_30_70")
slot = lambda x, y: 20 <= x <= 40
print("\ntable-driven 3D etch of a 20 um slot (140 steps):")
for orient in ("100", "110"):
    r = koh.simulate_3d(slot, 60, 60, 40, dx=0.7, diagram=d,
                        orientation=orient, recipe=koh.WetEtch(steps=140))
    print("   (%s) wafer: center depth %.1f um" % (orient, depth(r, 30, 30)))
print("   -> (110) self-terminates slower: its {111} walls are vertical")

# --- degenerate cases: mask misalignment and an off-axis (miscut) wafer -----
def floor_width(res, y):
    nx, ny, nz, dx, j0 = res.nx, res.ny, res.nz, res.dx, res.j0
    jy = int(y / dx)
    kd = j0
    for k in range(j0, nz):
        if any(res.phi[ix + nx * (jy + ny * k)] > 0 for ix in range(nx)):
            kd = k
    return sum(1 for ix in range(nx)
               if res.phi[ix + nx * (jy + ny * kd)] > 0) * dx


sq = lambda x, y: 18 <= x <= 42 and 18 <= y <= 42
print("\nmask misaligned off the <110> flat (24 um square, (100), 130 steps):")
for a in (0, 15, 30, 45):
    r = koh.simulate_3d(sq, 60, 60, 30, dx=0.6, diagram=d, orientation="100",
                        misalign_deg=a, recipe=koh.WetEtch(steps=130))
    print("   %2d deg: floor still %4.1f um wide  (0 deg pinches to a pyramid;"
          " misaligned won't self-terminate)" % (a, floor_width(r, 30)))

# stereographic diagrams of the three cases (poles move w.r.t. the surface)
CR.render_diagram(d, os.path.join(OUT, "crystal_aligned.png"))
CR.render_diagram(d, os.path.join(OUT, "crystal_misalign.png"), misalign_deg=22)
CR.render_diagram(d, os.path.join(OUT, "crystal_miscut.png"),
                  miscut_deg=10, miscut_az=0)

print("\nwrote docs/crystal_rate_koh.png, crystal_rate_tmah.png,")
print("      crystal_aligned.png, crystal_misalign.png, crystal_miscut.png")
