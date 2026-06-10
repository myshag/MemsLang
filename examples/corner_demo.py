"""Corner-compensation demo for anisotropic (KOH) bulk micromachining.

A square silicon mesa has four convex corners.  A timed (100) etch undercuts
every convex corner along the fast <410> planes, beveling them off.  Adding a
<110> compensation square at each corner feeds the fast plane a sacrificial
block, so the real corner survives to the end of the etch.

Run:  python3 examples/corner_demo.py
Writes docs/corner_bare.png and docs/corner_compensated.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import corner as C                              # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUT, exist_ok=True)

# 100 um square mesa centred in a 200 um field; corners at the four (x,y).
X0, X1 = 50.0, 150.0
CORNERS = [(X0, X0), (X1, X0), (X0, X1), (X1, X1)]
OUT_DIRS = [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]

rec = C.UndercutEtch(depth=20.0, ratio=1.5, dx=0.6)
U = rec.r_fast
need = (2 ** 0.5) * U
print("anisotropic etch: depth %.0f um, convex-corner undercut budget %.1f um"
      % (rec.depth, U))
print("a <110> compensation square protects when its side > sqrt(2)*U = %.1f um"
      % need)

# ---- (1) bare mesa: corners get undercut -------------------------------
bare = C.Mask().add_rect(X0, X0, X1, X1)
rb = C.simulate(bare, 200, 200, rec)
print("\nbare mesa:")
for c in CORNERS:
    print("  corner %-12s survives? %s" % (c, rb.is_silicon(*c)))
C.render(rb, os.path.join(OUT, "corner_bare.png"), marks=CORNERS)

# ---- (2) compensated mesa: a square at each convex corner --------------
S = round(need + 6, 0)                      # a little over threshold
comp = C.Mask().add_rect(X0, X0, X1, X1)
for c in CORNERS:
    comp.add(C.square_comp(c, S))
rc = C.simulate(comp, 200, 200, rec)
print("\ncompensated mesa (square side %.0f um):" % S)
for c in CORNERS:
    print("  corner %-12s survives? %s" % (c, rc.is_silicon(*c)))
C.render(rc, os.path.join(OUT, "corner_compensated.png"), marks=CORNERS)

# ---- (3) the design curve: required square vs etch depth ---------------
print("\nrequired compensation square vs etch depth:")
for d in (10, 20, 30, 40):
    r = C.UndercutEtch(depth=d, ratio=1.5, dx=0.8)
    print("  depth %2d um -> undercut %4.1f um -> square >= %4.1f um"
          % (d, r.r_fast, (2 ** 0.5) * r.r_fast))

print("\nwrote docs/corner_bare.png and docs/corner_compensated.png")
