#!/usr/bin/env python3
"""Multi-wafer assembly: bond a cap wafer over the comb resonator and show how
the sealed cavity changes Q.  Writes a cross-section PNG.
Run: python examples/wafer_demo.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import compile_source, assembly
from soidlc.assembly import Cap

if __name__ == "__main__":
    art = compile_source(open("examples/comb_resonator.soidl").read())
    res, proc, model = art.result, art.process, art.model
    os.makedirs("out", exist_ok=True)
    assembly.cross_section(res, proc, Cap(gap=6, ring_w=40, cap_thick=120,
                                          pressure=1.0), "out/wafer_xsec.png")
    print("wrote out/wafer_xsec.png  (HANDLE / suspended DEVICE / sealed "
          "cavity / BOND ring / CAP)\n")
    print("Effect of wafer-level packaging on Q:")
    for lbl, c in [("uncapped, 1 atm", None),
                   ("capped, vacuum-sealed", Cap(gap=6, pressure=1.0)),
                   ("capped, 1 atm, 6 um gap", Cap(gap=6, pressure=1.01325e5)),
                   ("capped, 1 atm, 3 um gap", Cap(gap=3, pressure=1.01325e5))]:
        qd = assembly.capped_quality(res, proc, model, c)
        q = qd.get("capped", qd.get("uncapped_air"))
        print("  %-26s Q ~ %.0f" % (lbl, q))
