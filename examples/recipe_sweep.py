#!/usr/bin/env python3
"""Sweep a real Bosch recipe parameter and watch the profile respond.

Run:  python examples/recipe_sweep.py
Shows how the passivation step time (t_pass) moves the sidewall taper from
re-entrant (bowing) through vertical to positive (narrowing) and finally into
the micro-masking / grass regime -- all from the machine parameter, via the
process-physics layer, through the level-set etch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import etch
from soidlc.recipe_physics import BoschProcess

if __name__ == "__main__":
    # pressure sweep: low pressure is anisotropic; high pressure scatters
    # ions and lowers passivation balance -> re-entrant bowing
    print("pressure  r_iso  passiv  bow   taper(deg)  notes")
    for pres in (20, 35, 50, 70, 95):
        p = BoschProcess(pressure=pres, t_pass=3.0, cycles=28)
        rec, warn = p.to_recipe()
        r = etch.simulate([(10, 18)], domain_w=36, depth=18, dx=0.1,
                          process=p, box_at=30)
        note = "bowing/undercut" if any("bowing" in w for w in warn) else ""
        print("%6d    %5.2f  %5.2f  %4.2f  %+7.1f    %s"
              % (pres, rec.r_iso, rec.passivation, rec.bow, r.tapers[0], note))

    print("\n# t_pass -> passivation balance -> grass risk (C4F8=140 sccm):")
    for t_pass in (3.0, 7.0, 13.0):
        p = BoschProcess(t_pass=t_pass, c4f8=140, cycles=28)
        rec, warn = p.to_recipe()
        flag = "GRASS RISK" if any("grass" in w for w in warn) else "ok"
        print("  t_pass=%4.1f s -> passivation=%.2f  [%s]"
              % (t_pass, rec.passivation, flag))
