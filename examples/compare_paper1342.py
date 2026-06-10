#!/usr/bin/env python3
"""Compile examples/gyroscope.soidl and print a comparison table against the
published device (Sharaf et al., NSTI-Nanotech 2010, pp. 386-389).

Usage:  python examples/compare_paper1342.py
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import compile_source                              # noqa: E402

PAPER = {
    "drive_hz": 16769.0, "sense_hz": 16797.0, "split_pct": 0.08,
    "drive_cap_ff": 239.0, "sense_cap_ff": 274.5,
    "q_air": 10.0, "area_mm2": 1.6 * 1.6,
}


def main():
    src = open(os.path.join(os.path.dirname(__file__), "gyroscope.soidl")).read()
    art = compile_source(src, fem=True, fem_h=18, fem_closure=True)
    rep = art.report

    modes = re.search(r"island #1 modes: ([\d.]+) kHz, ([\d.]+) kHz",
                      "\n".join(rep))
    f1, f2 = float(modes.group(1)) * 1e3, float(modes.group(2)) * 1e3
    caps = {}
    b = None
    for l in rep:
        m = re.search(r"transducer \w+ \(net (\w+)\).*C0=([\d.]+) fF", l)
        if m:
            caps.setdefault(m.group(1), []).append(float(m.group(2)))
        mb = re.search(r"b=([\d.eE+-]+) N\*s/m", l)
        if mb:
            b = float(mb.group(1))
    mass = art.model["m"].value
    q_air = mass * 2 * math.pi * f1 / b
    x0, y0, x1, y1 = art.result.bbox
    area = (x1 - x0) * (y1 - y0) * 1e-6
    drive_cap = sum(caps.get("DRIVE", [0]))
    sense_cap = sum(caps.get("SENSE", [0]))
    split = abs(f2 - f1) / min(f1, f2) * 100

    def row(name, paper, ours, unit=""):
        print(f"  {name:<26} {paper:>14}   {ours:>14}   {unit}")

    print("\n  %-26s %14s   %14s\n  %s" % ("quantity", "paper", "soidlc",
                                           "-" * 60))
    row("drive mode freq", f"{PAPER['drive_hz']:.0f}", f"{f1:.0f}", "Hz")
    row("sense mode freq", f"{PAPER['sense_hz']:.0f}", f"{f2:.0f}", "Hz")
    row("mode split", f"{PAPER['split_pct']:.2f}", f"{split:.1f}", "%")
    row("drive capacitance", f"{PAPER['drive_cap_ff']:.1f}",
        f"{drive_cap:.1f}", "fF")
    row("sense capacitance", f"{PAPER['sense_cap_ff']:.1f}",
        f"{sense_cap:.1f}", "fF")
    row("Q (air)", f"~{PAPER['q_air']:.0f}", f"{q_air:.0f}", "")
    row("die area", f"{PAPER['area_mm2']:.2f}", f"{area:.2f}", "mm^2")
    print()
    for e in art.errors:
        print("  ERROR:", e)


if __name__ == "__main__":
    main()
