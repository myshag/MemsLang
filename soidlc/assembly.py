"""Multi-wafer assembly: bond a cap wafer over a device wafer.

The key point is not just stacking geometry -- a bonded cap encloses the
moving structure in a sealed cavity, which changes the gas environment and
therefore the damping/Q:

  * uncapped, in air        -> slide film under the device (BOX gap) + combs;
  * capped at 1 atm, small gap -> an extra slide film over the top -> Q drops;
  * capped and vacuum-sealed -> the gas is pumped out -> Q rises sharply
    (limited by anchor / thermo-elastic loss, not modelled here).

So this module assembles the 3D stack *and* recomputes Q for the sealed
device -- the reason vacuum encapsulation exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import geometry as G
from . import mesh as M
from .elaborate import InstanceResult, ProcessInfo

MU_AIR = 1.85e-5            # Pa*s
P_ATM = 1.01325e5          # Pa
Q_FLOOR = 50000.0          # anchor/TED-limited ceiling when gas is removed

CAP_COLORS = {"CAP": (0.55, 0.6, 0.7), "BOND": (0.85, 0.7, 0.3)}


@dataclass
class Cap:
    gap: float = 6.0           # um: cavity height above the device top
    ring_w: float = 40.0       # um: bond-ring (seal) width
    cap_thick: float = 150.0   # um: cap-wafer roof thickness
    pad: float = 30.0          # um: footprint margin
    pressure: float = P_ATM    # Pa: sealed cavity pressure
    material: str = "Si"


def cap_mesh(bbox: Tuple[float, float, float, float], device_top: float,
             cap: Cap) -> M.Mesh:
    """A cap wafer: a sealing bond ring (walls) + a roof, recessed to leave a
    ``cap.gap`` cavity over the device."""
    x0, y0, x1, y1 = bbox
    ox0, oy0 = x0 - cap.pad, y0 - cap.pad
    W, H = (x1 - x0) + 2 * cap.pad, (y1 - y0) + 2 * cap.pad
    outer = G.rect_corner(ox0, oy0, W, H)
    rw = cap.ring_w
    cavity = G.rect_corner(ox0 + rw, oy0 + rw, W - 2 * rw, H - 2 * rw)
    ring = G.Polygon(outer.exterior, [list(reversed(cavity.exterior))])

    out = M.Mesh()
    z_wall0, z_wall1 = device_top, device_top + cap.gap
    out.extend(M.extrude_polygon(ring, z_wall0, z_wall1, "BOND"))
    out.extend(M.extrude_polygon(outer, z_wall1, z_wall1 + cap.cap_thick, "CAP"))
    return out


def assemble(device_mesh: M.Mesh, result: InstanceResult, proc: ProcessInfo,
             cap: Optional[Cap] = None) -> M.Mesh:
    """Combine the device-wafer mesh with a bonded cap wafer."""
    cap = cap or Cap()
    dev = proc.device()
    device_top = dev.z1 if dev else 0.0
    out = M.Mesh()
    out.extend(device_mesh)
    out.extend(cap_mesh(result.bbox, device_top, cap))
    return out


# ---------------------------------------------------------------------------
# damping / Q with and without a cap
# ---------------------------------------------------------------------------
def _released_area(result: InstanceResult) -> float:
    return sum(s.polygon.area() for s in result.shapes
              if s.layer == "DEVICE" and s.mech == "released") * 1e-12   # m^2


def capped_quality(result: InstanceResult, proc: ProcessInfo,
                   model: Dict[str, object], cap: Optional[Cap]) -> dict:
    """Return Q for the (un)capped device.  ``model`` carries m and f0."""
    from .units import Quantity
    m = model.get("m")
    f0 = model.get("f0")
    if not (isinstance(m, Quantity) and isinstance(f0, Quantity)):
        return {}
    m, f0 = m.value, f0.value
    w = 2 * math.pi * f0
    A = _released_area(result)
    box = proc.box()
    g_box = (box.thickness if box else 2.0) * 1e-6

    # slide-film damping at 1 atm: bottom always present
    b_bottom = MU_AIR * A / g_box
    out = {}

    def Q_of(b):
        b = max(b, m * w / Q_FLOOR)        # anchor-limited ceiling
        return m * w / b

    out["uncapped_air"] = Q_of(b_bottom)
    if cap is not None:
        b_top_atm = MU_AIR * A / (cap.gap * 1e-6)
        pr = cap.pressure / P_ATM          # rarefied: mu_eff ~ P (linear)
        b_capped = pr * (b_bottom + b_top_atm)
        out["capped"] = Q_of(b_capped)
        out["cap_gap_um"] = cap.gap
        out["cap_pressure_pa"] = cap.pressure
    return out


def report(qd: dict) -> List[str]:
    lines = []
    if "uncapped_air" in qd:
        lines.append("cap    uncapped (1 atm): Q ~ %.0f" % qd["uncapped_air"])
    if "capped" in qd:
        lines.append(
            "cap    capped (gap %.1f um, %.0f Pa): Q ~ %.0f" % (
                qd["cap_gap_um"], qd["cap_pressure_pa"], qd["capped"]))
    return lines


# ---------------------------------------------------------------------------
# schematic cross-section of the bonded stack (y-slice through the middle)
# ---------------------------------------------------------------------------
_XS_COLORS = {
    "HANDLE": (110, 110, 120), "BOX": (150, 120, 60),
    "DEVICE": (70, 110, 170), "METAL": (210, 180, 70),
    "BOND": (217, 178, 76), "CAP": (140, 150, 180),
}


def cross_section(result: InstanceResult, proc: ProcessInfo, cap: Cap,
                  path: str, width_px: int = 620) -> None:
    """Render a labelled y-mid cross-section of the capped device to ``path``."""
    from .render import write_png
    x0, y0, x1, y1 = result.bbox
    ym = 0.5 * (y0 + y1)
    dev, box, handle = proc.device(), proc.box(), proc.handle()
    device_top = dev.z1
    zw1 = device_top + cap.gap
    X0, X1 = x0 - cap.pad - 12, x1 + cap.pad + 12
    Z1 = zw1 + cap.cap_thick + 12
    Wum, Hum = X1 - X0, Z1
    W = width_px
    H = max(1, int(W * Hum / Wum))
    img = bytearray([20, 20, 26] * (W * H))

    spans = []
    for s in result.shapes:
        if s.layer != "DEVICE":
            continue
        bx0, by0, bx1, by1 = s.polygon.bbox()
        if by0 <= ym <= by1:
            spans.append((bx0, bx1, s.mech == "anchored"))

    def dev_at(x):
        anc = None
        for a, b, an in spans:
            if a <= x <= b:
                if an:
                    return True
                anc = False
        return anc

    rw = cap.ring_w
    for px in range(W):
        x = X0 + (px + 0.5) * Wum / W
        anc = dev_at(x)
        in_foot = (x0 - cap.pad <= x <= x1 + cap.pad)
        in_ring = in_foot and (x <= x0 - cap.pad + rw or x >= x1 + cap.pad - rw)
        for pz in range(H):
            z = Z1 - (pz + 0.5) * Hum / H
            mat = None
            if z <= handle.z1:
                mat = "HANDLE"
            if box and box.z0 <= z <= box.z1 and anc:
                mat = "BOX"
            if dev.z0 <= z <= dev.z1 and anc is not None:
                mat = "DEVICE"
            if in_foot and zw1 <= z <= zw1 + cap.cap_thick:
                mat = "CAP"
            if in_ring and device_top <= z <= zw1:
                mat = "BOND"
            if mat:
                o = (pz * W + px) * 3
                img[o:o + 3] = bytes(_XS_COLORS[mat])
    write_png(path, W, H, img)

