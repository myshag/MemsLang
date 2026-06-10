"""Geometry generators for SOIDL primitives.

Each primitive takes already-evaluated arguments (in :class:`Quantity` form)
plus a :class:`PrimitiveCtx` carrying process defaults, and returns a list of
:class:`Shape` in micrometres, centred on the local origin.  Placement
(translation/rotation) is applied by the elaborator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import geometry as G
from .units import Quantity


@dataclass
class PrimitiveCtx:
    device_layer: str = "DEVICE"
    metal_layer: str = "METAL"
    hole_size: float = 5.0       # um
    hole_pitch: float = 25.0     # um
    max_solid_span: float = 30.0  # um


def _um(v, default: Optional[float] = None) -> float:
    if v is None:
        if default is None:
            raise ValueError("missing required length argument")
        return default
    if isinstance(v, Quantity):
        return v.um
    return float(v)


def _num(v, default: Optional[float] = None) -> float:
    if v is None:
        return float(default)
    if isinstance(v, Quantity):
        return v.as_float()
    return float(v)


def _arg(args, kwargs, idx, name, default=None):
    if name in kwargs:
        return kwargs[name]
    if idx < len(args):
        return args[idx]
    return default


# ---------------------------------------------------------------------------
def prim_beam(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    L = _um(_arg(args, kwargs, 0, "L"))
    w = _um(_arg(args, kwargs, 1, "w"))
    direction = _dir(_arg(args, kwargs, 2, "dir", "y"))
    if direction == "x":
        poly = G.rect(L, w)
    else:
        poly = G.rect(w, L)
    return [G.Shape(ctx.device_layer, poly, "beam", mech="released")]


def prim_anchor(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    w = _um(_arg(args, kwargs, 0, "w"))
    h = _um(_arg(args, kwargs, 1, "h"))
    return [G.Shape(ctx.device_layer, G.rect(w, h), "anchor", mech="anchored")]


def prim_plate(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    W = _um(_arg(args, kwargs, 0, "W"))
    H = _um(_arg(args, kwargs, 1, "H"))
    holes_arg = _arg(args, kwargs, 2, "holes", "auto")
    poly = G.rect(W, H)
    want_holes = (holes_arg == "auto" or holes_arg is True
                  or (isinstance(holes_arg, str) and holes_arg == "auto"))
    if want_holes and (W > ctx.max_solid_span or H > ctx.max_solid_span):
        poly = G.Polygon(poly.exterior, _hole_grid(W, H, ctx))
    return [G.Shape(ctx.device_layer, poly, "plate", mech="released")]


def _hole_grid(W: float, H: float, ctx: PrimitiveCtx) -> List[G.Ring]:
    holes: List[G.Ring] = []
    s = ctx.hole_size
    p = ctx.hole_pitch
    # keep a solid margin around the rim
    margin = max(p, s * 2)
    x = -W / 2 + margin
    while x <= W / 2 - margin:
        y = -H / 2 + margin
        while y <= H / 2 - margin:
            holes.append([
                (x - s / 2, y - s / 2),
                (x + s / 2, y - s / 2),
                (x + s / 2, y + s / 2),
                (x - s / 2, y + s / 2),
            ])
            y += p
        x += p
    return holes


def prim_comb(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    N = int(round(_num(_arg(args, kwargs, 0, "N", 10))))
    Lf = _um(_arg(args, kwargs, 1, "Lf", Quantity(40e-6, (1, 0, 0, 0))))
    wf = _um(_arg(args, kwargs, 2, "wf", Quantity(3e-6, (1, 0, 0, 0))))
    g = _um(_arg(args, kwargs, 3, "g", Quantity(2e-6, (1, 0, 0, 0))))
    ov = _um(_arg(args, kwargs, 4, "ov", Quantity(20e-6, (1, 0, 0, 0))))
    direction = _dir(_arg(args, kwargs, 5, "dir", "x"))

    shapes: List[G.Shape] = []
    pitch = wf + g            # one rotor + gap
    span = N * (2 * (wf + g))
    bar = 8.0                 # backbone bar width (um)

    # rotor backbone (released) on the left, stator backbone (anchored) right
    rotor_x = -Lf / 2 - bar / 2
    stator_x = Lf / 2 + bar / 2
    shapes.append(G.Shape(ctx.device_layer,
                          G.rect(bar, span, rotor_x, 0), "comb_rotor_bar",
                          mech="released"))
    shapes.append(G.Shape(ctx.device_layer,
                          G.rect(bar, span, stator_x, 0), "comb_stator_bar",
                          mech="anchored"))

    y = -span / 2 + pitch / 2
    for i in range(N):
        # rotor finger reaches right from the rotor bar
        rf = G.rect(Lf, wf, rotor_x + bar / 2 + Lf / 2, y)
        shapes.append(G.Shape(ctx.device_layer, rf, "rotor_finger",
                              mech="released"))
        # stator finger reaches left, offset by half a pitch
        sf = G.rect(Lf, wf, stator_x - bar / 2 - Lf / 2, y + (wf + g))
        shapes.append(G.Shape(ctx.device_layer, sf, "stator_finger",
                              mech="anchored"))
        y += 2 * (wf + g)

    out = shapes
    if direction == "y":
        out = [G.Shape(s.layer, s.polygon.rotated(90), s.label, s.mech)
               for s in shapes]
    return out


def prim_gap_stop(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    d = _um(_arg(args, kwargs, 0, "d"))
    return [G.Shape(ctx.device_layer, G.rect(max(d, 2.0), max(d, 2.0)),
                    "gap_stop", mech="anchored")]


def prim_trench(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    # an isolation trench is an *absence* of silicon; not extruded as solid.
    return []


def prim_via_metal(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    w = _um(_arg(args, kwargs, 0, "w", Quantity(80e-6, (1, 0, 0, 0))))
    h = _um(_arg(args, kwargs, 1, "h", Quantity(80e-6, (1, 0, 0, 0))))
    return [G.Shape(ctx.metal_layer, G.rect(w, h), "via_metal",
                    mech="anchored")]


def _dir(v) -> str:
    if v is None:
        return "y"
    if hasattr(v, "id"):       # AST Name leaked through
        return v.id
    return str(v)


PRIMITIVES = {
    "beam": prim_beam,
    "anchor": prim_anchor,
    "plate": prim_plate,
    "comb": prim_comb,
    "combdrive": prim_comb,    # combdrive renders its comb geometry
    "gap_stop": prim_gap_stop,
    "trench": prim_trench,
    "via_metal": prim_via_metal,
}


def is_primitive(name: str) -> bool:
    return name in PRIMITIVES
