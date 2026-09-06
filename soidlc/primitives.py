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
    # finger tips must not reach the opposing backbone: leave a tip gap = g
    D = Lf + g                # distance between the two backbone inner faces

    # rotor backbone (released) on the left, stator backbone (anchored) right
    rotor_x = -D / 2 - bar / 2
    stator_x = D / 2 + bar / 2
    shapes.append(G.Shape(ctx.device_layer,
                          G.rect(bar, span, rotor_x, 0), "comb_rotor_bar",
                          mech="released"))
    shapes.append(G.Shape(ctx.device_layer,
                          G.rect(bar, span, stator_x, 0), "comb_stator_bar",
                          mech="anchored"))

    y = -span / 2 + pitch / 2
    for i in range(N):
        # rotor finger reaches right from the rotor bar
        rf = G.rect(Lf, wf, -D / 2 + Lf / 2, y)
        shapes.append(G.Shape(ctx.device_layer, rf, "rotor_finger",
                              mech="released"))
        # stator finger reaches left, offset by half a pitch
        sf = G.rect(Lf, wf, D / 2 - Lf / 2, y + (wf + g))
        shapes.append(G.Shape(ctx.device_layer, sf, "stator_finger",
                              mech="anchored"))
        y += 2 * (wf + g)

    out = shapes
    if direction == "y":
        out = [G.Shape(s.layer, s.polygon.rotated(90), s.label, s.mech)
               for s in shapes]
    return out


def prim_meander(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """Serpentine spring as ONE continuous folded path.

    Drawn as a single wire rather than a row of beams on purpose.  A meander
    built from separate spans that all reach into the proof mass puts them in
    parallel instead of in series, and if the last span is then anchored the
    mass is bolted to the substrate: one legal electrical island, no LVS
    complaint, and a resonator an order of magnitude too stiff.  A single path
    cannot be wired wrong -- only its first point touches the mass.

    Spans run along y at `pitch` intervals; the path alternates between the
    far and near ends, and always finishes at the far end so its anchor lands
    well clear of the mass.
    """
    L = _um(_arg(args, kwargs, 0, "L", Quantity(120e-6, (1, 0, 0, 0))))
    w = _um(_arg(args, kwargs, 1, "w", Quantity(3e-6, (1, 0, 0, 0))))
    n = int(round(_num(_arg(args, kwargs, 2, "n_turns", 4))))
    pitch = _um(_arg(args, kwargs, 3, "pitch", Quantity(12e-6, (1, 0, 0, 0))))
    if n < 2:
        raise ValueError("meander() needs at least 2 turns")

    y_far = 2.0 - L
    y_near = -6.0          # stops short of the mass edge at y = 0

    pts = [(0.0, 2.0)]     # the only point that reaches into the mass
    for i in range(n):
        far = (i % 2 == 0) or (i == n - 1)
        y = y_far if far else y_near
        pts.append((i * pitch, y))
        if i < n - 1:
            pts.append(((i + 1) * pitch, y))

    shapes = [G.Shape(ctx.device_layer, G.wire(pts, w), "meander",
                      mech="released")]
    # anchor caps the far end of the last span, away from the mass
    shapes.append(G.Shape(
        ctx.device_layer,
        G.rect(max(pitch, 16.0), 20.0, (n - 1) * pitch, y_far - 8.0),
        "anchor", mech="anchored"))
    return shapes


def prim_parallel_plate(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """Gap-closing electrode pairs: released plates facing anchored ones.

    Unlike a comb, whose capacitance is linear in displacement and whose force
    is therefore constant with position, a parallel plate's capacitance goes as
    1/(g - x).  The force rises as the gap closes, and past x = g/3 the
    mechanical restoring force loses the race and the pair snaps shut.  That
    pull-in is the whole reason RF switches are built this way -- and the
    reason a gap-closing sensor must stay below it.  metrics.pull_in()
    computes the collapse voltage.
    """
    W = _um(_arg(args, kwargs, 0, "W", Quantity(100e-6, (1, 0, 0, 0))))
    H = _um(_arg(args, kwargs, 1, "H", Quantity(40e-6, (1, 0, 0, 0))))
    g = _um(_arg(args, kwargs, 2, "g", Quantity(3e-6, (1, 0, 0, 0))))
    n = int(round(_num(_arg(args, kwargs, 3, "n", 1))))
    if g <= 0:
        raise ValueError("parallel_plate() needs a positive gap")

    shapes: List[G.Shape] = []
    pitch = 2 * H + 2 * g
    y = -(n - 1) * pitch / 2.0
    for _ in range(n):
        shapes.append(G.Shape(ctx.device_layer, G.rect(W, H, 0.0, y),
                              "rotor_plate", mech="released"))
        shapes.append(G.Shape(ctx.device_layer,
                              G.rect(W, H, 0.0, y + H + g),
                              "stator_plate", mech="anchored"))
        y += pitch
    return shapes


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
    "meander": prim_meander,
    "parallel_plate": prim_parallel_plate,
    "gap_stop": prim_gap_stop,
    "trench": prim_trench,
    "via_metal": prim_via_metal,
}


def is_primitive(name: str) -> bool:
    return name in PRIMITIVES
