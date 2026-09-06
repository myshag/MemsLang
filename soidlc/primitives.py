"""Geometry generators for SOIDL primitives.

Each primitive takes already-evaluated arguments (in :class:`Quantity` form)
plus a :class:`PrimitiveCtx` carrying process defaults, and returns a list of
:class:`Shape` in micrometres, centred on the local origin.  Placement
(translation/rotation) is applied by the elaborator.
"""

from __future__ import annotations

import math
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

    Laid out like a comb -- all rotor plates on one backbone, all stators on
    the other -- so `attach (rotor -> M.top)` has a single rotor face to abut.
    Interleaving them instead leaves the rotor a floating island.
    """
    W = _um(_arg(args, kwargs, 0, "W", Quantity(100e-6, (1, 0, 0, 0))))
    H = _um(_arg(args, kwargs, 1, "H", Quantity(40e-6, (1, 0, 0, 0))))
    g = _um(_arg(args, kwargs, 2, "g", Quantity(3e-6, (1, 0, 0, 0))))
    n = int(round(_num(_arg(args, kwargs, 3, "n", 1))))
    if g <= 0:
        raise ValueError("parallel_plate() needs a positive gap")
    if n < 1:
        raise ValueError("parallel_plate() needs at least one pair")

    bar = 8.0
    pitch_x = W + 8.0
    total = n * pitch_x
    y_r = -(g / 2.0 + H / 2.0)          # rotor plates below the gap
    y_s = +(g / 2.0 + H / 2.0)          # stator plates above it

    shapes: List[G.Shape] = [
        G.Shape(ctx.device_layer,
                G.rect(total, bar, 0.0, y_r - H / 2.0 - bar / 2.0),
                "plate_rotor_bar", mech="released"),
        G.Shape(ctx.device_layer,
                G.rect(total, bar, 0.0, y_s + H / 2.0 + bar / 2.0),
                "plate_stator_bar", mech="anchored"),
    ]
    for i in range(n):
        x = -total / 2.0 + pitch_x / 2.0 + i * pitch_x
        shapes.append(G.Shape(ctx.device_layer, G.rect(W, H, x, y_r),
                              "rotor_plate", mech="released"))
        shapes.append(G.Shape(ctx.device_layer, G.rect(W, H, x, y_s),
                              "stator_plate", mech="anchored"))
    return shapes


def prim_chevron(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """V-beam thermal actuator: n pairs of beams inclined by `angle` degrees
    from two anchors in to a central shuttle.

    Current through the beams heats them and they expand.  Because they are
    pre-inclined, that expansion resolves into shuttle motion along +y instead
    of buckling in a direction nobody chose.  Large force, small stroke -- the
    opposite trade to a comb drive, which is why grippers and latches use
    chevrons and resonators do not.
    """
    n = int(round(_num(_arg(args, kwargs, 0, "n", 4))))
    L = _um(_arg(args, kwargs, 1, "L", Quantity(200e-6, (1, 0, 0, 0))))
    w = _um(_arg(args, kwargs, 2, "w", Quantity(6e-6, (1, 0, 0, 0))))
    angle = _num(_arg(args, kwargs, 3, "angle", 6.0))     # degrees
    if n < 1:
        raise ValueError("chevron() needs at least one beam pair")

    a = math.radians(angle)
    dx = L * math.cos(a)
    dy = L * math.sin(a)
    shuttle_w = 20.0
    pitch = 3 * w + 10.0

    shapes: List[G.Shape] = []
    y = -(n - 1) * pitch / 2.0
    for _ in range(n):
        # each beam runs from its anchor up to the shuttle centreline
        shapes.append(G.Shape(
            ctx.device_layer, G.wire([(-dx, y), (0.0, y + dy)], w),
            "chevron_beam", mech="released"))
        shapes.append(G.Shape(
            ctx.device_layer, G.wire([(dx, y), (0.0, y + dy)], w),
            "chevron_beam", mech="released"))
        y += pitch

    # shuttle spans every beam tip; tips sit at y + dy, so the bar must cover
    # the full tip range, not just the anchor range
    span = (n - 1) * pitch + 4 * w
    shapes.append(G.Shape(
        ctx.device_layer, G.rect(shuttle_w, span, 0.0, dy),
        "chevron_shuttle", mech="released"))
    # anchors at both ends of every beam row
    anc_h = (n - 1) * pitch + 4 * w
    for sign in (-1.0, 1.0):
        shapes.append(G.Shape(
            ctx.device_layer,
            G.rect(30.0, anc_h, sign * (dx + 15.0 - w), 0.0),
            "chevron_anchor", mech="anchored"))
    return shapes


def prim_hot_arm(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """U-shaped hot-arm / cold-arm thermal actuator.

    Both arms carry the same current, but the thin one has the higher
    resistance per unit length, so it runs hotter and expands more.  The pair
    is joined at the tip, so the difference bends the actuator toward the cold
    arm -- it traces an arc, not a translation.
    """
    L = _um(_arg(args, kwargs, 0, "L", Quantity(200e-6, (1, 0, 0, 0))))
    w_hot = _um(_arg(args, kwargs, 1, "w_hot", Quantity(3e-6, (1, 0, 0, 0))))
    w_cold = _um(_arg(args, kwargs, 2, "w_cold", Quantity(12e-6, (1, 0, 0, 0))))
    g = _um(_arg(args, kwargs, 3, "g", Quantity(4e-6, (1, 0, 0, 0))))
    if w_hot >= w_cold:
        raise ValueError("hot_arm() needs w_hot < w_cold: the asymmetry is "
                         "what makes it an actuator")

    y_hot = (w_hot + g) / 2.0
    y_cold = -(w_cold + g) / 2.0
    span = abs(y_hot - y_cold) + w_cold
    return [
        G.Shape(ctx.device_layer, G.rect(L, w_hot, 0.0, y_hot),
                "hot_arm", mech="released"),
        G.Shape(ctx.device_layer, G.rect(L, w_cold, 0.0, y_cold),
                "cold_arm", mech="released"),
        # tip yoke joining the two arms
        G.Shape(ctx.device_layer,
                G.rect(w_cold, span, L / 2.0 - w_cold / 2.0,
                       (y_hot + y_cold) / 2.0),
                "hot_arm_yoke", mech="released"),
        # anchors at the driven end, one per arm (current in one, out the other)
        G.Shape(ctx.device_layer, G.rect(24.0, 24.0, -L / 2.0 - 8.0, y_hot),
                "hot_arm_anchor", mech="anchored"),
        G.Shape(ctx.device_layer, G.rect(24.0, 24.0, -L / 2.0 - 8.0, y_cold),
                "hot_arm_anchor", mech="anchored"),
    ]


def prim_gap_stop(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    d = _um(_arg(args, kwargs, 0, "d"))
    return [G.Shape(ctx.device_layer, G.rect(max(d, 2.0), max(d, 2.0)),
                    "gap_stop", mech="anchored")]


def prim_trench(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A backside etch opening: an *absence* of substrate, not silicon.

    Emitted on the pseudo-layer TRENCH, which is deliberately not part of the
    process stack.  build3d subtracts these footprints from the HANDLE slab
    and from the oxide beneath them instead of extruding them as solid, and
    the connectivity extractor ignores them because they are not DEVICE -- so
    a trench can never create or break an electrical island.
    """
    W = _um(_arg(args, kwargs, 0, "W", Quantity(200e-6, (1, 0, 0, 0))))
    H = _um(_arg(args, kwargs, 1, "H", Quantity(200e-6, (1, 0, 0, 0))))
    return [G.Shape("TRENCH", G.rect(W, H), "trench", mech="anchored")]


def prim_via_metal(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    w = _um(_arg(args, kwargs, 0, "w", Quantity(80e-6, (1, 0, 0, 0))))
    h = _um(_arg(args, kwargs, 1, "h", Quantity(80e-6, (1, 0, 0, 0))))
    return [G.Shape(ctx.metal_layer, G.rect(w, h), "via_metal",
                    mech="anchored")]


def prim_pad(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A bond pad on METAL."""
    w = _um(_arg(args, kwargs, 0, "w", Quantity(100e-6, (1, 0, 0, 0))))
    h = _um(_arg(args, kwargs, 1, "h", Quantity(100e-6, (1, 0, 0, 0))))
    return [G.Shape(ctx.metal_layer, G.rect(w, h), "pad", mech="anchored")]


def prim_route(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A straight metal interconnect track.

    Takes a length and a direction like beam() rather than a polyline,
    because SOIDL has no list literal: a points= form would be callable only
    from Python, and a primitive no .soidl file can invoke is not a language
    feature.  route_path() covers the polyline case for callers that have one.
    """
    L = _um(_arg(args, kwargs, 0, "L", Quantity(100e-6, (1, 0, 0, 0))))
    w = _um(_arg(args, kwargs, 1, "w", Quantity(8e-6, (1, 0, 0, 0))))
    direction = _dir(_arg(args, kwargs, 2, "dir", "x"))
    poly = G.rect(L, w) if direction == "x" else G.rect(w, L)
    return [G.Shape(ctx.metal_layer, poly, "route", mech="anchored")]


def prim_route_path(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A metal track following an arbitrary polyline (Python callers)."""
    pts = _arg(args, kwargs, 0, "points")
    w = _um(_arg(args, kwargs, 1, "w", Quantity(8e-6, (1, 0, 0, 0))))
    if not pts:
        raise ValueError("route_path() needs a points list")
    coords = [(_um(p[0]) if isinstance(p[0], Quantity) else float(p[0]),
               _um(p[1]) if isinstance(p[1], Quantity) else float(p[1]))
              for p in pts]
    return [G.Shape(ctx.metal_layer, G.wire(coords, w), "route",
                    mech="anchored")]


def prim_ring(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """An annular resonator: the body of a ring gyroscope.

    A ring's two degenerate wine-glass (cos 2-theta) modes sit 45 degrees
    apart, and rotation Coriolis-couples energy between them -- that transfer
    is the rate signal.  The degeneracy is why the ring has to be round: a
    square would split the two modes and destroy the coupling.
    """
    R = _um(_arg(args, kwargs, 0, "R", Quantity(200e-6, (1, 0, 0, 0))))
    w = _um(_arg(args, kwargs, 1, "w", Quantity(20e-6, (1, 0, 0, 0))))
    n_seg = int(round(_num(_arg(args, kwargs, 2, "n_seg", 64))))
    return [G.Shape(ctx.device_layer, G.annulus(R, w, n_seg), "ring",
                    mech="released")]


def prim_disk(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A solid disk, perforated from the process release rules like a plate.

    Holes are taken from the same grid a plate uses and then filtered by
    radius, so none reaches past the rim -- a clipped hole would be a notch in
    the outline, not a release hole.

    Note: a perforated disk is a non-rectilinear exterior with rectilinear
    holes, which goes through _bridge_holes rather than the band path, and so
    inherits the non-manifold seam documented in mesh.triangulate_with_holes.
    Nothing in this library uses a perforated disk; a watertight one is the
    case that would force a real fix to hole bridging.
    """
    R = _um(_arg(args, kwargs, 0, "R", Quantity(150e-6, (1, 0, 0, 0))))
    n_seg = int(round(_num(_arg(args, kwargs, 1, "n_seg", 64))))
    holes_arg = _arg(args, kwargs, 2, "holes", "auto")
    poly = G.circle(R, n_seg)
    want_holes = (holes_arg == "auto" or holes_arg is True)
    if want_holes and 2 * R > ctx.max_solid_span:
        keep = [h for h in _hole_grid(2 * R, 2 * R, ctx)
                if all(math.hypot(x, y) < R - ctx.hole_size for (x, y) in h)]
        if keep:
            poly = G.Polygon(poly.exterior, keep)
    return [G.Shape(ctx.device_layer, poly, "disk", mech="released")]


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
    "chevron": prim_chevron,
    "hot_arm": prim_hot_arm,
    "ring": prim_ring,
    "disk": prim_disk,
    "gap_stop": prim_gap_stop,
    "trench": prim_trench,
    "via_metal": prim_via_metal,
    "pad": prim_pad,
    "route": prim_route,
    "route_path": prim_route_path,
}


def is_primitive(name: str) -> bool:
    return name in PRIMITIVES
