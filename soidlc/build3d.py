"""Assemble a 3D mesh from elaborated shapes and the process stack.

Mechanical status drives the vertical build:
  * every DEVICE/METAL shape is extruded at its layer's z range;
  * an *anchored* DEVICE shape also gets a BOX pillar beneath it (oxide that
    survived the release), tying it to the handle;
  * *released* shapes float above the BOX gap;
  * a HANDLE slab spans the whole footprint underneath.
"""

from __future__ import annotations

from typing import Dict, List

from . import geometry as G
from . import mesh as M
from .elaborate import InstanceResult, ProcessInfo


def build_mesh(result: InstanceResult, proc: ProcessInfo,
               include_handle: bool = True) -> M.Mesh:
    out = M.Mesh()
    box = proc.box()
    handle = proc.handle()

    trenches = [sh.polygon for sh in result.shapes if sh.layer == "TRENCH"]

    # handle slab spanning the footprint (a little margin), minus the backside
    # etch openings.  Both are rectangles, so the holed slab stays rectilinear
    # and keeps the watertight voxel mesher.
    if include_handle and handle is not None and result.shapes:
        x0, y0, x1, y1 = result.bbox
        pad = 20.0
        slab = G.rect_corner(x0 - pad, y0 - pad,
                             (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad)
        if trenches:
            slab = G.Polygon(slab.exterior,
                             [list(t.exterior) for t in trenches])
        out.extend(M.extrude_polygon(slab, handle.z0, handle.z1, "HANDLE"))

    for sh in result.shapes:
        lay = proc.layers.get(sh.layer)
        if lay is None:
            continue                     # TRENCH and any other pseudo-layer
        out.extend(M.extrude_polygon(sh.polygon, lay.z0, lay.z1, sh.layer))
        # anchored device silicon keeps the oxide beneath it -- unless the
        # backside etch took the substrate out from under it, in which case
        # there is nothing for the pillar to stand on
        if (sh.layer == "DEVICE" and sh.mech == "anchored" and box is not None
                and not _inside_any(sh.polygon, trenches)):
            solid = G.Polygon(sh.polygon.exterior)   # drop holes for the pillar
            out.extend(M.extrude_polygon(solid, box.z0, box.z1, "BOX"))

    return out


def _inside_any(poly: G.Polygon, trenches) -> bool:
    """True if `poly` lies wholly within some trench footprint.

    Wholly contained, not merely overlapping: a shape straddling a trench edge
    still has substrate under part of it and keeps its pillar.  That is not a
    corner cut -- the rim of a membrane is exactly the straddling case, and it
    has to stay anchored or the diaphragm falls into the cavity.
    """
    px0, py0, px1, py1 = poly.bbox()
    for t in trenches:
        tx0, ty0, tx1, ty1 = t.bbox()
        if px0 >= tx0 and py0 >= ty0 and px1 <= tx1 and py1 <= ty1:
            return True
    return False


def layer_colors(proc: ProcessInfo) -> Dict[str, tuple]:
    return {name: lay.color for name, lay in proc.layers.items()}
