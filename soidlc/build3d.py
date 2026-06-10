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

    # handle slab spanning the footprint (a little margin)
    if include_handle and handle is not None and result.shapes:
        x0, y0, x1, y1 = result.bbox
        pad = 20.0
        slab = G.rect_corner(x0 - pad, y0 - pad,
                             (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad)
        out.extend(M.extrude_polygon(slab, handle.z0, handle.z1, "HANDLE"))

    for sh in result.shapes:
        lay = proc.layers.get(sh.layer)
        if lay is None:
            continue
        out.extend(M.extrude_polygon(sh.polygon, lay.z0, lay.z1, sh.layer))
        # anchored device silicon keeps the oxide beneath it
        if sh.layer == "DEVICE" and sh.mech == "anchored" and box is not None:
            solid = G.Polygon(sh.polygon.exterior)   # drop holes for the pillar
            out.extend(M.extrude_polygon(solid, box.z0, box.z1, "BOX"))

    return out


def layer_colors(proc: ProcessInfo) -> Dict[str, tuple]:
    return {name: lay.color for name, lay in proc.layers.items()}
