"""Export a compiled soidlc device to a generic web-viewer JSON bundle.

Contract: { meta, geometry:{positions,indices,vertexLayer}, results:[...] }.
The bundle is device-agnostic; the frontend renders any result without
soidlc-specific knowledge.
"""
from __future__ import annotations

import math
from typing import List, Optional

from . import build3d
from .parser import parse
from .elaborate import Elaborator


def _compile(path: str):
    """Elaborate a .soidl file and build its 3D mesh; return (elab, result, mesh)."""
    with open(path) as f:
        ast = parse(f.read())
    elab = Elaborator(ast)
    from . import closure
    overrides = closure.run(elab, None, fem_calibrate=False, fem_h=20.0)
    result = elab.elaborate_device(None, overrides=overrides)
    mesh = build3d.build_mesh(result, elab.process, include_handle=True)
    return elab, result, mesh


def _geometry(mesh, process):
    colors = build3d.layer_colors(process)
    layer_names = sorted({g for g in mesh.tri_group})
    layer_idx = {name: i for i, name in enumerate(layer_names)}
    positions: List[float] = []
    for (x, y, z) in mesh.vertices:
        positions.extend((float(x), float(y), float(z)))
    indices: List[int] = []
    vertex_layer = [0] * len(mesh.vertices)
    seen = [False] * len(mesh.vertices)
    for (a, b, c), grp in zip(mesh.triangles, mesh.tri_group):
        indices.extend((a, b, c))
        li = layer_idx[grp]
        for v in (a, b, c):           # first-writer-wins per shared vertex
            if not seen[v]:
                vertex_layer[v] = li
                seen[v] = True
    layers = [{"name": n,
               "color": [float(c) for c in colors.get(n, (0.6, 0.6, 0.6))]}
              for n in layer_names]
    # bounds() returns ((x0,y0,z0),(x1,y1,z1)) — flatten to a 6-list
    lo, hi = mesh.bounds()
    bbox = [float(lo[0]), float(lo[1]), float(lo[2]),
            float(hi[0]), float(hi[1]), float(hi[2])]
    meta = {"units": "um",
            "bbox": bbox,
            "layers": layers}
    return {"positions": positions, "indices": indices,
            "vertexLayer": vertex_layer}, meta


def build_bundle(path: str, n_modes: int = 3,
                 static_cases: Optional[list] = None) -> dict:
    """Compile `path` and return the web-viewer bundle dict."""
    elab, result, mesh = _compile(path)
    geometry, meta = _geometry(mesh, elab.process)
    meta["device"] = getattr(getattr(elab, "device_ast", None), "name",
                             "device")
    f0 = result.model.get("f0")
    if f0 is not None and hasattr(f0, "value"):
        meta["lumped_f0_hz"] = f0.value
    results = _modal_results(elab, mesh, n_modes) if n_modes else []
    if static_cases:
        results += _static_results(elab, mesh, static_cases)
    return {"meta": meta, "geometry": geometry, "results": results}


def _suspended_islands(elab):
    out = []
    for cid, ss in getattr(elab, "islands", []):
        if any(s.mech == "anchored" for s in ss) and \
           any(s.mech == "released" for s in ss):
            out.append((cid, ss))
    return out


def _modal_results(elab, mesh, n_modes):   # implemented in Task 2
    return []


def _static_results(elab, mesh, cases):    # implemented in Task 3
    return []
