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


def _nearest_disp_batch(verts_xy, femmesh, vec, dof_of, max_dist):
    """Return lists of (ux, uy) for each vertex in verts_xy using cKDTree.

    Vertices farther than max_dist from any FEM node get (0.0, 0.0).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    nodes = np.array(femmesh.nodes, dtype=float)  # (N, 2)
    qpts = np.array(verts_xy, dtype=float)         # (V, 2)

    tree = cKDTree(nodes)
    dists, idxs = tree.query(qpts, k=1)

    results = []
    for dist, ni in zip(dists, idxs):
        if dist > max_dist:
            results.append((0.0, 0.0))
        else:
            base = dof_of.get(int(ni), -1)
            if base >= 0:
                results.append((vec[base], vec[base + 1]))
            else:
                results.append((0.0, 0.0))
    return results


def _modal_results(elab, mesh, n_modes):
    dev = elab.process.device()
    if dev is None:
        return []
    E, nu, rho, t = dev.E, dev.nu, dev.rho, dev.thickness * 1e-6
    from . import fem
    verts = mesh.vertices
    islands = _suspended_islands(elab)
    out = []
    for cid, ss in islands:
        fm = fem.build_mesh(ss, 12.0)
        if not fm.cells or fm.n_cells > fem.MAX_ELEMENTS:
            continue
        freqs, vecs, dof_of = fem.modal(fm, E, nu, rho, t, n_modes=n_modes)
        xs = [p[0] for p in fm.nodes]
        ys = [p[1] for p in fm.nodes]
        max_dist = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0) + 12.0

        # Precompute xy of all 3D vertices for batched KD-tree queries
        verts_xy = [(vx, vy) for (vx, vy, vz) in verts]

        for mi, (f, vec) in enumerate(zip(freqs, vecs), 1):
            ux_uy_list = _nearest_disp_batch(verts_xy, fm, vec, dof_of, max_dist)
            disp = []
            mags = []
            for (ux, uy) in ux_uy_list:
                disp.extend((ux, uy, 0.0))
                mags.append(math.hypot(ux, uy))
            mmax = max(mags) or 1.0
            out.append({
                "type": "modal",
                "label": (f"Island {cid} — Mode {mi}" if len(islands) > 1
                          else f"Mode {mi}"),
                "freq_hz": float(f),
                "animate": True,
                "disp": [d / mmax for d in disp],
                "dmax_um": float(mmax),
                "fields": {"disp_mag": [m / mmax for m in mags]},
            })
    return out


def _static_results(elab, mesh, cases):
    dev = elab.process.device()
    if dev is None:
        return []
    E, nu, t = dev.E, dev.nu, dev.thickness * 1e-6
    from . import fem
    islands = _suspended_islands(elab)
    if not islands:
        return []
    _cid, ss = islands[0]
    fm = fem.build_mesh(ss, 12.0)
    if not fm.cells:
        return []
    xs = [p[0] for p in fm.nodes]
    ys = [p[1] for p in fm.nodes]
    max_dist = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0) + 12.0
    out = []
    verts_xy = [(vx, vy) for (vx, vy, vz) in mesh.vertices]
    for case in cases:
        disp_map = fem.static_solve(fm, E, nu, t, case["forces"])
        # Build a flat vec[] and dof_of dict compatible with _nearest_disp_batch
        vec = []
        dof_of = {}
        base = 0
        for n in range(len(fm.nodes)):
            uxuy = disp_map.get(n)
            if uxuy is None:
                dof_of[n] = -1
            else:
                dof_of[n] = base
                vec.extend((uxuy[0], uxuy[1]))
                base += 2
        ux_uy_list = _nearest_disp_batch(verts_xy, fm, vec, dof_of, max_dist)
        disp = []
        mags = []
        for (ux, uy) in ux_uy_list:
            disp.extend((ux, uy, 0.0))
            mags.append(math.hypot(ux, uy))
        mmax = max(mags) or 1.0
        out.append({"type": "static", "label": case["label"],
                    "freq_hz": None, "animate": False,
                    "disp": [d / mmax for d in disp],
                    "dmax_um": float(mmax),
                    "fields": {"disp_mag": [m / mmax for m in mags]}})
    return out
