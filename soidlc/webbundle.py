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

# Layer names whose vertices must always carry zero displacement (static parts).
_STATIC_LAYERS = frozenset(("BOX", "HANDLE"))


def _compile_source(src: str):
    """Elaborate SOIDL source text and build its 3D mesh; return (elab, result, mesh)."""
    ast = parse(src)
    elab = Elaborator(ast)
    from . import closure
    overrides = closure.run(elab, None, fem_calibrate=False, fem_h=20.0)
    result = elab.elaborate_device(None, overrides=overrides)
    mesh = build3d.build_mesh(result, elab.process, include_handle=True)
    return elab, result, mesh


def _compile(path: str):
    """Elaborate a .soidl file and build its 3D mesh; return (elab, result, mesh)."""
    with open(path) as f:
        return _compile_source(f.read())


def build_bundle_from_source(src: str, n_modes: int = 6,
                             static_cases: Optional[list] = None) -> dict:
    """Compile SOIDL source text and return the web-viewer bundle dict.

    Raises on parse/elaboration failure; if the elaborator collected
    errors (elab.errors non-empty), raises ValueError with them joined.
    """
    elab, result, mesh = _compile_source(src)
    if getattr(elab, "errors", None):
        raise ValueError("; ".join(str(e) for e in elab.errors))
    return build_bundle_from_compiled(elab, result, mesh,
                                      n_modes=n_modes,
                                      static_cases=static_cases)


def _geometry(mesh, process):
    colors = build3d.layer_colors(process)

    # Fix 4: order layer_names by process stack order, not alphabetically
    present = set(mesh.tri_group)
    layer_names = [n for n in process.layers if n in present]
    # append any unexpected names that aren't in process.layers
    layer_names += [n for n in present if n not in process.layers]

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
    # Fix 4: cap colors to 3 components
    layers = [{"name": n,
               "color": [float(c) for c in colors.get(n, (0.6, 0.6, 0.6))][:3]}
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


def build_bundle(path: str, n_modes: int = 6,
                 static_cases: Optional[list] = None) -> dict:
    """Compile `path` and return the web-viewer bundle dict."""
    elab, result, mesh = _compile(path)
    return build_bundle_from_compiled(elab, result, mesh,
                                      n_modes=n_modes,
                                      static_cases=static_cases)


def build_bundle_from_compiled(elab, result, mesh, n_modes: int = 6,
                                static_cases: Optional[list] = None) -> dict:
    """Build the web-viewer bundle from already-compiled artifacts.

    This avoids re-compiling when the caller already has elab/result/mesh
    (e.g. the CLI after ``compile_file``).
    """
    geometry, meta = _geometry(mesh, elab.process)
    meta["device"] = getattr(getattr(elab, "device_ast", None), "name",
                             "device")
    f0 = result.model.get("f0")
    if f0 is not None and hasattr(f0, "value"):
        meta["lumped_f0_hz"] = f0.value

    # Build per-vertex static mask: True => BOX or HANDLE layer (zero disp).
    layer_names = [l["name"] for l in meta["layers"]]
    vertex_layer = geometry["vertexLayer"]
    nverts = len(vertex_layer)
    static_mask = [layer_names[vertex_layer[v]] in _STATIC_LAYERS
                   for v in range(nverts)]

    results = (_modal_results(elab, mesh, n_modes, static_mask)
               if n_modes else [])
    if static_cases:
        results += _static_results(elab, mesh, static_cases, static_mask)
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


def _nearest_disp_batch_3d(verts_xyz, nodes_shifted, vec, dof_of, max_dist):
    """Return lists of (ux, uy, uz) for each 3D render vertex using cKDTree.

    ``nodes_shifted`` is the Solid3DMesh node array shifted to process
    coordinates (z += z0_device), so distances to render vertices are valid.
    Vertices farther than max_dist from any FEM node get (0.0, 0.0, 0.0).
    Fixed nodes (dof_of[n] == -1) also return (0.0, 0.0, 0.0).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    qpts = np.array(verts_xyz, dtype=float)   # (V, 3)
    tree = cKDTree(nodes_shifted)              # (N, 3)
    dists, idxs = tree.query(qpts, k=1)

    results = []
    for dist, ni in zip(dists, idxs):
        if dist > max_dist:
            results.append((0.0, 0.0, 0.0))
        else:
            base = dof_of.get(int(ni), -1)
            if base >= 0:
                results.append((float(vec[base]),
                                float(vec[base + 1]),
                                float(vec[base + 2])))
            else:
                results.append((0.0, 0.0, 0.0))
    return results


def _modal_results(elab, mesh, n_modes, static_mask):
    """Compute modal displacement results using the 3D solid solver.

    ``static_mask[v]`` is True for vertices whose layer (BOX or HANDLE) must
    always carry zero displacement; these are zeroed before normalisation so
    the normalisation max is taken over unmasked (moving) vertices only.

    Note: _static_results intentionally remains on the 2D plane-stress path.
    """
    dev = elab.process.device()
    if dev is None:
        return []
    from .fem import solid3d
    verts = mesh.vertices
    islands = _suspended_islands(elab)
    # z0 of the DEVICE layer in process coordinates (um) — shifts local mesh z
    z0_device = dev.z0
    out = []
    for cid, ss in islands:
        try:
            m3 = solid3d.mesh_island_3d(ss, thickness_um=dev.thickness, h=10.0)
        except ValueError:
            # Too many tets or other meshing error — skip this island silently.
            continue
        if m3.n_tets == 0:
            continue
        if m3.n_tets > solid3d.MAX_TETS:
            continue
        try:
            freqs, vecs, dof_of = solid3d.modal3d(
                m3, dev.E, dev.nu, dev.rho,
                dev.thickness * 1e-6, n_modes=n_modes)
        except Exception:
            continue
        if not freqs:
            continue

        # Shift 3D mesh nodes from local z (0..thickness) to process z coords
        import numpy as np
        nodes_shifted = m3.nodes.copy()
        nodes_shifted[:, 2] += z0_device

        # max_dist guard based on island XY bbox (same logic as 2D path)
        xs = m3.nodes[:, 0]
        ys = m3.nodes[:, 1]
        max_dist = 0.05 * max(float(xs.max() - xs.min()),
                              float(ys.max() - ys.min()), 1.0) + 12.0

        # Query all render vertices in xyz
        verts_xyz = [(float(vx), float(vy), float(vz)) for (vx, vy, vz) in verts]

        for mi, (f, vec) in enumerate(zip(freqs, vecs), 1):
            raw = _nearest_disp_batch_3d(verts_xyz, nodes_shifted,
                                         vec, dof_of, max_dist)
            disp = []
            mags = []
            for v, (ux, uy, uz) in enumerate(raw):
                if static_mask[v]:
                    disp.extend((0.0, 0.0, 0.0))
                    mags.append(0.0)
                else:
                    disp.extend((ux, uy, uz))
                    mags.append(math.sqrt(ux*ux + uy*uy + uz*uz))
            # Normalise over unmasked vertices only
            mmax = max(
                (m for v, m in enumerate(mags) if not static_mask[v]),
                default=0.0
            ) or 1.0

            # Determine in-plane vs out-of-plane character
            unmasked_disps = [(disp[3*v], disp[3*v+1], disp[3*v+2])
                              for v in range(len(static_mask)) if not static_mask[v]]
            if unmasked_disps:
                mean_dz = sum(abs(d[2]) for d in unmasked_disps) / len(unmasked_disps)
                mean_dxy = sum(math.hypot(d[0], d[1]) for d in unmasked_disps) / len(unmasked_disps)
                oop_suffix = " (out-of-plane)" if mean_dz > mean_dxy else ""
            else:
                oop_suffix = ""

            label_base = (f"Island {cid} — Mode {mi}"
                          if len(islands) > 1 else f"Mode {mi}")
            out.append({
                "type": "modal",
                "label": label_base + oop_suffix,
                "freq_hz": float(f),
                "animate": True,
                "disp": [d / mmax for d in disp],
                "dmax_um": float(mmax),
                "fields": {"disp_mag": [m / mmax for m in mags]},
            })
    return out


def _static_results(elab, mesh, cases, static_mask):
    """Compute static-load displacement results (intentionally kept on 2D plane-stress path).

    Phase 1: static cases apply to the FIRST suspended island only (case node
    indices refer to that island's FEM mesh).  If there are multiple islands
    the island id is included in the result label.
    """
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
    multi_island = len(islands) > 1
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
        raw_ux_uy = _nearest_disp_batch(verts_xy, fm, vec, dof_of, max_dist)
        disp = []
        mags = []
        for v, (ux, uy) in enumerate(raw_ux_uy):
            if static_mask[v]:
                disp.extend((0.0, 0.0, 0.0))
                mags.append(0.0)
            else:
                disp.extend((ux, uy, 0.0))
                mags.append(math.hypot(ux, uy))
        # Normalise over unmasked vertices only
        mmax = max(
            (m for v, m in enumerate(mags) if not static_mask[v]),
            default=0.0
        ) or 1.0
        label = case["label"]
        if multi_island:
            label = f"{label} (island {_cid})"
        out.append({"type": "static", "label": label,
                    "freq_hz": None, "animate": False,
                    "disp": [d / mmax for d in disp],
                    "dmax_um": float(mmax),
                    "fields": {"disp_mag": [m / mmax for m in mags]}})
    return out
