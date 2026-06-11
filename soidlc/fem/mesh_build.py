"""Mesh an elaborated island's polygons into a triangle FemMesh via gmsh.

Holes are smeared into a per-cell mass fill factor (not cut from the mesh);
fingers (label contains ``lump_label``) are lumped as point mass onto the
nearest meshed node. These are deliberate phase-1 simplifications inherited
from the original hand-written solver.
"""
from __future__ import annotations

from typing import List

import gmsh

from .. import geometry as G
from .result import FemMesh

MAX_ELEMENTS = 30000


def _point_in_ring(px: float, py: float, ring) -> bool:
    """Ray-cast point-in-polygon for a single ring."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > py) != (yj > py):
            xint = (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            if px < xint:
                inside = not inside
        j = i
    return inside


def _add_surface(poly: G.Polygon, h: float) -> int:
    """Add one OCC plane surface (exterior + holes) at mesh size h."""
    def loop(ring) -> int:
        pts = [gmsh.model.occ.addPoint(x, y, 0.0, h) for x, y in ring]
        lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                 for i in range(len(pts))]
        return gmsh.model.occ.addCurveLoop(lines)

    outer = loop(poly.exterior)
    inners = [loop(hr) for hr in poly.holes]
    return gmsh.model.occ.addPlaneSurface([outer, *inners])


def _shape_h(s: G.Shape, h_global: float) -> float:
    """Per-shape mesh size: cap at half the shape's minimum dimension.

    Narrow flexures (e.g. 5 um beams) would be under-meshed if we used the
    global h (e.g. 12 um), so we reduce h to ensure at least two elements
    fit across the narrowest axis.  A floor of 1.0 um avoids degenerate
    meshes for sub-micron features.
    """
    x0, y0, x1, y1 = s.polygon.bbox()
    w_min = min(x1 - x0, y1 - y0)
    return min(h_global, max(w_min / 2.0, 1.0))


def _add_size_fields(shapes: List[G.Shape], h_global: float) -> None:
    """Add Gmsh Box mesh-size fields for shapes narrower than h_global.

    After a boolean Fuse the per-point mesh sizes set during surface creation
    are discarded.  Box fields re-impose fine meshing inside each narrow shape's
    bounding box so that MEMS flexures (typically 2-10 um wide) are adequately
    resolved at the global mesh size h_global (e.g. 12 um).
    """
    field_ids = []
    for s in shapes:
        h_s = _shape_h(s, h_global)
        if h_s < h_global * 0.9:          # only bother if noticeably finer
            x0, y0, x1, y1 = s.polygon.bbox()
            fid = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(fid, "VIn",  h_s)
            gmsh.model.mesh.field.setNumber(fid, "VOut", h_global)
            gmsh.model.mesh.field.setNumber(fid, "XMin", x0)
            gmsh.model.mesh.field.setNumber(fid, "XMax", x1)
            gmsh.model.mesh.field.setNumber(fid, "YMin", y0)
            gmsh.model.mesh.field.setNumber(fid, "YMax", y1)
            gmsh.model.mesh.field.setNumber(fid, "ZMin", -1.0)
            gmsh.model.mesh.field.setNumber(fid, "ZMax",  1.0)
            field_ids.append(fid)
    if field_ids:
        min_fid = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_fid, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_fid)


def build_mesh(shapes: List[G.Shape], h: float,
               lump_label: str = "finger") -> FemMesh:
    solids = [s for s in shapes if lump_label not in s.label]
    lumped = [s for s in shapes if lump_label in s.label]
    mesh = FemMesh()
    if not solids:
        return mesh

    # interruptible=False: skip gmsh's SIGINT handler so meshing works in
    # non-main threads (e.g. the web backend's request workers)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("island")
        tags = [(2, _add_surface(s.polygon.normalized(), h)) for s in solids]
        gmsh.model.occ.synchronize()
        # fuse overlapping/adjacent solids into one domain
        if len(tags) > 1:
            gmsh.model.occ.fuse([tags[0]], tags[1:])
            gmsh.model.occ.synchronize()
        # Re-apply fine mesh sizes for narrow shapes (fuse discards per-point h)
        _add_size_fields(solids, h)
        gmsh.model.mesh.generate(2)

        # nodes
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        idx_of = {int(t): i for i, t in enumerate(node_tags)}
        mesh.nodes = [(coords[3 * i], coords[3 * i + 1])
                      for i in range(len(node_tags))]
        # triangles (type 2)
        etypes, etags, enodes = gmsh.model.mesh.getElements(2)
        for et, conn in zip(etypes, enodes):
            if et != 2:               # only linear triangles
                continue
            for k in range(0, len(conn), 3):
                mesh.cells.append((idx_of[int(conn[k])],
                                   idx_of[int(conn[k + 1])],
                                   idx_of[int(conn[k + 2])]))
    finally:
        gmsh.finalize()

    _mark_fixed(mesh, shapes)
    _assign_fill(mesh, shapes)
    _lump_fingers(mesh, lumped)
    return mesh


def _mark_fixed(mesh: FemMesh, shapes: List[G.Shape]) -> None:
    """Mark all mesh nodes inside or on the boundary of any anchored shape.

    Uses bounding-box containment (with a small tolerance) for the common
    case of axis-aligned rectangular anchors; falls back to the ray-cast
    test for non-rectangular shapes.  Boundary nodes (exactly on an edge)
    are correctly fixed by the bbox test.
    """
    anchored = [s.polygon for s in shapes if s.mech == "anchored"]
    anchor_bboxes = [poly.bbox() for poly in anchored]
    tol = 1e-6
    for n, (px, py) in enumerate(mesh.nodes):
        for poly, (x0, y0, x1, y1) in zip(anchored, anchor_bboxes):
            if not (x0 - tol <= px <= x1 + tol
                    and y0 - tol <= py <= y1 + tol):
                continue
            # bbox is exact for rectangular anchors (and catches boundary
            # nodes the ray-cast misses); ray-cast handles the rare
            # non-rectangular anchor.
            if len(poly.exterior) == 4 or _point_in_ring(px, py, poly.exterior):
                mesh.fixed.add(n)
                break


def _assign_fill(mesh: FemMesh, shapes: List[G.Shape]) -> None:
    """Per-cell mass fill: 1.0 minus hole-area coverage at the centroid."""
    holes = [hr for s in shapes for hr in s.polygon.holes]
    for (a, b, c) in mesh.cells:
        cx = (mesh.nodes[a][0] + mesh.nodes[b][0] + mesh.nodes[c][0]) / 3.0
        cy = (mesh.nodes[a][1] + mesh.nodes[b][1] + mesh.nodes[c][1]) / 3.0
        in_hole = any(_point_in_ring(cx, cy, hr) for hr in holes)
        mesh.fill.append(0.0 if in_hole else 1.0)


def _lump_fingers(mesh: FemMesh, lumped: List[G.Shape]) -> None:
    if not mesh.nodes:
        return
    for s in lumped:
        x0, y0, x1, y1 = s.polygon.bbox()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        best = min(range(len(mesh.nodes)),
                   key=lambda n: (mesh.nodes[n][0] - cx) ** 2
                   + (mesh.nodes[n][1] - cy) ** 2)
        mesh.extra_mass.append((best, s.polygon.area()))
