"""Mesh an elaborated island's polygons into a triangle FemMesh via gmsh.

Holes are smeared into a per-cell mass fill factor (not cut from the mesh);
fingers (label contains ``lump_label``) are lumped as point mass onto the
nearest meshed node. These are deliberate phase-1 simplifications inherited
from the original fem2d.
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


def build_mesh(shapes: List[G.Shape], h: float,
               lump_label: str = "finger") -> FemMesh:
    solids = [s for s in shapes if lump_label not in s.label]
    lumped = [s for s in shapes if lump_label in s.label]
    mesh = FemMesh()
    if not solids:
        return mesh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("island")
        tags = [(2, _add_surface(s.polygon.normalized(), h)) for s in solids]
        gmsh.model.occ.synchronize()
        # fuse overlapping/adjacent solids into one domain
        if len(tags) > 1:
            gmsh.model.occ.fuse([tags[0]], tags[1:])
            gmsh.model.occ.synchronize()
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
    anchored = [s.polygon for s in shapes if s.mech == "anchored"]
    for n, (px, py) in enumerate(mesh.nodes):
        for poly in anchored:
            if _point_in_ring(px, py, poly.exterior):
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
