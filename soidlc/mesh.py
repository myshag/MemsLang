"""Triangulation and 2.5D extrusion: layered polygons -> a 3D triangle mesh.

The polygon top/bottom caps are triangulated with ear clipping; polygons with
holes are first reduced to a single simple ring by bridging each hole to the
outer boundary.  Side walls are emitted as quads (two triangles each).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from . import geometry as G

Pt3 = Tuple[float, float, float]


@dataclass
class Mesh:
    vertices: List[Pt3] = field(default_factory=list)
    triangles: List[Tuple[int, int, int]] = field(default_factory=list)
    # parallel to triangles: name of the group/layer each tri belongs to
    tri_group: List[str] = field(default_factory=list)

    def add_vertex(self, p: Pt3) -> int:
        self.vertices.append(p)
        return len(self.vertices) - 1

    def add_triangle(self, a: int, b: int, c: int, group: str = "default"):
        self.triangles.append((a, b, c))
        self.tri_group.append(group)

    def extend(self, other: "Mesh") -> None:
        base = len(self.vertices)
        self.vertices.extend(other.vertices)
        for (a, b, c), g in zip(other.triangles, other.tri_group):
            self.triangles.append((a + base, b + base, c + base))
            self.tri_group.append(g)

    def bounds(self):
        if not self.vertices:
            return (0, 0, 0), (0, 0, 0)
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# ---------------------------------------------------------------------------
# Ear-clipping triangulation
# ---------------------------------------------------------------------------
def _area(ring: Sequence[G.Pt]) -> float:
    return G.signed_area(list(ring))


def _is_convex(a, b, c) -> bool:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0


def _point_in_tri(p, a, b, c) -> bool:
    d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
    d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
    d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def triangulate_simple(ring: List[G.Pt]) -> List[Tuple[int, int, int]]:
    """Ear-clip a simple polygon (CCW).  Returns index triples into ``ring``."""
    n = len(ring)
    if n < 3:
        return []
    idx = list(range(n))
    if _area(ring) < 0:
        idx.reverse()
    tris: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 2 and guard < 10000:
        guard += 1
        ear_found = False
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = ring[i0], ring[i1], ring[i2]
            if not _is_convex(a, b, c):
                continue
            # no other vertex inside this ear
            bad = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _point_in_tri(ring[j], a, b, c):
                    bad = True
                    break
            if bad:
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            ear_found = True
            break
        if not ear_found:
            # fallback: fan triangulation (degenerate cases)
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
            break
    return tris


def _is_rectilinear(ring: List[G.Pt]) -> bool:
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        if abs(x0 - x1) > 1e-7 and abs(y0 - y1) > 1e-7:
            return False
    return True


def _y_spans(ring: List[G.Pt]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), max(xs), min(ys), max(ys)


def _decompose_rectilinear(poly: G.Polygon) -> List[Tuple[G.Pt, G.Pt, G.Pt]]:
    """Cap triangles for an axis-aligned rectangle with axis-aligned holes,
    via a vertical-slab sweep.  Produces clean quads (no slivers)."""
    ex0, ex1, ey0, ey1 = _y_spans(poly.exterior)
    holes = [_y_spans(h) for h in poly.holes]
    xs = sorted({ex0, ex1} | {h[0] for h in holes} | {h[1] for h in holes})
    tris: List[Tuple[G.Pt, G.Pt, G.Pt]] = []
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        if xb - xa < 1e-9:
            continue
        xm = (xa + xb) / 2
        # holes covering this slab -> subtract their y-intervals
        blocked = [(h[2], h[3]) for h in holes if h[0] < xm < h[1]]
        blocked.sort()
        y = ey0
        for (hy0, hy1) in blocked:
            if hy0 > y:
                tris.extend(_quad(xa, xb, y, hy0))
            y = max(y, hy1)
        if ey1 > y:
            tris.extend(_quad(xa, xb, y, ey1))
    return tris


def _quad(xa, xb, ya, yb):
    p00, p10 = (xa, ya), (xb, ya)
    p11, p01 = (xb, yb), (xa, yb)
    return [(p00, p10, p11), (p00, p11, p01)]


def triangulate_with_holes(poly: G.Polygon) -> List[Tuple[G.Pt, G.Pt, G.Pt]]:
    """Return cap triangles (as coordinate triples) for a polygon with holes."""
    rectilinear = (_is_rectilinear(poly.exterior)
                   and all(_is_rectilinear(h) for h in poly.holes))
    if rectilinear:
        return _decompose_rectilinear(poly)
    ring = _bridge_holes(poly)
    idx = triangulate_simple(ring)
    return [(ring[a], ring[b], ring[c]) for (a, b, c) in idx]


def _bridge_holes(poly: G.Polygon) -> List[G.Pt]:
    """Reduce a polygon-with-holes to a single simple ring by cutting a bridge
    from each hole's rightmost vertex to the outer ring."""
    ext = G.ensure_ccw(list(poly.exterior))
    if not poly.holes:
        return ext
    holes = [G.ensure_cw(list(h)) for h in poly.holes]
    # process holes from rightmost to leftmost for stable bridging
    holes.sort(key=lambda h: max(p[0] for p in h), reverse=True)
    ring = ext
    for hole in holes:
        ring = _bridge_one(ring, hole)
    return ring


def _bridge_one(outer: List[G.Pt], hole: List[G.Pt]) -> List[G.Pt]:
    # rightmost vertex of the hole
    hi = max(range(len(hole)), key=lambda i: hole[i][0])
    hx, hy = hole[hi]
    # nearest outer vertex to the right of the hole's rightmost point
    best = None
    best_d = None
    for i, (ox, oy) in enumerate(outer):
        if ox < hx:
            continue
        d = (ox - hx) ** 2 + (oy - hy) ** 2
        if best_d is None or d < best_d:
            best_d, best = d, i
    if best is None:
        best = min(range(len(outer)),
                   key=lambda i: (outer[i][0] - hx) ** 2 + (outer[i][1] - hy) ** 2)
    # build new ring: outer[0..best], bridge into hole (starting at hi, going
    # around), back to hi, then outer[best..end]
    hole_seq = hole[hi:] + hole[:hi] + [hole[hi]]
    new_ring = (outer[:best + 1] + hole_seq + [outer[best]] + outer[best + 1:])
    return new_ring


# ---------------------------------------------------------------------------
# Extrusion
# ---------------------------------------------------------------------------
def _point_in_ring(ring: List[G.Pt], x: float, y: float) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xc:
                inside = not inside
    return inside


def extrude_polygon(poly: G.Polygon, z0: float, z1: float,
                    group: str = "default") -> Mesh:
    """Extrude a polygon (with holes) between z0 and z1 into a closed prism.

    For axis-aligned rectilinear polygons (all SOIDL primitives) a shared-grid
    voxel-surface method is used, yielding a watertight, T-junction-free mesh.
    Other shapes fall back to ear-clip caps + per-ring walls.
    """
    poly = poly.normalized()
    if len(poly.exterior) < 3:
        return Mesh()
    rectilinear = (_is_rectilinear(poly.exterior)
                   and all(_is_rectilinear(h) for h in poly.holes))
    if rectilinear:
        return _extrude_grid(poly, z0, z1, group)
    return _extrude_general(poly, z0, z1, group)


def _extrude_grid(poly: G.Polygon, z0: float, z1: float, group: str) -> Mesh:
    mesh = Mesh()
    xs = sorted({p[0] for p in poly.exterior}
                | {p[0] for h in poly.holes for p in h})
    ys = sorted({p[1] for p in poly.exterior}
                | {p[1] for h in poly.holes for p in h})
    nx, ny = len(xs) - 1, len(ys) - 1
    if nx < 1 or ny < 1:
        return mesh

    def solid(i, j) -> bool:
        if i < 0 or j < 0 or i >= nx or j >= ny:
            return False
        cx = (xs[i] + xs[i + 1]) / 2
        cy = (ys[j] + ys[j + 1]) / 2
        if not _point_in_ring(poly.exterior, cx, cy):
            return False
        return not any(_point_in_ring(h, cx, cy) for h in poly.holes)

    cache: dict = {}

    def vert(x, y, z) -> int:
        key = (round(x, 4), round(y, 4), round(z, 4))
        k = cache.get(key)
        if k is None:
            k = mesh.add_vertex((x, y, z))
            cache[key] = k
        return k

    for i in range(nx):
        for j in range(ny):
            if not solid(i, j):
                continue
            xa, xb, ya, yb = xs[i], xs[i + 1], ys[j], ys[j + 1]
            # caps
            t = [vert(xa, ya, z1), vert(xb, ya, z1),
                 vert(xb, yb, z1), vert(xa, yb, z1)]
            mesh.add_triangle(t[0], t[1], t[2], group)
            mesh.add_triangle(t[0], t[2], t[3], group)
            b = [vert(xa, ya, z0), vert(xb, ya, z0),
                 vert(xb, yb, z0), vert(xa, yb, z0)]
            mesh.add_triangle(b[0], b[2], b[1], group)
            mesh.add_triangle(b[0], b[3], b[2], group)
            # walls only where the neighbour is empty
            if not solid(i - 1, j):     # left (x = xa)
                _wall(mesh, vert, xa, ya, xa, yb, z0, z1, group)
            if not solid(i + 1, j):     # right (x = xb)
                _wall(mesh, vert, xb, yb, xb, ya, z0, z1, group)
            if not solid(i, j - 1):     # bottom (y = ya)
                _wall(mesh, vert, xb, ya, xa, ya, z0, z1, group)
            if not solid(i, j + 1):     # top (y = yb)
                _wall(mesh, vert, xa, yb, xb, yb, z0, z1, group)
    return mesh


def _wall(mesh, vert, x0, y0, x1, y1, z0, z1, group):
    a = vert(x0, y0, z0); b = vert(x1, y1, z0)
    c = vert(x1, y1, z1); d = vert(x0, y0, z1)
    mesh.add_triangle(a, b, c, group)
    mesh.add_triangle(a, c, d, group)


def _extrude_general(poly: G.Polygon, z0: float, z1: float, group: str) -> Mesh:
    mesh = Mesh()
    cache: dict = {}

    def vert(x, y, z) -> int:
        key = (round(x, 4), round(y, 4), round(z, 4))
        k = cache.get(key)
        if k is None:
            k = mesh.add_vertex((x, y, z))
            cache[key] = k
        return k

    for (a, b, c) in triangulate_with_holes(poly):
        mesh.add_triangle(vert(*a, z1), vert(*b, z1), vert(*c, z1), group)
        mesh.add_triangle(vert(*a, z0), vert(*c, z0), vert(*b, z0), group)
    for ring in [poly.exterior] + list(poly.holes):
        n = len(ring)
        for i in range(n):
            x0, y0 = ring[i]
            x1, y1 = ring[(i + 1) % n]
            _wall(mesh, vert, x0, y0, x1, y1, z0, z1, group)
    return mesh
