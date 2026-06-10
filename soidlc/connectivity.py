"""Electrical connectivity extraction (stage 3 of the soidlc pipeline).

In SOI the structural layer is conductive: electrical nodes are exactly the
connected components of the DEVICE-layer polygons.  This module computes
those components so the elaborator can verify the declared ``net`` /
``isolate`` statements against geometric reality — the MEMS analogue of an
LVS check.  Abutting edges (within ``EPS``) count as electrically connected.
"""

from __future__ import annotations

from typing import List

from . import geometry as G

EPS = 0.01   # um


def _is_axis_rect(ring) -> bool:
    if len(ring) != 4:
        return False
    for i in range(4):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % 4]
        if abs(x0 - x1) > 1e-6 and abs(y0 - y1) > 1e-6:
            return False
    return True


def _point_in(ring, x: float, y: float) -> bool:
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


def _segs(ring):
    n = len(ring)
    for i in range(n):
        yield ring[i], ring[(i + 1) % n]


def _orient(a, b, c) -> int:
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(v) < 1e-12:
        return 0
    return 1 if v > 0 else -1


def _on_seg(a, b, c) -> bool:
    return (_orient(a, b, c) == 0
            and min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS
            and min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS)


def _seg_intersect(p1, p2, p3, p4) -> bool:
    o1, o2 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    o3, o4 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return (_on_seg(p1, p2, p3) or _on_seg(p1, p2, p4)
            or _on_seg(p3, p4, p1) or _on_seg(p3, p4, p2))


def touches(a: G.Polygon, b: G.Polygon, eps: float = EPS) -> bool:
    """True if two polygons overlap or abut (silicon-continuous)."""
    ax0, ay0, ax1, ay1 = a.bbox()
    bx0, by0, bx1, by1 = b.bbox()
    if ax0 > bx1 + eps or bx0 > ax1 + eps:
        return False
    if ay0 > by1 + eps or by0 > ay1 + eps:
        return False
    # for axis-aligned rectangles the bbox test is exact
    if _is_axis_rect(a.exterior) and _is_axis_rect(b.exterior):
        return True
    if any(_point_in(b.exterior, x, y) for x, y in a.exterior):
        return True
    if any(_point_in(a.exterior, x, y) for x, y in b.exterior):
        return True
    for s1 in _segs(a.exterior):
        for s2 in _segs(b.exterior):
            if _seg_intersect(s1[0], s1[1], s2[0], s2[1]):
                return True
    return False


def components(polys: List[G.Polygon], eps: float = EPS) -> List[int]:
    """Union-find over polygons; returns a 1-based island id per polygon."""
    n = len(polys)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if touches(polys[i], polys[j], eps):
                parent[find(i)] = find(j)

    ids: dict = {}
    out: List[int] = []
    for i in range(n):
        r = find(i)
        if r not in ids:
            ids[r] = len(ids) + 1
        out.append(ids[r])
    return out
