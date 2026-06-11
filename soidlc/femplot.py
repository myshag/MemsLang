"""Visualisation of FEM results: mode-shape panels and deformed 3D renders.

Pure standard library, like the rest of the toolchain: triangles are
rasterised with per-vertex colour interpolation into a raw RGB buffer and
written through the PNG encoder in :mod:`soidlc.render`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import geometry as G
from . import mesh as M3
from .fem import FemMesh as Mesh2D
from .render import render_mesh, write_png

BG = (24, 24, 30)
EDGE = (70, 70, 82)
TEXT = (210, 210, 215)


# ---------------------------------------------------------------------------
# colour map (blue -> cyan -> green -> yellow -> red)
# ---------------------------------------------------------------------------
_STOPS = [
    (0.10, 0.15, 0.45),
    (0.05, 0.55, 0.85),
    (0.25, 0.80, 0.30),
    (0.95, 0.85, 0.20),
    (0.90, 0.15, 0.10),
]


def _cmap(t: float) -> Tuple[int, int, int]:
    t = min(1.0, max(0.0, t)) * (len(_STOPS) - 1)
    i = min(int(t), len(_STOPS) - 2)
    f = t - i
    a, b = _STOPS[i], _STOPS[i + 1]
    return tuple(int(255 * (a[k] + (b[k] - a[k]) * f)) for k in range(3))


# ---------------------------------------------------------------------------
# tiny 5x7 bitmap font (just enough for mode labels)
# ---------------------------------------------------------------------------
_FONT = {
    "M": (0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "D": (0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110),
    "E": (0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111),
    "K": (0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "Z": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111),
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    ".": (0, 0, 0, 0, 0, 0b01100, 0b01100),
    " ": (0, 0, 0, 0, 0, 0, 0),
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "F": (0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10011, 0b10001, 0b10001, 0b01110),
    "I": (0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "N": (0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001),
    "P": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "Q": (0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "U": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001),
    "X": (0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
    "-": (0, 0, 0, 0b01110, 0, 0, 0),
    "/": (0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000),
}


def _text(img, W, H, x0, y0, s, color, scale=2):
    for ch in s.upper():
        glyph = _FONT.get(ch)
        if glyph is None:
            x0 += 6 * scale
            continue
        for r, bits in enumerate(glyph):
            for c in range(5):
                if bits & (1 << (4 - c)):
                    for dy in range(scale):
                        for dx in range(scale):
                            px = x0 + c * scale + dx
                            py = y0 + r * scale + dy
                            if 0 <= px < W and 0 <= py < H:
                                o = (py * W + px) * 3
                                img[o:o + 3] = bytes(color)
        x0 += 6 * scale


# ---------------------------------------------------------------------------
# 2D rasterisation helpers
# ---------------------------------------------------------------------------
def _fill_tri(img, W, H, p0, p1, p2, c0, c1, c2):
    minx = max(0, int(min(p0[0], p1[0], p2[0])))
    maxx = min(W - 1, int(max(p0[0], p1[0], p2[0])) + 1)
    miny = max(0, int(min(p0[1], p1[1], p2[1])))
    maxy = min(H - 1, int(max(p0[1], p1[1], p2[1])) + 1)
    area = ((p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    if abs(area) < 1e-12:
        return
    inv = 1.0 / area
    for y in range(miny, maxy + 1):
        py = y + 0.5
        for x in range(minx, maxx + 1):
            px = x + 0.5
            w0 = ((p1[0] - px) * (p2[1] - py)
                  - (p1[1] - py) * (p2[0] - px)) * inv
            w1 = ((p2[0] - px) * (p0[1] - py)
                  - (p2[1] - py) * (p0[0] - px)) * inv
            w2 = 1.0 - w0 - w1
            if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                continue
            r = int(w0 * c0[0] + w1 * c1[0] + w2 * c2[0])
            g = int(w0 * c0[1] + w1 * c1[1] + w2 * c2[1])
            b = int(w0 * c0[2] + w1 * c1[2] + w2 * c2[2])
            o = (y * W + x) * 3
            img[o] = max(0, min(255, r))
            img[o + 1] = max(0, min(255, g))
            img[o + 2] = max(0, min(255, b))


def _line(img, W, H, x0, y0, x1, y1, color):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for i in range(n + 1):
        t = i / n
        x = int(x0 + (x1 - x0) * t)
        y = int(y0 + (y1 - y0) * t)
        if 0 <= x < W and 0 <= y < H:
            o = (y * W + x) * 3
            img[o:o + 3] = bytes(color)


# ---------------------------------------------------------------------------
# mode-shape panels
# ---------------------------------------------------------------------------
def _node_disp(mesh: Mesh2D, vec: List[float], dof_of: Dict[int, int]
               ) -> List[Tuple[float, float]]:
    out = []
    for n in range(len(mesh.nodes)):
        b = dof_of.get(n, -1)
        out.append((vec[b], vec[b + 1]) if b >= 0 else (0.0, 0.0))
    return out


def plot_modes(mesh: Mesh2D, freqs: List[float], vectors: List[List[float]],
               dof_of: Dict[int, int], path: str,
               panel: Tuple[int, int] = (420, 460),
               scale_frac: float = 0.10) -> None:
    """One panel per mode: grey undeformed mesh + colour-mapped deformed."""
    if not freqs:
        return
    pw, ph = panel
    n = len(freqs)
    W, H = pw * n, ph
    img = bytearray(BG * (W * H))

    xs = [p[0] for p in mesh.nodes]
    ys = [p[1] for p in mesh.nodes]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    extent = max(x1 - x0, y1 - y0)
    label_h = 26
    m = 18
    sc = min((pw - 2 * m) / max(x1 - x0, 1e-9),
             (ph - 2 * m - label_h) / max(y1 - y0, 1e-9)) / (1 + 2 * scale_frac)

    for k, (f, vec) in enumerate(zip(freqs, vectors)):
        disp = _node_disp(mesh, vec, dof_of)
        umax = max(math.hypot(ux, uy) for ux, uy in disp) or 1.0
        s = scale_frac * extent / umax
        offx = k * pw + (pw - (x1 - x0) * sc) / 2 - x0 * sc
        offy = (ph - label_h - (y1 - y0) * sc) / 2 - y0 * sc

        def tx(x):
            return offx + x * sc

        def ty(y):
            return (ph - label_h) - offy - y * sc + 0

        # undeformed wireframe
        for (n0, n1, n2, n3, _dx, _dy, _f) in mesh.elems:
            quad = [mesh.nodes[i] for i in (n0, n1, n2, n3)]
            for a, b in zip(quad, quad[1:] + quad[:1]):
                _line(img, W, H, tx(a[0]), ty(a[1]), tx(b[0]), ty(b[1]), EDGE)

        # deformed, coloured by |u|
        pts, cols = [], []
        for (px, py), (ux, uy) in zip(mesh.nodes, disp):
            pts.append((tx(px + s * ux), ty(py + s * uy)))
            cols.append(_cmap(math.hypot(ux, uy) / umax))
        for (n0, n1, n2, n3, _dx, _dy, _f) in mesh.elems:
            _fill_tri(img, W, H, pts[n0], pts[n1], pts[n2],
                      cols[n0], cols[n1], cols[n2])
            _fill_tri(img, W, H, pts[n0], pts[n2], pts[n3],
                      cols[n0], cols[n2], cols[n3])

        _text(img, W, H, k * pw + m, ph - label_h + 4,
              f"MODE {k + 1}  {f / 1e3:.2f} KHZ", TEXT, scale=2)

    write_png(path, W, H, img)


# ---------------------------------------------------------------------------
# Allan deviation plot (log-log)
# ---------------------------------------------------------------------------
def plot_allan(pts, arw_analytic: float, arw_est: float, path: str,
               size: Tuple[int, int] = (640, 480)) -> None:
    """Simulated Allan deviation points vs the analytic ARW/sqrt(tau) line.

    ``pts`` is [(tau_s, sigma_rad_per_s)]; ARW values in rad/sqrt(s).
    """
    W, H = size
    img = bytearray(BG * (W * H))
    ml, mr, mt, mb = 64, 20, 38, 46          # margins

    taus = [t for t, _ in pts]
    sigs = [s for _, s in pts]
    lx0 = math.floor(math.log10(min(taus)))
    lx1 = math.ceil(math.log10(max(taus)))
    ana = [arw_analytic / math.sqrt(t) for t in taus]
    ymin = min(min(sigs), min(ana))
    ymax = max(max(sigs), max(ana))
    ly0 = math.floor(math.log10(ymin))
    ly1 = math.ceil(math.log10(ymax))

    def tx(t):
        return ml + (math.log10(t) - lx0) / max(lx1 - lx0, 1) * (W - ml - mr)

    def ty(s):
        return H - mb - (math.log10(s) - ly0) / max(ly1 - ly0, 1) \
            * (H - mt - mb)

    # decade grid + tick labels
    grid = (44, 44, 54)
    for d in range(lx0, lx1 + 1):
        x = tx(10.0 ** d)
        _line(img, W, H, x, mt, x, H - mb, grid)
        _text(img, W, H, int(x) - 14, H - mb + 8, f"1E{d}", TEXT, 1)
    for d in range(ly0, ly1 + 1):
        y = ty(10.0 ** d)
        _line(img, W, H, ml, y, W - mr, y, grid)
        _text(img, W, H, 8, int(y) - 4, f"1E{d}", TEXT, 1)

    # analytic ARW/sqrt(tau) line
    line_c = (110, 110, 125)
    for (t0, s0), (t1, s1) in zip(list(zip(taus, ana)), list(zip(taus, ana))[1:]):
        _line(img, W, H, tx(t0), ty(s0), tx(t1), ty(s1), line_c)

    # simulated points (3x3 squares, colour-mapped)
    for t, s in pts:
        px, py = int(tx(t)), int(ty(s))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                x, y = px + dx, py + dy
                if 0 <= x < W and 0 <= y < H:
                    o = (y * W + x) * 3
                    img[o:o + 3] = bytes((240, 170, 60))

    deg = 180.0 / math.pi * 60.0
    _text(img, W, H, ml, 8, "ALLAN DEVIATION RAD/S VS TAU S", TEXT, 2)
    _text(img, W, H, ml, H - 18,
          f"ARW {arw_est * deg:.4f} DEG/RT-H  ANALYTIC "
          f"{arw_analytic * deg:.4f}", (240, 170, 60), 1)
    write_png(path, W, H, img)


# ---------------------------------------------------------------------------
# deformed 3D render
# ---------------------------------------------------------------------------
def _densify(ring: List[G.Pt], step: float) -> List[G.Pt]:
    out: List[G.Pt] = []
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        out.append((x0, y0))
        d = math.hypot(x1 - x0, y1 - y0)
        for j in range(1, int(d // step) + 1):
            t = j * step / d
            if t < 0.999:
                out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def render_deformed_3d(island_shapes: List[G.Shape],
                       other_shapes: List[G.Shape],
                       proc, mesh2d: Mesh2D, vec: List[float],
                       dof_of: Dict[int, int], path: str,
                       scale_frac: float = 0.06,
                       include_handle: bool = True) -> None:
    """Re-extrude the island with the mode shape applied (exaggerated) and
    render it with the standard isometric renderer."""
    disp = _node_disp(mesh2d, vec, dof_of)
    umax = max(math.hypot(ux, uy) for ux, uy in disp) or 1.0
    xs = [p[0] for p in mesh2d.nodes]
    ys = [p[1] for p in mesh2d.nodes]
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    s = scale_frac * extent / umax
    nodes = mesh2d.nodes

    def u_at(x: float, y: float) -> Tuple[float, float]:
        best, bd = 0, None
        for i, (px, py) in enumerate(nodes):
            d = (px - x) ** 2 + (py - y) ** 2
            if bd is None or d < bd:
                best, bd = i, d
        ux, uy = disp[best]
        return s * ux, s * uy

    def deformed(poly: G.Polygon, step: float = 10.0) -> G.Polygon:
        ext = _densify(poly.exterior, step)
        us = [u_at(x, y) for (x, y) in ext]
        dux = max(u[0] for u in us) - min(u[0] for u in us)
        duy = max(u[1] for u in us) - min(u[1] for u in us)
        # nearly-rigid shapes (the plate, bars, fingers) translate as a
        # whole: this keeps them rectilinear -> clean watertight grid mesh
        if math.hypot(dux, duy) < max(0.5, 0.05 * s * umax):
            mx = sum(u[0] for u in us) / len(us)
            my = sum(u[1] for u in us) / len(us)
            return poly.translated(mx, my)
        warped = [(x + ux, y + uy) for (x, y), (ux, uy) in zip(ext, us)]
        holes = []
        for h in poly.holes:
            hh = []
            for (x, y) in h:
                dx, dy = u_at(x, y)
                hh.append((x + dx, y + dy))
            holes.append(hh)
        return G.Polygon(warped, holes)

    out = M3.Mesh()
    box = proc.box()
    handle = proc.handle()
    all_shapes = island_shapes + other_shapes
    if include_handle and handle is not None and all_shapes:
        x0, y0, x1, y1 = G.bbox_of(all_shapes)
        pad = 20.0
        slab = G.rect_corner(x0 - pad, y0 - pad,
                             (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad)
        out.extend(M3.extrude_polygon(slab, handle.z0, handle.z1, "HANDLE"))

    for sh in island_shapes:
        lay = proc.layers.get(sh.layer)
        if lay is None:
            continue
        poly = deformed(sh.polygon)
        out.extend(M3.extrude_polygon(poly, lay.z0, lay.z1, sh.layer))
        if sh.layer == "DEVICE" and sh.mech == "anchored" and box is not None:
            out.extend(M3.extrude_polygon(G.Polygon(poly.exterior),
                                          box.z0, box.z1, "BOX"))
    for sh in other_shapes:
        lay = proc.layers.get(sh.layer)
        if lay is None:
            continue
        out.extend(M3.extrude_polygon(sh.polygon, lay.z0, lay.z1, sh.layer))
        if sh.layer == "DEVICE" and sh.mech == "anchored" and box is not None:
            out.extend(M3.extrude_polygon(G.Polygon(sh.polygon.exterior),
                                          box.z0, box.z1, "BOX"))

    colors = {name: lay.color for name, lay in proc.layers.items()}
    render_mesh(out, colors, path)
