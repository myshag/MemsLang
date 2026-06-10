"""Zero-dependency software renderer: 3D mesh -> shaded PNG (isometric view).

Used both as a verification aid and as a shareable preview artifact.  Includes
a minimal PNG encoder (stdlib ``zlib`` only) and a z-buffered triangle
rasteriser with simple Lambert shading.
"""

from __future__ import annotations

import math
import struct
import zlib
from typing import Dict, List, Tuple

from . import mesh as M


# ---- PNG encoder ----------------------------------------------------------
def write_png(path: str, width: int, height: int, rgb: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        raw.extend(rgb[y * width * 3:(y + 1) * width * 3])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        f.write(chunk(b"IEND", b""))


# ---- geometry helpers -----------------------------------------------------
def _rotate(p, az, el):
    x, y, z = p
    ca, sa = math.cos(az), math.sin(az)
    x, y = x * ca - y * sa, x * sa + y * ca
    ce, se = math.cos(el), math.sin(el)
    y, z = y * ce - z * se, y * se + z * ce
    return x, y, z


def render_mesh(mesh: M.Mesh, colors: Dict[str, Tuple[float, float, float]],
                path: str, size: Tuple[int, int] = (900, 680),
                az_deg: float = 35.0, el_deg: float = 58.0,
                bg=(24, 24, 30)) -> None:
    W, H = size
    az, el = math.radians(az_deg), math.radians(el_deg)

    # transform vertices into view space
    vt = [_rotate(v, az, el) for v in mesh.vertices]
    if not vt:
        write_png(path, W, H, bytearray(bg * (W * H)))
        return
    xs = [p[0] for p in vt]
    zs = [p[2] for p in vt]
    minx, maxx = min(xs), max(xs)
    minz, maxz = min(zs), max(zs)
    spanx = (maxx - minx) or 1.0
    spanz = (maxz - minz) or 1.0
    margin = 0.08
    scale = min(W * (1 - 2 * margin) / spanx, H * (1 - 2 * margin) / spanz)
    ox = (W - spanx * scale) / 2 - minx * scale
    oz = (H - spanz * scale) / 2 - minz * scale

    def project(p):
        # screen x from view-x, screen y from view-z (z is "up"); depth = view-y
        sx = p[0] * scale + ox
        sy = H - (p[2] * scale + oz)
        return sx, sy, p[1]

    light = _norm((0.4, -0.5, 0.85))
    img = bytearray(bg * (W * H))
    zbuf = [1e30] * (W * H)

    for (a, b, c), g in zip(mesh.triangles, mesh.tri_group):
        pa, pb, pc = vt[a], vt[b], vt[c]
        n = _norm(_cross(_sub(pb, pa), _sub(pc, pa)))
        shade = 0.25 + 0.75 * max(0.0, _dot(n, light))
        base = colors.get(g, (0.6, 0.6, 0.6))
        col = (int(min(255, base[0] * 255 * shade)),
               int(min(255, base[1] * 255 * shade)),
               int(min(255, base[2] * 255 * shade)))
        _raster(img, zbuf, W, H, project(pa), project(pb), project(pc), col)

    write_png(path, W, H, img)


def _raster(img, zbuf, W, H, A, B, C, col):
    ax, ay, az = A
    bx, by, bz = B
    cx, cy, cz = C
    minx = max(0, int(math.floor(min(ax, bx, cx))))
    maxx = min(W - 1, int(math.ceil(max(ax, bx, cx))))
    miny = max(0, int(math.floor(min(ay, by, cy))))
    maxy = min(H - 1, int(math.ceil(max(ay, by, cy))))
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(area) < 1e-9:
        return
    inv = 1.0 / area
    r, g, b = col
    for y in range(miny, maxy + 1):
        py = y + 0.5
        for x in range(minx, maxx + 1):
            px = x + 0.5
            w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * inv
            w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * inv
            w2 = 1.0 - w0 - w1
            if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                continue
            depth = w0 * az + w1 * bz + w2 * cz
            idx = y * W + x
            if depth < zbuf[idx]:
                zbuf[idx] = depth
                o = idx * 3
                img[o] = r
                img[o + 1] = g
                img[o + 2] = b


def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    m = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / m, a[1] / m, a[2] / m)
