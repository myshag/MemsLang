"""Mesh exporters: binary STL, OBJ + MTL (per-layer colours)."""

from __future__ import annotations

import struct
from typing import Dict, Tuple

from . import mesh as M


def _normal(v0, v1, v2):
    ux, uy, uz = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    vx, vy, vz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    n = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
    return nx / n, ny / n, nz / n


def write_stl(mesh: M.Mesh, path: str) -> None:
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(mesh.triangles)))
        for (a, b, c) in mesh.triangles:
            v0, v1, v2 = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            nx, ny, nz = _normal(v0, v1, v2)
            f.write(struct.pack("<3f", nx, ny, nz))
            for v in (v0, v1, v2):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def write_obj(mesh: M.Mesh, path: str,
              colors: Dict[str, Tuple[float, float, float]] | None = None
              ) -> None:
    colors = colors or {}
    mtl_path = path.rsplit(".", 1)[0] + ".mtl"
    mtl_name = mtl_path.rsplit("/", 1)[-1]

    # group triangles by material
    groups: Dict[str, list] = {}
    for tri, g in zip(mesh.triangles, mesh.tri_group):
        groups.setdefault(g, []).append(tri)

    with open(path, "w") as f:
        f.write("# SOIDL 2.5D model\n")
        f.write(f"mtllib {mtl_name}\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for g, tris in groups.items():
            f.write(f"g {g}\n")
            f.write(f"usemtl {g}\n")
            for (a, b, c) in tris:
                f.write(f"f {a + 1} {b + 1} {c + 1}\n")

    with open(mtl_path, "w") as f:
        for g in groups:
            r, gr, b = colors.get(g, (0.6, 0.6, 0.6))
            f.write(f"newmtl {g}\n")
            f.write(f"Kd {r:.3f} {gr:.3f} {b:.3f}\n")
            f.write("Ka 0.1 0.1 0.1\nKs 0.2 0.2 0.2\nNs 16\n\n")


def mesh_stats(mesh: M.Mesh) -> str:
    (x0, y0, z0), (x1, y1, z1) = mesh.bounds()
    return (f"{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles; "
            f"bbox = [{x1 - x0:.1f} x {y1 - y0:.1f} x {z1 - z0:.1f}] um")
