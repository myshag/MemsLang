"""Top-view SVG preview of the elaborated layout (for quick visual checks)."""

from __future__ import annotations

from typing import Dict, Tuple

from .elaborate import InstanceResult, ProcessInfo


def write_svg(result: InstanceResult, proc: ProcessInfo, path: str,
              px_per_um: float = 1.5) -> None:
    x0, y0, x1, y1 = result.bbox
    pad = 20.0
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    w = (x1 - x0) * px_per_um
    h = (y1 - y0) * px_per_um

    def tx(x): return (x - x0) * px_per_um

    def ty(y): return (y1 - y) * px_per_um   # flip Y for screen coords

    # draw order bottom-up so DEVICE sits on top of HANDLE/BOX tints
    order = list(reversed(proc.order))
    by_layer: Dict[str, list] = {}
    for sh in result.shapes:
        by_layer.setdefault(sh.layer, []).append(sh)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" '
        f'height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}">',
        f'<rect width="{w:.0f}" height="{h:.0f}" fill="#1b1b20"/>',
    ]
    for layer in order:
        for sh in by_layer.get(layer, []):
            fill = _hex(proc.layers[sh.layer].color)
            opacity = 0.55 if sh.mech == "anchored" else 0.95
            d = _path(sh.polygon, tx, ty)
            parts.append(
                f'<path d="{d}" fill="{fill}" fill-opacity="{opacity}" '
                f'fill-rule="evenodd" stroke="#000" stroke-width="0.3"/>')
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def _path(poly, tx, ty) -> str:
    segs = []
    for ring in [poly.exterior] + list(poly.holes):
        if not ring:
            continue
        pts = " ".join(
            f"{'M' if i == 0 else 'L'}{tx(x):.2f},{ty(y):.2f}"
            for i, (x, y) in enumerate(ring))
        segs.append(pts + " Z")
    return " ".join(segs)


def _hex(c: Tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(v * 255))) for v in c)
