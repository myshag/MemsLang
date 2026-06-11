"""Calibrated anisotropic etch-rate diagram R(n) for wet etching of silicon.

Instead of the three-number {100}/{110}/{111} model in ``koh.WetEtch``, this
stores *measured* etch rates of the low-index planes for named process
conditions (KOH/TMAH at a given concentration and temperature) and reconstructs
a continuous, cubic-symmetry-correct rate diagram R(n) over the unit sphere by
interpolating between them.  The diagram is sampled onto a (theta, phi) lookup
table that the C level-set kernel reads per cell -- so the crystallography is
*data*, not code.

The anchor rates are representative literature values (Seidel et al., J.
Electrochem. Soc. 137 (1990) 3612; Sato et al., Sensors & Actuators A 73 (1999)
131).  Absolute numbers depend strongly on concentration, temperature, and
stirring, so every set is a calibration starting point, not a constant of
nature -- edit the dict or use ``.scaled()`` to match your lab.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

Vec = Tuple[float, float, float]


def _norm(v: Vec) -> Vec:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


# --- cubic m-3m point group: 48 ops = axis permutations x sign flips --------
def _cubic_ops():
    ops = []
    for perm in itertools.permutations((0, 1, 2)):
        for sgn in itertools.product((1, -1), repeat=3):
            ops.append((perm, sgn))
    return ops


_OPS = _cubic_ops()


def _family(hkl: Vec) -> List[Vec]:
    """All cubic-symmetry-equivalent unit normals of a plane {hkl}."""
    b = _norm(hkl)
    seen, out = set(), []
    for perm, sgn in _OPS:
        w = (sgn[0] * b[perm[0]], sgn[1] * b[perm[1]], sgn[2] * b[perm[2]])
        key = tuple(round(c, 6) for c in w)
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# named process conditions: rates relative to {100}, with an absolute scale
# ---------------------------------------------------------------------------
# Each entry: r100_um_min = absolute {100} rate, planes = {hkl: rate/R100}.
KOH_30_70 = {
    "name": "KOH 30wt% / 70C",
    "r100_um_min": 0.80,
    "planes": {
        (1, 0, 0): 1.00,
        (1, 1, 0): 1.75,     # {110} is the fastest low-index plane in KOH
        (3, 1, 1): 1.45,     # high-index peaks -> convex-corner undercut
        (2, 1, 1): 1.30,
        (1, 1, 1): 0.013,    # {111} ~ 80x slower than {100}
    },
}

KOH_44_85 = {
    "name": "KOH 44wt% / 85C",
    "r100_um_min": 1.40,
    "planes": {
        (1, 0, 0): 1.00,
        (1, 1, 0): 1.40,     # the {110}/{100} ratio shrinks at high conc.
        (3, 1, 1): 1.20,
        (2, 1, 1): 1.10,
        (1, 1, 1): 0.005,    # higher anisotropy (R100/R111 ~ 200)
    },
}

TMAH_25_80 = {
    "name": "TMAH 25wt% / 80C",
    "r100_um_min": 0.60,
    "planes": {
        (1, 0, 0): 1.00,
        (1, 1, 0): 1.40,
        (3, 1, 1): 1.30,
        (2, 1, 1): 1.25,
        (1, 1, 1): 0.025,    # TMAH less anisotropic than KOH
    },
}

CONDITIONS = {"KOH_30_70": KOH_30_70, "KOH_44_85": KOH_44_85,
              "TMAH_25_80": TMAH_25_80}


# ---------------------------------------------------------------------------
@dataclass
class RateDiagram:
    """Continuous etch rate R(n) (um/min) over the unit sphere, reconstructed
    from measured low-index planes by symmetry-aware RBF interpolation."""
    points: Dict[Vec, float]      # symmetry-expanded normal -> absolute rate
    sigma_deg: float = 6.0        # interpolation width (sharpness of the cusps)
    name: str = ""

    @classmethod
    def from_condition(cls, cond, sigma_deg: float = 6.0) -> "RateDiagram":
        if isinstance(cond, str):
            cond = CONDITIONS[cond]
        r100 = cond["r100_um_min"]
        pts: Dict[Vec, float] = {}
        for hkl, rel in cond["planes"].items():
            rate = rel * r100
            for v in _family(hkl):
                pts[v] = rate
        return cls(pts, sigma_deg, cond.get("name", ""))

    def rate(self, n: Vec) -> float:
        """Etch rate (um/min) of a surface whose outward normal is ``n``."""
        n = _norm(n)
        s = math.radians(self.sigma_deg)
        num = den = 0.0
        for v, r in self.points.items():
            d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2]
            d = 1.0 if d > 1.0 else (-1.0 if d < -1.0 else d)
            ang = math.acos(d)
            w = math.exp(-(ang / s) * (ang / s))
            num += w * r
            den += w
        return num / den if den else 0.0

    def plane_rate(self, hkl: Vec) -> float:
        return self.rate(hkl)

    def scaled(self, factor: float) -> "RateDiagram":
        return RateDiagram({v: r * factor for v, r in self.points.items()},
                           self.sigma_deg, self.name)

    def rmax(self) -> float:
        return max(self.points.values())


# ---------------------------------------------------------------------------
# wafer orientation: crystal-frame basis vectors (X, Y, Z) of the wafer axes,
# where Z is the surface normal and X, Y span the wafer surface.
# ---------------------------------------------------------------------------
def _rodrigues(v: Vec, k: Vec, ang: float) -> Vec:
    """Rotate vector ``v`` about unit axis ``k`` by ``ang`` radians."""
    c, s = math.cos(ang), math.sin(ang)
    kxv = (k[1] * v[2] - k[2] * v[1],
           k[2] * v[0] - k[0] * v[2],
           k[0] * v[1] - k[1] * v[0])
    kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2]
    return tuple(v[i] * c + kxv[i] * s + k[i] * kd * (1.0 - c) for i in range(3))


def wafer_basis(orientation: str, misalign_deg: float = 0.0,
                miscut_deg: float = 0.0, miscut_az: float = 0.0
                ) -> Tuple[Vec, Vec, Vec]:
    """Crystal-frame basis (X, Y, Z) of the wafer axes (Z = surface normal).

    ``misalign_deg``  rotates the mask in-plane about Z -- i.e. the mask edges
                      are turned off the wafer flat (<110>); this is what makes
                      even straight edges and concave corners undercut.
    ``miscut_deg/az`` tilts Z itself off the ideal pole (an off-axis / vicinal
                      wafer), tipping the facets asymmetrically.
    """
    o = orientation.replace("(", "").replace(")", "").strip()
    if o == "100":
        X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    elif o == "110":
        X, Y, Z = _norm((-1, 1, 0)), (0.0, 0.0, 1.0), _norm((1, 1, 0))
    elif o == "111":
        X, Y, Z = _norm((1, -1, 0)), _norm((1, 1, -2)), _norm((1, 1, 1))
    else:
        raise ValueError("unknown wafer orientation %r (use 100/110/111)" % orientation)
    if miscut_deg:                       # tilt the whole frame about an in-plane axis
        a = math.radians(miscut_az)
        axis = _norm((X[0] * math.cos(a) + Y[0] * math.sin(a),
                      X[1] * math.cos(a) + Y[1] * math.sin(a),
                      X[2] * math.cos(a) + Y[2] * math.sin(a)))
        m = math.radians(miscut_deg)
        X, Y, Z = (_rodrigues(X, axis, m), _rodrigues(Y, axis, m),
                   _rodrigues(Z, axis, m))
    if misalign_deg:                     # spin the mask in-plane about Z
        r = math.radians(misalign_deg)
        X, Y = _rodrigues(X, Z, r), _rodrigues(Y, Z, r)
    return X, Y, Z


def sample_table(diagram: RateDiagram, orientation: str = "100",
                 ntheta: int = 90, nphi: int = 180, misalign_deg: float = 0.0,
                 miscut_deg: float = 0.0, miscut_az: float = 0.0
                 ) -> Tuple[List[float], float]:
    """Sample the diagram onto a (theta, phi) table in the *wafer* frame:
    theta in [0, pi] from the surface normal, phi in [0, 2pi).  Returns the
    flat table normalised to max-rate 1.0 plus the absolute max rate (um/min)
    so callers know the real scale."""
    X, Y, Z = wafer_basis(orientation, misalign_deg, miscut_deg, miscut_az)
    tab = [0.0] * (ntheta * nphi)
    rmax = 0.0
    for it in range(ntheta):
        th = math.pi * it / (ntheta - 1)
        st, ct = math.sin(th), math.cos(th)
        for ip in range(nphi):
            ph = 2 * math.pi * ip / nphi
            wx, wy, wz = st * math.cos(ph), st * math.sin(ph), ct
            # wafer-frame direction -> crystal frame
            cx = X[0] * wx + Y[0] * wy + Z[0] * wz
            cy = X[1] * wx + Y[1] * wy + Z[1] * wz
            cz = X[2] * wx + Y[2] * wy + Z[2] * wz
            r = diagram.rate((cx, cy, cz))
            tab[it * nphi + ip] = r
            if r > rmax:
                rmax = r
    if rmax > 0:
        tab = [r / rmax for r in tab]
    return tab, rmax


# ---------------------------------------------------------------------------
# stereographic projection of the upper hemisphere -> rate-diagram PNG
# ---------------------------------------------------------------------------
def _colormap(t: float) -> Tuple[int, int, int]:
    """t in [0,1] -> blue(slow) .. cyan .. yellow .. red(fast)."""
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    stops = [(0.0, (30, 40, 110)), (0.35, (30, 150, 160)),
             (0.65, (235, 205, 60)), (1.0, (200, 50, 40))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(int(c0[k] + f * (c1[k] - c0[k])) for k in range(3))
    return stops[-1][1]


def render_diagram(diagram: RateDiagram, path: str, orientation: str = "100",
                   size: int = 340, misalign_deg: float = 0.0,
                   miscut_deg: float = 0.0, miscut_az: float = 0.0) -> None:
    """Stereographic plot of the upper hemisphere R(n) in the wafer frame, with
    the {100}/{110}/{111} poles marked."""
    from .render import write_png
    X, Y, Z = wafer_basis(orientation, misalign_deg, miscut_deg, miscut_az)
    W = H = size
    img = bytearray([18, 18, 24] * (W * H))
    rmax = diagram.rmax() or 1.0
    cx0 = cy0 = size / 2
    R = size / 2 - 6
    for py in range(H):
        for px in range(W):
            u = (px - cx0) / R
            v = (cy0 - py) / R
            rr = u * u + v * v
            if rr > 1.0:
                continue
            # inverse stereographic: disk -> upper hemisphere (wafer frame)
            wz = (1 - rr) / (1 + rr)
            wx = 2 * u / (1 + rr)
            wy = 2 * v / (1 + rr)
            cxv = X[0] * wx + Y[0] * wy + Z[0] * wz
            cyv = X[1] * wx + Y[1] * wy + Z[1] * wz
            czv = X[2] * wx + Y[2] * wy + Z[2] * wz
            r = diagram.rate((cxv, cyv, czv))
            o = (py * W + px) * 3
            img[o:o + 3] = bytes(_colormap(r / rmax))
    # mark the principal poles (project crystal directions into the wafer disk)
    def mark(hkl, col):
        c = _norm(hkl)
        wx = X[0] * c[0] + X[1] * c[1] + X[2] * c[2]
        wy = Y[0] * c[0] + Y[1] * c[1] + Y[2] * c[2]
        wz = Z[0] * c[0] + Z[1] * c[1] + Z[2] * c[2]
        if wz < 0:
            return
        u, vv = wx / (1 + wz), wy / (1 + wz)
        px = int(cx0 + u * R)
        py = int(cy0 - vv * R)
        for ddx in range(-2, 3):
            for ddy in range(-2, 3):
                qx, qy = px + ddx, py + ddy
                if 0 <= qx < W and 0 <= qy < H and ddx * ddx + ddy * ddy <= 4:
                    img[(qy * W + qx) * 3:(qy * W + qx) * 3 + 3] = bytes(col)
    for hkl in ((0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
                (1, 1, -1), (-1, 1, 1), (1, -1, 1)):
        mark(hkl, (245, 245, 245))
    write_png(path, W, H, bytes(img))
