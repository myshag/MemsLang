"""Tiny pure-Python 2D plane-stress FEM for phase-1 verification.

In-plane SOI mechanics is a plane-stress problem with the DEVICE-layer
thickness: this module meshes an electrical island's polygons on a tensor
grid of axis-aligned rectangles, clamps everything under ``anchored`` shapes
(where the BOX survives), and solves

  * static:  K u = f                       (deflection under a load)
  * modal:   K phi = omega^2 M phi          (lowest few resonant modes)

Element: Q4 rectangle enriched with Wilson incompatible modes (Q6),
condensed at element level — accurate in bending even for the slender,
high-aspect elements that MEMS flexures produce (plain Q4 would lock and
over-stiffen thin beams by integer factors).

Solvers: banded Cholesky factorisation (mesh nodes are numbered along the
shorter grid axis to keep the band small) and shifted-free inverse iteration
with M-orthogonal deflation for the lowest modes.  Pure standard library.

Simplifications (documented, deliberate for phase 1):
  * release holes are smeared into the mass density (fill factor), they are
    not meshed — perforation barely affects plate stiffness, only mass;
  * comb *finger* shapes are not meshed; their mass is lumped onto the
    nearest backbone node (fingers add mass, negligible stiffness).
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import geometry as G

MAX_ELEMENTS = 30000


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
@dataclass
class Mesh2D:
    nodes: List[Tuple[float, float]] = field(default_factory=list)   # um
    # n0..n3 CCW from lower-left, cell size (um), mass fill factor
    elems: List[Tuple[int, int, int, int, float, float, float]] = \
        field(default_factory=list)
    fixed: set = field(default_factory=set)
    extra_mass: List[Tuple[int, float]] = field(default_factory=list)  # um^2

    @property
    def n_free_dof(self) -> int:
        return 2 * (len(self.nodes) - len(self.fixed))


def _dedupe(vals, tol=1e-6):
    out: List[float] = []
    for v in sorted(vals):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _subdivide(lines: List[float], h: float) -> List[float]:
    out = [lines[0]]
    for a, b in zip(lines, lines[1:]):
        gap = b - a
        if gap > h:
            k = int(math.ceil(gap / h))
            for i in range(1, k):
                out.append(a + gap * i / k)
        out.append(b)
    return out


def build_mesh(shapes: List[G.Shape], h_max: float,
               lump_label: str = "finger") -> Mesh2D:
    """Tensor-grid quad mesh over an island's rectilinear polygons."""
    solids = [s for s in shapes if lump_label not in s.label]
    lumped = [s for s in shapes if lump_label in s.label]
    mesh = Mesh2D()
    if not solids:
        return mesh

    xs, ys = set(), set()
    boxes = []
    for s in solids:
        x0, y0, x1, y1 = s.polygon.bbox()
        boxes.append((x0, y0, x1, y1, s))
        xs.update((round(x0, 6), round(x1, 6)))
        ys.update((round(y0, 6), round(y1, 6)))
        # at least two elements across a narrow feature (beam widths)
        if x1 - x0 <= 1.5 * h_max:
            xs.add(round((x0 + x1) / 2, 6))
        if y1 - y0 <= 1.5 * h_max:
            ys.add(round((y0 + y1) / 2, 6))
    gx = _subdivide(_dedupe(xs), h_max)
    gy = _subdivide(_dedupe(ys), h_max)

    nx, ny = len(gx) - 1, len(gy) - 1
    if nx < 1 or ny < 1:
        return mesh

    cells = {}          # (i, j) -> (fill, anchored)
    for i in range(nx):
        cx = (gx[i] + gx[i + 1]) / 2
        for j in range(ny):
            cy = (gy[j] + gy[j + 1]) / 2
            fill = 0.0
            anchored = False
            hit = False
            for (x0, y0, x1, y1, s) in boxes:
                if x0 < cx < x1 and y0 < cy < y1:
                    hit = True
                    ext = abs(G.signed_area(s.polygon.exterior))
                    fill = max(fill, s.polygon.area() / ext if ext else 1.0)
                    if s.mech == "anchored":
                        anchored = True
            if hit:
                cells[(i, j)] = (fill, anchored)

    # node numbering along the axis with fewer grid lines -> small bandwidth
    corner_ids: Dict[Tuple[int, int], int] = {}
    corners = set()
    for (i, j) in cells:
        corners.update(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))
    if len(gy) <= len(gx):
        order = sorted(corners, key=lambda c: (c[0], c[1]))
    else:
        order = sorted(corners, key=lambda c: (c[1], c[0]))
    for n, (i, j) in enumerate(order):
        corner_ids[(i, j)] = n
        mesh.nodes.append((gx[i], gy[j]))

    for (i, j), (fill, anchored) in sorted(cells.items()):
        n00 = corner_ids[(i, j)]
        n10 = corner_ids[(i + 1, j)]
        n11 = corner_ids[(i + 1, j + 1)]
        n01 = corner_ids[(i, j + 1)]
        if anchored:
            mesh.fixed.update((n00, n10, n11, n01))
        else:
            mesh.elems.append((n00, n10, n11, n01,
                               gx[i + 1] - gx[i], gy[j + 1] - gy[j], fill))

    # lump excluded shapes (comb fingers) as point masses
    for s in lumped:
        x0, y0, x1, y1 = s.polygon.bbox()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best, best_d = None, None
        for n, (px, py) in enumerate(mesh.nodes):
            if n in mesh.fixed:
                continue
            d = (px - cx) ** 2 + (py - cy) ** 2
            if best_d is None or d < best_d:
                best, best_d = n, d
        if best is not None:
            mesh.extra_mass.append((best, s.polygon.area()))
    return mesh


# ---------------------------------------------------------------------------
# Element stiffness: Q4 + Wilson incompatible modes, axis-aligned rectangle
# ---------------------------------------------------------------------------
def _solve_dense(A: List[List[float]], B: List[List[float]]):
    """Solve A X = B for small dense A (Gauss, partial pivoting)."""
    n = len(A)
    m = len(B[0])
    M = [A[i][:] + B[i][:] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / piv
            if f:
                Mr, Mc = M[r], M[c]
                for k in range(c, n + m):
                    Mr[k] -= f * Mc[k]
    X = [[0.0] * m for _ in range(n)]
    for r in range(n - 1, -1, -1):
        Mr = M[r]
        for k in range(m):
            s = Mr[n + k] - sum(Mr[c] * X[c][k] for c in range(r + 1, n))
            X[r][k] = s / Mr[r]
    return X


_KE_CACHE: Dict[Tuple[float, float, float, float, float], list] = {}


def _q6_ke(dx: float, dy: float, E: float, nu: float, t: float):
    """Condensed 8x8 stiffness (N/m); dx, dy in metres."""
    key = (round(dx, 12), round(dy, 12), E, nu, t)
    ke = _KE_CACHE.get(key)
    if ke is not None:
        return ke
    D00 = E / (1.0 - nu * nu)
    D01 = nu * D00
    D22 = 0.5 * E / (1.0 + nu)
    XI = (-1.0, 1.0, 1.0, -1.0)
    ETA = (-1.0, -1.0, 1.0, 1.0)
    jx, jy = 2.0 / dx, 2.0 / dy
    w = 0.25 * dx * dy * t           # gauss weight * |J| * thickness
    K = [[0.0] * 12 for _ in range(12)]
    g = 1.0 / math.sqrt(3.0)
    for xi in (-g, g):
        for eta in (-g, g):
            B0 = [0.0] * 12
            B1 = [0.0] * 12
            B2 = [0.0] * 12
            for k in range(4):
                dNdx = 0.25 * XI[k] * (1.0 + eta * ETA[k]) * jx
                dNdy = 0.25 * ETA[k] * (1.0 + xi * XI[k]) * jy
                B0[2 * k] = dNdx
                B2[2 * k] = dNdy
                B1[2 * k + 1] = dNdy
                B2[2 * k + 1] = dNdx
            p1x = -2.0 * xi * jx      # d(1-xi^2)/dx
            p2y = -2.0 * eta * jy     # d(1-eta^2)/dy
            B0[8] = p1x               # ux ~ (1-xi^2)
            B2[9] = p2y               # ux ~ (1-eta^2) -> shear only
            B2[10] = p1x              # uy ~ (1-xi^2) -> shear only
            B1[11] = p2y              # uy ~ (1-eta^2)
            for a in range(12):
                b0, b1, b2 = B0[a], B1[a], B2[a]
                if b0 == 0.0 and b1 == 0.0 and b2 == 0.0:
                    continue
                d0 = D00 * b0 + D01 * b1
                d1 = D01 * b0 + D00 * b1
                d2 = D22 * b2
                Ka = K[a]
                for c in range(a, 12):
                    Ka[c] += (d0 * B0[c] + d1 * B1[c] + d2 * B2[c]) * w
    for a in range(12):
        for c in range(a + 1, 12):
            K[c][a] = K[a][c]
    # static condensation of the 4 incompatible dofs
    Kaa = [[K[8 + i][8 + j] for j in range(4)] for i in range(4)]
    Kau = [[K[8 + i][j] for j in range(8)] for i in range(4)]
    X = _solve_dense(Kaa, Kau)                       # Kaa^-1 Kau
    ke = [[K[i][j] - sum(K[i][8 + r] * X[r][j] for r in range(4))
           for j in range(8)] for i in range(8)]
    _KE_CACHE[key] = ke
    return ke


# ---------------------------------------------------------------------------
# Assembly (banded, lower triangle) and solvers
# ---------------------------------------------------------------------------
def _assemble(mesh: Mesh2D, E: float, nu: float, rho: float, t: float):
    dof_of: Dict[int, int] = {}
    nxt = 0
    for n in range(len(mesh.nodes)):
        if n not in mesh.fixed:
            dof_of[n] = nxt
            nxt += 2
    ndof = nxt
    if ndof == 0:
        return None, 0, [], dof_of, 0

    # bandwidth
    band = 0
    for (n0, n1, n2, n3, _dx, _dy, _f) in mesh.elems:
        ds = [dof_of[n] for n in (n0, n1, n2, n3) if n in dof_of]
        if len(ds) > 1:
            band = max(band, max(ds) + 1 - min(ds))
    rows = [[0.0] * (min(i, band) + 1) for i in range(ndof)]
    Mdiag = [0.0] * ndof

    for (n0, n1, n2, n3, dx, dy, fill) in mesh.elems:
        ke = _q6_ke(dx * 1e-6, dy * 1e-6, E, nu, t)
        nodes = (n0, n1, n2, n3)
        gd = []
        for k, n in enumerate(nodes):
            base = dof_of.get(n, -1)
            gd.append(base)
            gd.append(-1 if base < 0 else base + 1)
        for a in range(8):
            ga = gd[a]
            if ga < 0:
                continue
            ra = rows[ga]
            sa = ga - len(ra) + 1
            kea = ke[a]
            for c in range(8):
                gb = gd[c]
                if gb < 0 or gb > ga:
                    continue
                ra[gb - sa] += kea[c]
        # lumped mass: quarter of the cell to each node
        mq = rho * t * (dx * 1e-6) * (dy * 1e-6) * fill * 0.25
        for n in nodes:
            base = dof_of.get(n, -1)
            if base >= 0:
                Mdiag[base] += mq
                Mdiag[base + 1] += mq

    for (n, area_um2) in mesh.extra_mass:
        base = dof_of.get(n, -1)
        if base >= 0:
            mkg = area_um2 * 1e-12 * t * rho
            Mdiag[base] += mkg
            Mdiag[base + 1] += mkg
    return rows, band, Mdiag, dof_of, ndof


def _chol_banded(rows) -> None:
    """In-place Cholesky of a banded SPD matrix (lower rows)."""
    dot = _dot
    for i in range(len(rows)):
        ri = rows[i]
        si = i - len(ri) + 1
        for j in range(si, i + 1):
            rj = rows[j]
            sj = j - len(rj) + 1
            kmin = max(si, sj)
            s = ri[j - si]
            if kmin < j:
                s -= dot(ri[kmin - si: j - si], rj[kmin - sj: j - sj])
            if j < i:
                ri[j - si] = s / rj[-1]
            else:
                if s <= 0.0:
                    raise ArithmeticError("matrix not positive definite "
                                          "(unconstrained mechanism?)")
                ri[-1] = math.sqrt(s)


def _dot(a, b) -> float:
    return sum(map(operator.mul, a, b))


def _solve_factored(rows, rhs: List[float]) -> List[float]:
    n = len(rows)
    y = [0.0] * n
    for i in range(n):
        ri = rows[i]
        si = i - len(ri) + 1
        s = rhs[i]
        if si < i:
            s -= _dot(ri[: i - si], y[si: i])
        y[i] = s / ri[-1]
    x = y[:]
    for i in range(n - 1, -1, -1):
        ri = rows[i]
        si = i - len(ri) + 1
        xi = x[i] / ri[-1]
        x[i] = xi
        if xi:
            for k in range(si, i):
                x[k] -= ri[k - si] * xi
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def static_solve(mesh: Mesh2D, E: float, nu: float, t: float,
                 loads: Dict[int, Tuple[float, float]]
                 ) -> Dict[int, Tuple[float, float]]:
    """Displacements (metres) under nodal loads (newtons)."""
    rows, band, Mdiag, dof_of, ndof = _assemble(mesh, E, nu, 1.0, t)
    if ndof == 0:
        return {}
    rhs = [0.0] * ndof
    for n, (fx, fy) in loads.items():
        base = dof_of.get(n, -1)
        if base >= 0:
            rhs[base] += fx
            rhs[base + 1] += fy
    _chol_banded(rows)
    u = _solve_factored(rows, rhs)
    return {n: (u[b], u[b + 1]) for n, b in dof_of.items()}


def _seed_vec(ndof: int, salt: int) -> List[float]:
    return [math.sin((i + 1) * 12.9898 + salt * 78.233) for i in range(ndof)]


def _m_orth(v: List[float], found, Mdiag) -> None:
    for phi in found:
        c = sum(v[i] * Mdiag[i] * phi[i] for i in range(len(v)))
        if c:
            for i in range(len(v)):
                v[i] -= c * phi[i]


def modal(mesh: Mesh2D, E: float, nu: float, rho: float, t: float,
          n_modes: int = 3, max_iter: int = 100, tol: float = 1e-10,
          return_vectors: bool = False):
    """Lowest natural frequencies (Hz) by inverse iteration + deflation.

    With ``return_vectors=True`` returns ``(freqs, vectors, dof_of)`` where
    each vector is the M-normalised mode shape over the free dofs.
    """
    rows, band, Mdiag, dof_of, ndof = _assemble(mesh, E, nu, rho, t)
    if ndof == 0:
        return ([], [], dof_of) if return_vectors else []
    _chol_banded(rows)
    found: List[List[float]] = []
    freqs: List[float] = []
    for m in range(min(n_modes, ndof)):
        x = _seed_vec(ndof, m)
        _m_orth(x, found, Mdiag)
        lam_prev = 0.0
        lam = 0.0
        for it in range(max_iter):
            rhs = [Mdiag[i] * x[i] for i in range(ndof)]
            y = _solve_factored(rows, rhs)
            _m_orth(y, found, Mdiag)
            num = sum(y[i] * rhs[i] for i in range(ndof))
            den = sum(y[i] * Mdiag[i] * y[i] for i in range(ndof))
            if den <= 0.0:
                x = _seed_vec(ndof, m * 31 + it + 1)
                _m_orth(x, found, Mdiag)
                continue
            lam = num / den
            inv = 1.0 / math.sqrt(den)
            x = [yi * inv for yi in y]
            if it > 2 and abs(lam - lam_prev) < tol * abs(lam):
                break
            lam_prev = lam
        found.append(x)
        freqs.append(math.sqrt(max(lam, 0.0)) / (2.0 * math.pi))
    if return_vectors:
        return freqs, found, dof_of
    return freqs


def analyze(elab, art, h: float = 12.0,
             plot_prefix: Optional[str] = None) -> None:
    """Pipeline hook: modal FEM for every suspended island; appends to the
    report, compares mode 1 against the lumped f0 estimate, and (with a
    ``plot_prefix``) writes mode-shape panels and a deformed 3D render."""
    dev = elab.process.device()
    if dev is None:
        return
    E, nu, rho = dev.E, dev.nu, dev.rho
    t = dev.thickness * 1e-6
    for cid, ss in getattr(elab, "islands", []):
        if not any(s.mech == "anchored" for s in ss):
            continue
        if not any(s.mech == "released" for s in ss):
            continue
        mesh = build_mesh(ss, h)
        if not mesh.elems:
            continue
        if len(mesh.elems) > MAX_ELEMENTS:
            art.warnings.append(
                f"fem: island #{cid} has {len(mesh.elems)} elements "
                f"(> {MAX_ELEMENTS}); increase --fem-h")
            continue
        freqs, vecs, dof_of = modal(mesh, E, nu, rho, t, n_modes=3,
                                    return_vectors=True)
        art.report.append(
            f"fem    island #{cid}: {len(mesh.elems)} elements, "
            f"{mesh.n_free_dof} free dof (h = {h:g} um)")
        if not freqs:
            continue
        art.report.append(
            "fem    island #%d modes: %s" % (
                cid, ", ".join(f"{f / 1e3:.2f} kHz" for f in freqs)))
        f0 = art.model.get("f0")
        if f0 is not None:
            d = (freqs[0] - f0.value) / f0.value * 100.0
            art.report.append(
                f"fem    mode 1 vs lumped f0: {freqs[0] / 1e3:.2f} kHz "
                f"vs {f0.value / 1e3:.2f} kHz ({d:+.1f}%)")
        if plot_prefix:
            from . import femplot
            modes_png = f"{plot_prefix}_fem_island{cid}_modes.png"
            femplot.plot_modes(mesh, freqs, vecs, dof_of, modes_png)
            art.files[f"fem_modes_{cid}"] = modes_png
            island_ids = {id(s) for s in ss}
            others = [s for s in art.result.shapes
                      if id(s) not in island_ids]
            d3_png = f"{plot_prefix}_fem_island{cid}_mode1_3d.png"
            femplot.render_deformed_3d(ss, others, elab.process, mesh,
                                       vecs[0], dof_of, d3_png)
            art.files[f"fem_3d_{cid}"] = d3_png
            art.report.append(
                f"fem    island #{cid} plots: {modes_png}, {d3_png}")
        # model order reduction: modal data -> behavioural model exports
        try:
            from . import reduce as _rom
            _rom.build(elab, art, ss, mesh, freqs, vecs, dof_of,
                       out_prefix=plot_prefix)
        except Exception as e:  # noqa: BLE001 - ROM is best-effort
            art.warnings.append(f"rom: extraction failed: {e}")
