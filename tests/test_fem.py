import math

import numpy as np

from soidlc.fem.result import FemMesh, FemResult
from soidlc import geometry as G
from soidlc.fem import mesh_build
from soidlc.fem import skfem_solve
from soidlc.fem.skfem_solve import _build_p1_mass


def _rect_shape(x0, y0, w, h, **kw):
    return G.Shape(layer="DEVICE", polygon=G.rect_corner(x0, y0, w, h), **kw)


def test_mesh_single_rect():
    beam = _rect_shape(0.0, 0.0, 100.0, 10.0)
    m = mesh_build.build_mesh([beam], h=3.0)
    assert m.n_cells > 50                      # meshed, not degenerate
    assert all(len(c) == 3 for c in m.cells)   # triangles
    x0 = min(p[0] for p in m.nodes)
    x1 = max(p[0] for p in m.nodes)
    assert x0 < 1.0 and x1 > 99.0              # spans the rectangle
    assert len(m.fill) == m.n_cells


def test_femmesh_basic():
    m = FemMesh(
        nodes=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        cells=[(0, 1, 2)],
        fixed={0},
        fill=[1.0],
        extra_mass=[(2, 5.0)],
    )
    assert m.n_cells == 1
    assert len(m.nodes) == 3
    assert 0 in m.fixed


def test_femresult_basic():
    r = FemResult(freqs=[1.0e3], vecs=[[0.1, 0.2]], dof_of={0: 0})
    assert r.freqs[0] == 1.0e3
    assert r.dof_of[0] == 0


def test_mesh_union_is_connected():
    a = _rect_shape(0.0, 0.0, 50.0, 10.0)
    b = _rect_shape(40.0, 0.0, 50.0, 10.0)   # overlaps a
    m = mesh_build.build_mesh([a, b], h=4.0)
    # one connected component: BFS over cell adjacency reaches every used node
    from collections import defaultdict
    adj = defaultdict(set)
    for (i, j, k) in m.cells:
        for u in (i, j, k):
            for v in (i, j, k):
                adj[u].add(v)
    seen, stack = set(), [m.cells[0][0]]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u] - seen)
    used = {n for c in m.cells for n in c}
    assert seen == used                      # fully connected


def test_mesh_anchored_marks_fixed():
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 5.0, 10.0, mech="anchored")
    m = mesh_build.build_mesh([body, anchor], h=3.0)
    assert len(m.fixed) > 0
    assert all(m.nodes[n][0] <= 5.5 for n in m.fixed)


def test_mesh_fingers_lumped_not_meshed():
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    finger = _rect_shape(20.0, 10.0, 2.0, 8.0, label="rotor_finger")
    m = mesh_build.build_mesh([body, finger], h=3.0)
    assert len(m.extra_mass) == 1
    # finger tip y=18 is not in the meshed body (height 10)
    assert max(p[1] for p in m.nodes) <= 10.5


def test_modal_cantilever_matches_euler():
    # clamped silicon beam L=100um H=10um; in-plane bending mode 1
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 2.0, 10.0, mech="anchored")
    m = mesh_build.build_mesh([body, anchor], h=2.0)
    E, nu, rho, t = 170.0e9, 0.28, 2330.0, 10.0e-6
    freqs, vecs, dof_of = skfem_solve.modal(m, E, nu, rho, t, n_modes=3)

    L, H = 100e-6, 10e-6
    I = (H ** 3) / 12.0
    A = H
    f1 = (1.8751 ** 2) / (2 * math.pi) * math.sqrt(E * I / (rho * A * L ** 4))
    assert abs(freqs[0] - f1) / f1 < 0.05          # within 5%
    assert len(vecs) == 3 and len(dof_of) == len(m.nodes)
    # a free tip node has nonzero displacement in mode 1
    tip = max(range(len(m.nodes)), key=lambda n: m.nodes[n][0])
    b = dof_of[tip]
    assert b >= 0 and abs(vecs[0][b + 1]) > 0.0


def test_modal_all_fixed_returns_empty():
    """A mesh where every node is anchored must return empty (not crash)."""
    m = FemMesh(
        nodes=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        cells=[(0, 1, 2)],
        fixed={0, 1, 2},
        fill=[1.0],
        extra_mass=[],
    )
    E, nu, rho, t = 170.0e9, 0.28, 2330.0, 10.0e-6
    freqs, vecs, dof_of = skfem_solve.modal(m, E, nu, rho, t, n_modes=3)
    assert freqs == [] and vecs == [] and dof_of == {}


def test_modal_m1_normalised():
    """Mode-1 vector, lifted to full P1 vertex space, must satisfy v^T M1 v ≈ 1."""
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 2.0, 10.0, mech="anchored")
    m = mesh_build.build_mesh([body, anchor], h=2.0)
    E, nu, rho, t = 170.0e9, 0.28, 2330.0, 10.0e-6
    freqs, vecs, dof_of = skfem_solve.modal(m, E, nu, rho, t, n_modes=1)

    p1_vec_basis, M1 = _build_p1_mass(m, rho, t)
    vd1 = p1_vec_basis.nodal_dofs
    n_verts = len(m.nodes)
    free_nodes = [ni for ni in range(n_verts) if ni not in m.fixed]

    full_p1 = np.zeros(2 * n_verts)
    raw = vecs[0]
    for idx, n in enumerate(free_nodes):
        full_p1[int(vd1[0, n])] = raw[2 * idx]
        full_p1[int(vd1[1, n])] = raw[2 * idx + 1]

    norm_sq = float(full_p1 @ M1 @ full_p1)
    assert 0.9 < norm_sq < 1.1, f"v^T M1 v = {norm_sq:.4f}, expected ≈ 1.0"


def test_modal_extra_mass_lowers_freq():
    """Adding a large lumped mass at a free node must lower the first frequency."""
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 2.0, 10.0, mech="anchored")
    m_base = mesh_build.build_mesh([body, anchor], h=5.0)
    E, nu, rho, t = 170.0e9, 0.28, 2330.0, 10.0e-6

    freqs_base, _, _ = skfem_solve.modal(m_base, E, nu, rho, t, n_modes=1)

    tip = max(range(len(m_base.nodes)), key=lambda n: m_base.nodes[n][0])
    m_heavy = FemMesh(
        nodes=m_base.nodes,
        cells=m_base.cells,
        fixed=m_base.fixed,
        fill=m_base.fill,
        extra_mass=[(tip, 1.0e7)],   # 1e7 um^2 — very large lumped mass
    )
    freqs_heavy, _, _ = skfem_solve.modal(m_heavy, E, nu, rho, t, n_modes=1)

    assert freqs_heavy[0] < freqs_base[0], (
        f"extra_mass should lower freq: {freqs_heavy[0]/1e6:.4f} MHz vs "
        f"baseline {freqs_base[0]/1e6:.4f} MHz"
    )


def test_static_cantilever_tip_load():
    # clamped beam, line force at the tip; Euler-Bernoulli tip deflection
    # uses per-unit-thickness area moment I = H^3/12 (forces are N/m).
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 2.0, 10.0, mech="anchored")
    m = mesh_build.build_mesh([body, anchor], h=2.0)
    E, nu, t = 170.0e9, 0.28, 10.0e-6
    tip = max(range(len(m.nodes)), key=lambda n: m.nodes[n][0])
    F_line = 1.0  # N/m (per unit thickness)
    disp = skfem_solve.static_solve(m, E, nu, t, forces={tip: (0.0, -F_line)})

    L, H = 100e-6, 10e-6
    I = (H ** 3) / 12.0
    y_analytic = F_line * L ** 3 / (3 * E * I)
    uy = abs(disp[tip][1])
    assert 0.5 * y_analytic < uy < 1.6 * y_analytic
