import math

from soidlc.fem.result import FemMesh, FemResult
from soidlc import geometry as G
from soidlc.fem import mesh_build
from soidlc.fem import skfem_solve


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
