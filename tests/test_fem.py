from soidlc.fem.result import FemMesh, FemResult
from soidlc import geometry as G
from soidlc.fem import mesh_build


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
