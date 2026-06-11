from soidlc.fem.result import FemMesh, FemResult


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
