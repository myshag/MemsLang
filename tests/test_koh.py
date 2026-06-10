import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import koh

class TestKOH(unittest.TestCase):
    def _rec(self, **kw):
        d = dict(r100=1.0, r110=1.0, r111=0.008, notch_w=8)
        d.update(kw); return koh.WetEtch(**d)

    def test_111_facet_angle(self):
        # walls settle near the {111}/{100} angle of 54.74 deg
        r = koh.simulate([(16, 44)], 60, 26, dx=0.2, recipe=self._rec(steps=200))
        self.assertTrue(45 < r.facet_angle < 60, r.facet_angle)

    def test_narrow_opening_self_terminates(self):
        r = koh.simulate([(26, 34)], 60, 26, dx=0.2, recipe=self._rec(steps=320))
        self.assertTrue(r.self_terminated)

    def test_wider_opening_deeper_before_pinchoff(self):
        n = koh.simulate([(27, 33)], 60, 30, dx=0.2, recipe=self._rec(steps=260))
        w = koh.simulate([(18, 42)], 60, 30, dx=0.2, recipe=self._rec(steps=260))
        # a wider mask reaches a deeper apex before the {111} facets meet
        self.assertGreater(w.depth, n.depth)

if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestKOH3D(unittest.TestCase):
    def _pit(self, steps=120, dx=0.6):
        return koh.simulate_3d(lambda x, y: 12 <= x <= 48 and 12 <= y <= 48,
                               60, 60, 28, dx=dx,
                               recipe=koh.WetEtch(r100=1.0, r111=0.01,
                                                  notch_w=9, steps=steps))

    def _depth(self, res, x, y):
        nx, ny, nz, dx, j0 = res.nx, res.ny, res.nz, res.dx, res.j0
        ix, jy = int(x / dx), int(y / dx)
        kd = j0
        for k in range(j0, nz):
            if res.phi[ix + nx * (jy + ny * k)] > 0:
                kd = k
        return (kd - j0) * dx

    def test_cavity_etched_and_mask_protects(self):
        r = self._pit()
        self.assertGreater(self._depth(r, 30, 30), 5.0)   # opening etched
        self.assertLess(self._depth(r, 4, 4), 1.0)        # masked corner kept

    def test_four_fold_symmetry(self):
        r = self._pit()
        d = [self._depth(r, x, y) for x, y in
             ((30, 18), (30, 42), (18, 30), (42, 30))]
        self.assertLess(max(d) - min(d), 1.5 * r.dx, d)

    def test_heightmap_mesh_nonempty(self):
        r = self._pit(steps=80)
        m = koh.heightmap_mesh(r)
        self.assertGreater(len(m.triangles), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
