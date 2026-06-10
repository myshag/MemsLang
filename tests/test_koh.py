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
