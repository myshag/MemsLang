"""Corner compensation / convex-corner undercut tests.

As with the etch modules we assert the established *directions* and geometric
rules, not calibration magic numbers:

  * a convex mesa corner is undercut (lost) by the fast <410> planes;
  * a straight <110> mask edge barely recedes ({111} sidewall);
  * a <110> compensation square preserves the corner once it is larger than
    sqrt(2) * undercut, and fails when it is too small;
  * a deeper etch (more undercut) needs a larger compensation square.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import corner as C                              # noqa: E402


def _mesa(extra=None):
    m = C.Mask().add_rect(40, 40, 200, 200)        # convex corner at (40, 40)
    if extra:
        m.add(extra)
    return m


class TestUndercut(unittest.TestCase):
    def setUp(self):
        self.rec = C.UndercutEtch(depth=18, ratio=1.4, dx=0.6)   # U = 25.2 um

    def test_convex_corner_is_undercut(self):
        r = C.simulate(_mesa(), 240, 240, self.rec)
        self.assertFalse(r.is_silicon(40, 40))     # the sharp corner is gone

    def test_straight_110_edge_holds(self):
        r = C.simulate(_mesa(), 240, 240, self.rec)
        # mid-edge, far from both corners: recedes by << the corner undercut
        rec = r.undercut_along(40, 120, +1, 0, maxd=40)
        self.assertLess(rec, 3.0, rec)

    def test_corner_undercut_scales_with_depth(self):
        shallow = C.simulate(_mesa(), 240, 240,
                             C.UndercutEtch(depth=10, ratio=1.4, dx=0.6))
        deep = C.simulate(_mesa(), 240, 240,
                          C.UndercutEtch(depth=24, ratio=1.4, dx=0.6))
        us = shallow.undercut_along(40, 40, 1, 1, maxd=120)
        ud = deep.undercut_along(40, 40, 1, 1, maxd=120)
        self.assertGreater(ud, us + 4.0, (us, ud))


class TestCompensation(unittest.TestCase):
    def setUp(self):
        self.rec = C.UndercutEtch(depth=18, ratio=1.4, dx=0.6)
        self.U = self.rec.r_fast

    def test_square_above_threshold_protects(self):
        S = 1.5 * (2 ** 0.5) * self.U              # comfortably over sqrt(2)*U
        r = C.simulate(_mesa(C.square_comp((40, 40), S)), 240, 240, self.rec)
        self.assertTrue(r.is_silicon(40, 40))

    def test_square_below_threshold_fails(self):
        S = 0.5 * (2 ** 0.5) * self.U              # too small to reach the end
        r = C.simulate(_mesa(C.square_comp((40, 40), S)), 240, 240, self.rec)
        self.assertFalse(r.is_silicon(40, 40))

    def test_deeper_etch_needs_bigger_square(self):
        deep = C.UndercutEtch(depth=26, ratio=1.4, dx=0.6)   # bigger undercut
        S = 1.2 * (2 ** 0.5) * self.U              # sized for the shallow etch
        ok_shallow = C.simulate(_mesa(C.square_comp((40, 40), S)),
                                240, 240, self.rec).is_silicon(40, 40)
        fails_deep = C.simulate(_mesa(C.square_comp((40, 40), S)),
                                240, 240, deep).is_silicon(40, 40)
        self.assertTrue(ok_shallow)
        self.assertFalse(fails_deep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
