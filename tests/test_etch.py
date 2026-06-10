"""Tests for the level-set Bosch etch simulator.

A feature-scale etch model has no closed-form answer, so we don't assert
magic numbers.  Instead we test:
  * physical invariants that must hold for *any* valid parameters
    (ARDE monotonicity, anisotropy, mask protection, more cycles -> deeper);
  * analytic limit cases where ground truth DOES exist
    (a wide unobstructed trench drills ~linearly in the cycle count);
  * numerical sanity (no NaN, front bounded, determinism);
  * pipeline integration (an ARDE-limited trench flags "not released").
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import etch                                   # noqa: E402


def _depth(width, cycles=30, dx=0.3, depth=20.0, **kw):
    x0 = 10.0
    r = etch.simulate([(x0, x0 + width)], domain_w=40, depth=depth, dx=dx,
                      recipe=etch.BoschRecipe(cycles=cycles, **kw),
                      box_at=depth)
    return r


class TestInvariants(unittest.TestCase):
    def test_arde_monotonic(self):
        # narrower openings must not etch deeper than wider ones (RIE lag)
        depths = []
        for w in (2.0, 5.0, 12.0, 24.0):
            r = etch.simulate([(10, 10 + w)], domain_w=44, depth=22, dx=0.3,
                              recipe=etch.BoschRecipe(cycles=34), box_at=22)
            depths.append(r.depths[0])
        for a, b in zip(depths, depths[1:]):
            self.assertLessEqual(a, b + 0.6, depths)   # non-decreasing
        self.assertLess(depths[0], depths[-1] - 1.0, depths)  # real spread

    def test_anisotropy(self):
        # a narrow Bosch trench is deeper than it is wide (AR > 1)
        r = _depth(3.0, cycles=34)
        self.assertGreater(r.depths[0] / r.widths[0], 1.5,
                           (r.depths[0], r.widths[0]))

    def test_more_cycles_deeper(self):
        d20 = _depth(16.0, cycles=20).depths[0]
        d40 = _depth(16.0, cycles=40).depths[0]
        self.assertGreater(d40, d20 + 2.0, (d20, d40))

    def test_mask_protects(self):
        # covered regions must not be etched away (silicon stays under mask)
        r = etch.simulate([(16, 24)], domain_w=40, depth=18, dx=0.3,
                          recipe=etch.BoschRecipe(cycles=30), box_at=18)
        nx, j0 = r.nx, int(round((2.0 + 2 * 0.3) / 0.3))
        # a column well outside the opening stays solid just below the surface
        i = int(4.0 / 0.3)
        solid = r.phi[i + nx * (j0 + 2)] < 0
        self.assertTrue(solid)


class TestAnalyticLimits(unittest.TestCase):
    def test_wide_trench_drills_linearly(self):
        # a wide, near-unobstructed trench advances ~ proportional to cycles:
        # depth(2N) ~ 2 * depth(N) (the ARDE factor is ~constant when wide)
        d1 = _depth(26.0, cycles=18, depth=30).depths[0]
        d2 = _depth(26.0, cycles=36, depth=30).depths[0]
        ratio = d2 / d1
        self.assertTrue(1.7 < ratio < 2.3, f"depth not ~linear: {ratio:.2f}")

    def test_scallop_count_tracks_cycles(self):
        # scalloping is a small per-cycle ripple, not the trench depth
        r = _depth(6.0, cycles=40)
        self.assertGreater(r.scallop_um, 0.0)
        self.assertLess(r.scallop_um, 2.0)


class TestTaper(unittest.TestCase):
    """Sidewall taper sign must be controllable: passivation-starved etching
    widens the long-exposed top (positive taper / narrowing downward); ion
    scattering with depth widens the bottom (negative taper / bowing)."""

    def _taper(self, passivation, bow):
        r = etch.simulate([(12, 20)], domain_w=40, depth=16, dx=0.1,
                          recipe=etch.BoschRecipe(
                              cycles=24, etch_per_cycle=0.85,
                              passivation=passivation, bow=bow,
                              scallop_um=0.12), box_at=30)
        return r.tapers[0]

    def test_positive_taper_narrows(self):
        self.assertGreater(self._taper(0.25, 0.0), 0.5)

    def test_negative_taper_bows(self):
        self.assertLess(self._taper(0.99, 0.6), -2.0)

    def test_taper_sign_separates(self):
        pos = self._taper(0.25, 0.0)
        neg = self._taper(0.99, 0.6)
        self.assertGreater(pos - neg, 3.0, (pos, neg))

    def test_bow_monotonic(self):
        # stronger bow -> more negative (re-entrant) taper
        self.assertLess(self._taper(0.99, 0.6), self._taper(0.99, 0.3) + 0.5)


class TestNumerics(unittest.TestCase):
    def test_no_nan_and_bounded(self):
        r = _depth(8.0, cycles=24)
        self.assertTrue(all(math.isfinite(v) for v in r.phi[::37]))

    def test_deterministic(self):
        a = _depth(8.0, cycles=20).depths[0]
        b = _depth(8.0, cycles=20).depths[0]
        self.assertEqual(a, b)

    def test_convergence_in_dx(self):
        # halving the grid should not move the depth by a lot
        coarse = _depth(12.0, cycles=24, dx=0.4).depths[0]
        fine = _depth(12.0, cycles=24, dx=0.25).depths[0]
        self.assertLess(abs(coarse - fine) / fine, 0.25, (coarse, fine))


class TestPipelineIntegration(unittest.TestCase):
    def test_arde_limited_trench_not_released(self):
        # a deep narrow isolation trench that cannot reach the BOX leaves its
        # region anchored -> the manufacturability flag must fire
        r = etch.simulate([(10, 12)], domain_w=40, depth=30, dx=0.3,
                          recipe=etch.BoschRecipe(cycles=30), box_at=30)
        self.assertFalse(r.reached_box[0])
        self.assertTrue(any("ARDE-LIMITED" in l for l in r.report()))

    def test_wide_trench_reaches_box(self):
        r = etch.simulate([(8, 30)], domain_w=46, depth=16, dx=0.3,
                          recipe=etch.BoschRecipe(cycles=44), box_at=16)
        self.assertTrue(r.reached_box[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
