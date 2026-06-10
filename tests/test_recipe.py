"""Tests for the process-physics layer (recipe params -> effective knobs).

Like the etch tests, we don't assert magic numbers (those need calibration);
we assert the reference anchor and the *direction* of every dependency, which
is what is physically established.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc.etch import BoschRecipe                        # noqa: E402
from soidlc.recipe_physics import BoschProcess             # noqa: E402


class TestReferenceAnchor(unittest.TestCase):
    def test_reference_matches_defaults(self):
        # the reference recipe must reproduce the BoschRecipe defaults
        r, w = BoschProcess().to_recipe()
        d = BoschRecipe()
        self.assertAlmostEqual(r.etch_per_cycle, 0.80, places=2)
        self.assertAlmostEqual(r.r_iso, d.r_iso, places=3)
        self.assertAlmostEqual(r.passivation, 0.90, places=2)
        self.assertAlmostEqual(r.scallop_um, d.scallop_um, places=3)
        self.assertAlmostEqual(r.selectivity, d.selectivity, places=0)
        self.assertEqual(w, [])


class TestDependencyDirections(unittest.TestCase):
    def _rec(self, **kw):
        return BoschProcess(**kw).to_recipe()[0]

    def test_bias_deepens_and_lowers_selectivity(self):
        lo = self._rec(bias_w=10)
        hi = self._rec(bias_w=40)
        self.assertGreater(hi.etch_per_cycle, lo.etch_per_cycle)   # deeper
        self.assertLess(hi.selectivity, lo.selectivity)            # erodes mask
        self.assertGreater(hi.footing, lo.footing)                 # notching
        self.assertLess(hi.r_iso, lo.r_iso)                        # anisotropic

    def test_pressure_raises_isotropy_and_bowing(self):
        lo = self._rec(pressure=20)
        hi = self._rec(pressure=80)
        self.assertGreater(hi.r_iso, lo.r_iso)
        self.assertGreater(hi.bow, lo.bow)

    def test_passivation_balance(self):
        # more C4F8 / longer passivation step -> higher passivation fraction
        lo = self._rec(c4f8=60, t_pass=2)
        hi = self._rec(c4f8=160, t_pass=9)
        self.assertGreater(hi.passivation, lo.passivation)

    def test_etch_step_sets_per_cycle_and_scallop(self):
        short = self._rec(t_etch=4)
        long = self._rec(t_etch=10)
        self.assertGreater(long.etch_per_cycle, short.etch_per_cycle)
        self.assertGreater(long.scallop_um, short.scallop_um)


class TestWarnings(unittest.TestCase):
    def test_grass_risk_flagged(self):
        _, w = BoschProcess(c4f8=200, t_pass=14).to_recipe()
        self.assertTrue(any("grass" in m for m in w), w)

    def test_low_selectivity_flagged(self):
        _, w = BoschProcess(bias_w=200).to_recipe()
        self.assertTrue(any("selectivity" in m for m in w), w)

    def test_reference_has_no_warnings(self):
        _, w = BoschProcess().to_recipe()
        self.assertEqual(w, [])


class TestEndToEnd(unittest.TestCase):
    def test_recipe_drives_etch(self):
        from soidlc import etch
        p = BoschProcess(bias_w=30, cycles=24)
        r = etch.simulate([(10, 18)], domain_w=36, depth=18, dx=0.2,
                          process=p, box_at=26)
        self.assertGreater(r.depths[0], 5.0)

    def test_higher_bias_etches_deeper_end_to_end(self):
        from soidlc import etch
        lo = etch.simulate([(10, 18)], domain_w=36, depth=24, dx=0.2,
                           process=BoschProcess(bias_w=12, cycles=20),
                           box_at=28)
        hi = etch.simulate([(10, 18)], domain_w=36, depth=24, dx=0.2,
                           process=BoschProcess(bias_w=30, cycles=20),
                           box_at=28)
        self.assertGreater(hi.depths[0], lo.depths[0] + 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
