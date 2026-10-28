"""Tests for the plasma physics of a Bosch recipe (soidlc.plasma) and the
recipe physics driven by it.

The summaries in tests/data are ferma_icp's for the reference recipe
(soidlc/reactors, trimmed to what soidlc reads).  The recipe physics is
checked with a stand-in plasma whose fluxes and energies move with the recipe
the way the discharge's do, so the mapping is tested without the solver; the
solver itself runs only with MEMSLANG_FERMA_TESTS=1 and the binaries in FERMA
and FERMA_ICP (a few minutes per recipe).
"""
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import plasma                                  # noqa: E402
from soidlc.recipe_physics import BoschProcess             # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _summary(step):
    with open(os.path.join(_DATA, step + "_icp.json")) as f:
        return json.load(f)


class TestSummaries(unittest.TestCase):
    def test_etch_step(self):
        e = plasma.etch_plasma(_summary("bosch_etch"))
        self.assertAlmostEqual(e.etch_rate, 3.107846613e-08)
        self.assertAlmostEqual(e.ion_share, 0.7165886276)
        self.assertAlmostEqual(e.f_flux / 1.847663788e+22, 1.0, places=12)
        # SF5+, SF4+, SF3+, F2+
        self.assertAlmostEqual(e.ion_flux / 3.7e19, 1.0, delta=0.05)
        # the flux-weighted mean of the four ions' energies (124 - 125 eV)
        self.assertGreater(e.ion_energy, 123.9)
        self.assertLess(e.ion_energy, 124.3)

    def test_passivation_step(self):
        d = plasma.passivation_plasma(_summary("bosch_passivation"))
        s = {x["name"]: x["flux"] for x in _summary("bosch_passivation")["species"]
             if "flux" in x}  # F(s): a coverage of the walls
        self.assertAlmostEqual(d.film_flux / (s["CF"] + s["CF2"] + s["CF3"]), 1.0,
                               places=12)
        self.assertGreater(d.ion_flux, 0.0)

    def test_failures_reported(self):
        bad = _summary("bosch_etch")
        bad["converged"] = False
        with self.assertRaises(RuntimeError):
            plasma.etch_plasma(bad)
        unbiased = _summary("bosch_etch")
        del unbiased["bias"]
        with self.assertRaises(RuntimeError):
            plasma.etch_plasma(unbiased)
        # the passivation reactor's wafer has no etch chemistry
        with self.assertRaises(RuntimeError):
            plasma.etch_plasma(_summary("bosch_passivation"))


class _StandIn:
    """A plasma whose wafer fluxes and energies follow the recipe like the
    discharge's: fluxes with the coil power and the flow, the ions' energy
    with the square root of the bias power, the ion-driven share of the
    etching with that energy, and the F flux down with pressure (more
    collisions, more loss to the walls)."""

    def __init__(self):
        self.calls = 0

    def etch(self, coil_w, pressure_mtorr, sf6_sccm, bias_w):
        self.calls += 1
        energy = 124.0 * math.sqrt(bias_w / 15.0)
        share = energy / (energy + 50.0) * (30.0 / pressure_mtorr) ** 0.3
        share = min(0.95, share)
        f_flux = 1.8e22 * (coil_w / 600.0) * (sf6_sccm / 130.0) ** 0.5
        ion_flux = 3.7e19 * coil_w / 600.0
        rate = 3e-8 * (f_flux / 1.8e22) * (1 + ion_flux / 3.7e19 * energy / 124)
        return plasma.EtchPlasma(etch_rate=rate, ion_share=share, f_flux=f_flux,
                                 ion_flux=ion_flux, ion_energy=energy,
                                 ne=7e16, te=3.4)

    def passivation(self, coil_w, pressure_mtorr, c4f8_sccm):
        self.calls += 1
        return plasma.PassivationPlasma(
            film_flux=5e22 * (coil_w / 600.0) * (c4f8_sccm / 100.0),
            ion_flux=2e19 * coil_w / 600.0, ne=1e17, te=3.0)


class TestRecipeFromPlasma(unittest.TestCase):
    def _rec(self, **kw):
        return BoschProcess(plasma=_StandIn(), **kw).to_recipe()[0]

    def test_reference_matches_power_laws(self):
        # every drive term is relative to the reference recipe, so the
        # reference maps onto the same knobs with or without the plasma
        a, wa = BoschProcess(plasma=_StandIn()).to_recipe()
        b, wb = BoschProcess().to_recipe()
        for k in ("etch_per_cycle", "r_iso", "passivation", "scallop_um",
                  "selectivity", "footing", "bow"):
            self.assertAlmostEqual(getattr(a, k), getattr(b, k), places=12)
        self.assertEqual(wa, wb)

    def test_bias_deepens_and_lowers_selectivity(self):
        lo = self._rec(bias_w=10)
        hi = self._rec(bias_w=40)
        self.assertGreater(hi.etch_per_cycle, lo.etch_per_cycle)
        self.assertLess(hi.selectivity, lo.selectivity)
        self.assertGreater(hi.footing, lo.footing)
        self.assertLess(hi.r_iso, lo.r_iso)

    def test_passivation_balance(self):
        lo = self._rec(c4f8=60, t_pass=2)
        hi = self._rec(c4f8=160, t_pass=9)
        self.assertGreater(hi.passivation, lo.passivation)

    def test_pressure_raises_isotropy(self):
        lo = self._rec(pressure=20)
        hi = self._rec(pressure=80)
        self.assertGreater(hi.r_iso, lo.r_iso)
        self.assertGreater(hi.bow, lo.bow)

    def test_summary_reports_the_plasmas(self):
        lines = BoschProcess(plasma=_StandIn()).summary()
        self.assertTrue(any(s.startswith("plasma: etch step") for s in lines))
        self.assertTrue(any(s.startswith("plasma: passivation") for s in lines))


class TestCache(unittest.TestCase):
    def test_memo_and_disk(self):
        with tempfile.TemporaryDirectory() as cache:
            runs = []

            def solve(reactor, params):
                runs.append(params)
                return _summary("bosch_etch")

            p = plasma.FermaPlasma(ferma="none", ferma_icp="none",
                                   cache_dir=cache)
            p._solve = solve
            a = p.etch(600, 30, 130, 15)
            b = p.etch(600, 30, 130, 15)
            self.assertEqual(a, b)
            self.assertEqual(len(runs), 1)              # memo
            q = plasma.FermaPlasma(ferma="none", ferma_icp="none",
                                   cache_dir=cache)
            q._solve = solve
            self.assertEqual(q.etch(600, 30, 130, 15), a)
            self.assertEqual(len(runs), 1)              # disk
            q.etch(600, 30, 130, 20)
            self.assertEqual(len(runs), 2)              # a new recipe

    def test_missing_binaries(self):
        p = plasma.FermaPlasma(ferma="/nonexistent/ferma",
                               ferma_icp="/nonexistent/ferma_icp")
        self.assertFalse(p.available())
        with self.assertRaises(RuntimeError):
            p.etch(600, 30, 130, 15)


@unittest.skipUnless(os.environ.get("MEMSLANG_FERMA_TESTS") == "1"
                     and plasma.FermaPlasma().available(),
                     "set MEMSLANG_FERMA_TESTS=1 and FERMA, FERMA_ICP to run "
                     "Ferma's ICP solver (minutes)")
class TestFermaSolver(unittest.TestCase):
    def test_bias_raises_ion_energy_and_etch_rate(self):
        p = plasma.FermaPlasma(cache_dir=os.environ.get("MEMSLANG_FERMA_CACHE"))
        ref = p.etch(600, 30, 130, 15)
        hot = p.etch(600, 30, 130, 40)
        self.assertGreater(hot.ion_energy, ref.ion_energy)
        self.assertGreater(hot.etch_rate, ref.etch_rate)
        self.assertGreater(hot.ion_share, ref.ion_share)
        d = p.passivation(600, 30, 100)
        self.assertGreater(d.film_flux, 0.0)
        r, _ = BoschProcess(plasma=p, bias_w=40).to_recipe()
        r0, _ = BoschProcess(plasma=p).to_recipe()
        self.assertGreater(r.etch_per_cycle, r0.etch_per_cycle)
        self.assertLess(r.selectivity, r0.selectivity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
