"""Tests for the flux-driven Bosch model (drie.c through etch.simulate with
``fluxes``).

Ground truths that exist for any parameters: an open flat surface etches at
the plasma's own rate, a passivation step leaves the film its fluxes deposit,
the etch step cannot etch before the ions have worn that film through, the
mask recedes at its own rate; and the transport's physics: a narrow trench
starves of F (RIE lag) and keeps less of the film at its bottom.  The
reference plasmas are Ferma's (tests/data, trimmed ferma_icp summaries).
"""
import json
import math
import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import etch, plasma                           # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _summary(step):
    with open(os.path.join(_DATA, step + "_icp.json")) as f:
        return json.load(f)


def _reference(**kw):
    """The reference recipe's fluxes (Ferma: 600 W, 30 mTorr, 130 sccm SF6
    with 15 W of bias, 100 sccm C4F8)."""
    return etch.BoschFluxes.from_plasmas(
        plasma.etch_plasma(_summary("bosch_etch")),
        plasma.passivation_plasma(_summary("bosch_passivation")), **kw)


def _surface_at(r, x):
    """Depth (um, below the wafer's original top) of the front in the column
    at x, interpolated between cells; mask_thick and dx as in _etch."""
    i = int(x / r.dx)
    col = [r.phi[i + r.nx * j] for j in range(r.nz)]
    j0 = int(round((_MASK + 2 * r.dx) / r.dx))
    for j in range(r.nz - 1):
        if col[j] > 0 >= col[j + 1]:
            z = (j + 0.5 + col[j] / (col[j] - col[j + 1])) * r.dx
            return z - (j0 + 0.5) * r.dx
    return float("nan")


_MASK = 0.1    # um: thin enough not to shadow a wide opening


def _etch(fluxes, openings=((5.0, 45.0),), width=50.0, depth=4.0, dx=0.05,
          mask=_MASK):
    return etch.simulate(list(openings), domain_w=width, depth=depth, dx=dx,
                         fluxes=fluxes, box_at=depth, mask_thick=mask)


class TestFermaConsistency(unittest.TestCase):
    def test_open_wafer_etches_at_the_plasmas_rate(self):
        e = plasma.etch_plasma(_summary("bosch_etch"))
        f = _reference()
        self.assertAlmostEqual(f.flat_etch_rate() / e.etch_rate, 1.0, places=9)
        # the F flux the wafer's coverage implies is below the reported
        # chamber flux: the etching depletes F at the wafer
        self.assertLess(f.f_flux, e.f_flux)
        self.assertGreater(f.f_flux, 0.5 * e.f_flux)

    def test_flat_surface_without_film(self):
        f = _reference(t_pass=0.0, t_etch=3.0, cycles=1)
        r = _etch(f)
        self.assertAlmostEqual(r.diagnostics["etch_rate"] / f.flat_etch_rate(),
                               1.0, places=9)
        d = _surface_at(r, 25.0)
        self.assertAlmostEqual(d / (f.flat_etch_rate() * 3.0 * 1e6), 1.0,
                               delta=0.05)


class TestFilm(unittest.TestCase):
    def test_film_per_step(self):
        f = _reference(t_etch=0.5, cycles=1)
        r = _etch(f)
        self.assertAlmostEqual(r.diagnostics["film"] / f.flat_film(), 1.0,
                               delta=0.03)

    def test_no_etch_before_breakthrough(self):
        # the reference recipe as published: the film the passivation step
        # leaves (~87 nm) takes the ions ~14 s, twice the etch step
        f = _reference(cycles=1)
        self.assertGreater(f.breakthrough(), 10.0)
        r = _etch(f)
        self.assertLess(abs(_surface_at(r, 25.0)), 0.01)

    def test_etch_after_breakthrough(self):
        f = _reference(cycles=1).calibrated(2.0)
        self.assertAlmostEqual(f.breakthrough(), 2.0, places=9)
        r = _etch(f)
        expect = f.flat_etch_rate() * (f.t_etch - 2.0) * 1e6
        self.assertAlmostEqual(_surface_at(r, 25.0) / expect, 1.0, delta=0.08)
        self.assertAlmostEqual(r.diagnostics["clear_time"], 2.0, delta=0.4)


class TestTransport(unittest.TestCase):
    def test_rie_lag(self):
        # without the film F limits the etch: a 1 um trench etches slower
        # than a 10 um one once it is a few widths deep
        f = _reference(cycles=12, t_etch=7.0, t_pass=0.0)
        r = _etch(f, openings=((4.0, 5.0), (12.0, 22.0)), width=26.0,
                  depth=6.0, mask=1.0)
        narrow, wide = r.depths
        self.assertGreater(narrow, 1.0)
        self.assertLess(narrow, 0.9 * wide, r.depths)

    def test_less_film_at_the_bottom_of_a_narrow_trench(self):
        # the CFx come from above: a slot's bottom sees less of the plasma
        f = _reference(cycles=1, t_etch=0.5)
        wide = _etch(f, openings=((5.0, 45.0),), mask=2.0)
        narrow = _etch(f, openings=((24.5, 25.5),), mask=2.0)
        self.assertLess(narrow.diagnostics["film"],
                        0.8 * wide.diagnostics["film"])

    def test_mask_recedes_at_its_rate(self):
        f = _reference(t_pass=0.0, t_etch=100.0, cycles=1)
        r = _etch(f, mask=1.0, depth=8.0)
        s = f.surface
        y = s.alpha_mask * (math.sqrt(f.ion_energy) - math.sqrt(s.e_th_ion)) / (
            math.sqrt(s.e_ref) - math.sqrt(s.e_th_ion))
        rate = y * f.ion_flux + s.beta_mask * f.f_flux        # m/s
        # the mask's top, at x = 2 um (covered), moved down by rate * t
        i = int(2.0 / r.dx)
        col = [r.phi[i + r.nx * j] for j in range(r.nz)]
        z = next((j + 0.5 + col[j] / (col[j] - col[j + 1])) * r.dx
                 for j in range(r.nz - 1) if col[j] > 0 >= col[j + 1])
        top0 = (int(round((1.0 + 2 * r.dx) / r.dx)) - int(round(1.0 / r.dx))
                + 0.5) * r.dx
        self.assertAlmostEqual((z - top0) / (rate * 100.0 * 1e6), 1.0, delta=0.15)

    def test_deterministic(self):
        f = _reference(cycles=2).calibrated(1.0)
        a = _etch(f, openings=((4.0, 6.0),), width=10.0)
        b = _etch(f, openings=((4.0, 6.0),), width=10.0)
        self.assertEqual(a.phi, b.phi)

    def test_report(self):
        r = _etch(_reference(cycles=1).calibrated(1.0))
        self.assertTrue(any("from the plasma's fluxes" in s for s in r.report()))


class TestCalibration(unittest.TestCase):
    def test_breakthrough_scales_the_deposition_only(self):
        f = _reference()
        g = f.calibrated(1.5)
        k = g.surface.beta_dep / f.surface.beta_dep
        self.assertAlmostEqual(g.surface.alpha_dep / f.surface.alpha_dep, k)
        self.assertEqual(replace(g.surface, beta_dep=f.surface.beta_dep,
                                 alpha_dep=f.surface.alpha_dep), f.surface)
        self.assertAlmostEqual(g.breakthrough(), 1.5, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
