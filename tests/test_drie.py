"""Tests for the flux-driven Bosch model (drie.c through etch.simulate with
``fluxes``).

Ground truths that exist for any parameters: an open flat surface etches at
the plasma's own rate, a passivation step leaves the film its fluxes deposit,
the etch step cannot etch before the ions have worn that film through, the
mask recedes at its own rate; and the transport's physics: a narrow trench
starves of F (RIE lag) and keeps less of the film at its bottom.  The
reference is the published recipe of Zhang et al., Micromachines 14, 846
(2023) (recipe_physics.ZHANG2023): its plasmas are Ferma's for an Oxford
ICP180-like reactor with 1.3 % of the wafer open (tests/data, trimmed
ferma_icp summaries), and its measured depth checks the model with its one
fitted constant. ViennaPS's Bosch example, run in ViennaPS on the same
parameters, checks it code to code.
"""
import json
import math
import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import etch, plasma                           # noqa: E402
from soidlc.recipe_physics import ZHANG2023, ZHANG2023_MEASURED  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _summary(step):
    with open(os.path.join(_DATA, step + "_icp.json")) as f:
        return json.load(f)


def _reference(**kw):
    """The fluxes of Zhang et al.'s recipe (Ferma: 700 W; 40 mTorr, 100 sccm
    SF6, 25 W of bias, 8 s; 30 mTorr, 100 sccm C4F8, 10 s)."""
    kw = {"t_etch": ZHANG2023["t_etch"], "t_pass": ZHANG2023["t_pass"], **kw}
    return etch.BoschFluxes.from_plasmas(
        plasma.etch_plasma(_summary("icp180_etch")),
        plasma.passivation_plasma(_summary("icp180_passivation")), **kw)


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
        e = plasma.etch_plasma(_summary("icp180_etch"))
        f = _reference()
        self.assertAlmostEqual(f.flat_etch_rate() / e.etch_rate, 1.0, places=9)
        # the F flux the wafer's coverage implies is the transport model's at
        # the wafer, not the chamber's reported one; of the same order
        self.assertGreater(f.f_flux, 0.5 * e.f_flux)
        self.assertLess(f.f_flux, 3.0 * e.f_flux)

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
        # with the published film growth (Kokkoris et al.) the passivation
        # step leaves ~190 nm, which takes the ions ~17 s: twice the 8 s etch
        # step, and nothing etches
        f = _reference(cycles=1)
        self.assertGreater(f.breakthrough(), 2 * f.t_etch)
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
        # than a 10 um one once it is a few widths deep (with this recipe's
        # F-rich plasma, by some 6 %)
        f = _reference(cycles=12, t_etch=7.0, t_pass=0.0)
        r = _etch(f, openings=((4.0, 5.0), (12.0, 22.0)), width=26.0,
                  depth=7.0, mask=1.0)
        narrow, wide = r.depths
        self.assertGreater(narrow, 1.0)
        self.assertLess(narrow, 0.97 * wide, r.depths)

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


class TestZhang2023(unittest.TestCase):
    """The published recipe against its measurement: 100 cycles etch a 6 um
    trench under 6 um of AZ4620 to 30.72 um.  The film's growth is the one
    fitted constant: a breakthrough of 5.9 s.  The scallops (70 - 80 nm
    measured) need a finer grid than a test can afford (see the README)."""

    def test_depth_after_100_cycles(self):
        m = ZHANG2023_MEASURED
        f = _reference(cycles=m["cycles"]).calibrated(5.9)
        r = etch.simulate([(10.0, 16.0)], domain_w=26.0, depth=40.0, dx=0.1,
                          fluxes=f, box_at=40.0, mask_thick=6.0)
        self.assertAlmostEqual(r.depths[0] / m["depth_um"], 1.0, delta=0.03)
        self.assertAlmostEqual(r.widths[0], m["width_um"], delta=0.3)

    def test_open_silicon_etches_faster_than_a_bare_wafer(self):
        # the plasma's own numbers: 1.3 % of the wafer open
        e = plasma.etch_plasma(_summary("icp180_etch"))
        self.assertAlmostEqual(_summary("icp180_etch")["wafer"]["open_fraction"],
                               0.013)
        self.assertAlmostEqual(e.etch_rate * 6e7, 6.91, delta=0.05)   # um/min


# ViennaPS's Bosch example (examples/boschProcess/configBoschRayTracing.txt:
# Ertl & Selberherr, Microelectron. Eng. 87, 20 (2010), Kokkoris et al.'s
# volumes per particle) in 2D: a 2 um trench under 0.6 um of mask in a 3.5 um
# cell, 1e19 F and 4.375e15 ions per cm^2 and s in the etch step, 2e18 CFx and
# 3.125e15 ions in the passivation step; silicon's linear law (270 A^3 per
# ion, 0.9 A^3 per F arriving, F sticking 0.2), ions from cos^1000.
_ES_CELL, _ES_OPENING, _ES_MASK = 3.5, (0.75, 2.75), 0.6


def _es_fluxes(**kw):
    a3 = 1e-30
    s = etch.SurfaceKinetics(
        f_sticking=0.2, si_alpha=270 * a3, si_beta=0.9 * a3,
        alpha_poly=125 * a3, beta_poly=0.03 * a3, alpha_mask=13.5 * a3,
        beta_mask=0.045 * a3, alpha_dep=10 * a3, beta_dep=0.5 * a3,
        stick_dep=0.1, stick_poly=0.1, stick_mask=0.2, e_ref=100.0,
        ion_sigma_deg=math.degrees(1 / math.sqrt(1000)))
    kw = {"f_flux": 1e23, "ion_flux": 4.375e19, "ion_energy": 100.0,
          "film_flux": 2e22, "pass_ion_flux": 3.125e19, "t_etch": 11.0,
          "t_pass": 5.0, "cycles": 10, "surface": s, **kw}
    return etch.BoschFluxes(**kw)


def _es_heights(r):
    """The front's height (um above the wafer's original top) in a column at
    x, and the left wall's distance from the opening's centre at height y."""
    j0 = int(round((_ES_MASK + 2 * r.dx) / r.dx))
    p = lambda i, j: r.phi[i + r.nx * j]

    def height(i):
        for j in range(r.nz - 1):
            a, b = p(i, j), p(i, j + 1)
            if a > 0 >= b:
                return (j0 - j - a / (a - b)) * r.dx

    def half_width(y):
        jf = j0 - y / r.dx
        j, t = int(jf), jf - int(jf)
        row = [p(i, j) * (1 - t) + p(i, j + 1) * t for i in range(r.nx)]
        for i in range(r.nx // 2, 0, -1):
            a, b = row[i], row[i - 1]
            if a > 0 >= b:
                return r.nx * r.dx / 2 - ((i + 0.5) - a / (a - b)) * r.dx
    return height, half_width


class TestAgainstViennaPS(unittest.TestCase):
    """Against ViennaPS 4.7.0 (MultiParticleProcess, 2D, 25 nm grid, run on
    the same parameters): without the film the two codes' transport (F
    re-emitted with sticking 0.2, the ions' shadowing) and the undercut below
    the mask agree; over the Bosch cycles the depth and the mask's wear do,
    while ViennaPS's trench stays narrower (its film is a layer with a
    thickness, whose foot on the mask's wall keeps the silicon below it until
    the undercut reaches it; here the film is a thickness per face)."""

    def test_etch_without_film(self):
        # 6 s of the etch step only: ViennaPS 0.563 um deep, the wall
        # 1.281, 1.208, 1.052 um from the centre 0.1, 0.3, 0.45 um down
        r = etch.simulate([_ES_OPENING], _ES_CELL, 2.0, dx=0.025,
                          mask_thick=_ES_MASK, box=False,
                          fluxes=_es_fluxes(t_etch=6.0, t_pass=0.0, cycles=1,
                                            film_flux=0.0, pass_ion_flux=0.0))
        height, half_width = _es_heights(r)
        self.assertAlmostEqual(-height(r.nx // 2), 0.563, delta=0.02)
        for y, w in ((-0.1, 1.281), (-0.3, 1.208), (-0.45, 1.052)):
            self.assertAlmostEqual(half_width(y), w, delta=0.02)

    def test_bosch_depth_and_mask(self):
        # 10 cycles: ViennaPS 4.458 um deep, 0.350 um of mask left
        r = etch.simulate([_ES_OPENING], _ES_CELL, 6.0, dx=0.1,
                          mask_thick=_ES_MASK, box=False, fluxes=_es_fluxes())
        height, _ = _es_heights(r)
        self.assertAlmostEqual(-height(r.nx // 2) / 4.458, 1.0, delta=0.05)
        self.assertAlmostEqual(height(0), 0.350, delta=0.01)

    def test_linear_law_on_the_open_wafer(self):
        f = _es_fluxes()
        r = etch.simulate([(0.0, 3.5)], 3.5, 1.0, dx=0.1, mask_thick=0.1,
                          box=False, fluxes=replace(f, cycles=1))
        rate = (270e-30 * 4.375e19 + 0.9e-30 * 1e23)
        self.assertAlmostEqual(r.diagnostics["etch_rate"] / rate, 1.0,
                               places=9)


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
