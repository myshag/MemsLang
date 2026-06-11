"""Calibrated crystal etch-rate diagram tests.

We assert the crystallographic structure that is established physics -- plane
ordering, cubic symmetry, that the reconstruction reproduces its anchors, that
the sampled table matches the diagram, and that driving the 3D level-set from
the table still yields the right geometry (a self-terminating (100) pyramid;
deeper etching on a (110) wafer).
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import crystal_rates as CR                      # noqa: E402


class TestDiagram(unittest.TestCase):
    def setUp(self):
        self.d = CR.RateDiagram.from_condition("KOH_30_70")

    def test_plane_ordering(self):
        r100 = self.d.rate((1, 0, 0))
        r110 = self.d.rate((1, 1, 0))
        r111 = self.d.rate((1, 1, 1))
        self.assertGreater(r110, r100)            # {110} fastest low-index
        self.assertGreater(r100, 20 * r111)       # {111} strongly slow

    def test_anchor_reproduced(self):
        # the RBF reconstruction passes ~through the measured anchors
        self.assertAlmostEqual(self.d.rate((1, 0, 0)), 0.80, delta=0.02)
        self.assertAlmostEqual(self.d.rate((1, 1, 0)), 1.40, delta=0.03)
        self.assertLess(self.d.rate((1, 1, 1)), 0.02)   # ~0.0104

    def test_cubic_symmetry(self):
        # every member of a symmetry family has the same rate
        for hkl in ((1, 1, 1), (1, 1, 0), (3, 1, 1)):
            rs = [self.d.rate(v) for v in CR._family(hkl)]
            self.assertLess(max(rs) - min(rs), 1e-9, (hkl, rs))

    def test_family_sizes(self):
        self.assertEqual(len(CR._family((1, 0, 0))), 6)
        self.assertEqual(len(CR._family((1, 1, 1))), 8)
        self.assertEqual(len(CR._family((1, 1, 0))), 12)


class TestTable(unittest.TestCase):
    def test_table_matches_diagram_on_100(self):
        d = CR.RateDiagram.from_condition("KOH_30_70")
        ntheta, nphi = 90, 180
        tab, rmax = CR.sample_table(d, "100", ntheta, nphi)
        # pick the {111} direction: theta=54.74 deg, phi=45 deg, look it up
        th = math.acos(1 / math.sqrt(3))
        ph = math.radians(45.0)
        it = round(th / math.pi * (ntheta - 1))
        ip = round(ph / (2 * math.pi) * nphi) % nphi
        got = tab[it * nphi + ip] * rmax
        self.assertAlmostEqual(got, d.rate((1, 1, 1)), delta=0.05 * rmax)

    def test_table_normalised(self):
        d = CR.RateDiagram.from_condition("KOH_30_70")
        tab, rmax = CR.sample_table(d, "100")
        self.assertAlmostEqual(max(tab), 1.0, places=6)
        self.assertAlmostEqual(rmax, d.rmax(), delta=0.02)


class TestTableDrivenEtch(unittest.TestCase):
    def _depth(self, res, x, y):
        nx, ny, nz, dx, j0 = res.nx, res.ny, res.nz, res.dx, res.j0
        ix, jy = int(x / dx), int(y / dx)
        kd = j0
        for k in range(j0, nz):
            if res.phi[ix + nx * (jy + ny * k)] > 0:
                kd = k
        return (kd - j0) * dx

    def test_100_pit_forms_and_mask_protects(self):
        from soidlc import koh
        d = CR.RateDiagram.from_condition("KOH_30_70")
        r = koh.simulate_3d(lambda x, y: 12 <= x <= 48 and 12 <= y <= 48,
                            60, 60, 28, dx=0.6, diagram=d, orientation="100",
                            recipe=koh.WetEtch(steps=120))
        self.assertGreater(self._depth(r, 30, 30), 5.0)     # opening etched
        self.assertLess(self._depth(r, 4, 4), 1.0)          # mask holds
        ds = [self._depth(r, x, y) for x, y in
              ((30, 18), (30, 42), (18, 30), (42, 30))]
        self.assertLess(max(ds) - min(ds), 1.5 * r.dx, ds)  # 4-fold symmetric

    def test_110_wafer_etches_deeper_than_100(self):
        from soidlc import koh
        d = CR.RateDiagram.from_condition("KOH_30_70")
        kw = dict(dx=0.7, diagram=d, recipe=koh.WetEtch(steps=140))
        slot = lambda x, y: 20 <= x <= 40
        r100 = koh.simulate_3d(slot, 60, 60, 40, orientation="100", **kw)
        r110 = koh.simulate_3d(slot, 60, 60, 40, orientation="110", **kw)
        self.assertGreater(self._depth(r110, 30, 30),
                           self._depth(r100, 30, 30) + 1.0)


def _angle(u, v):
    d = sum(a * b for a, b in zip(u, v))
    d = 1.0 if d > 1 else (-1.0 if d < -1 else d)
    return math.degrees(math.acos(d))


class TestMisalignAndMiscut(unittest.TestCase):
    def test_misalign_spins_inplane_keeps_normal(self):
        X0, Y0, Z0 = CR.wafer_basis("100")
        X, Y, Z = CR.wafer_basis("100", misalign_deg=20.0)
        self.assertLess(_angle(Z, Z0), 1e-6)          # surface normal unchanged
        self.assertAlmostEqual(_angle(X, X0), 20.0, delta=1e-6)   # mask spun

    def test_miscut_tilts_normal_off_pole(self):
        _, _, Z0 = CR.wafer_basis("100")
        _, _, Z = CR.wafer_basis("100", miscut_deg=8.0)
        self.assertAlmostEqual(_angle(Z, Z0), 8.0, delta=1e-6)
        self.assertAlmostEqual(_angle(Z, (0, 0, 1)), 8.0, delta=1e-6)

    def _floor_width(self, res, y):
        nx, ny, nz, dx, j0 = res.nx, res.ny, res.nz, res.dx, res.j0
        jy = int(y / dx)
        kd = j0
        for k in range(j0, nz):
            if any(res.phi[ix + nx * (jy + ny * k)] > 0 for ix in range(nx)):
                kd = k
        return sum(1 for ix in range(nx)
                   if res.phi[ix + nx * (jy + ny * kd)] > 0) * dx

    def test_misaligned_mask_loses_self_termination(self):
        # aligned: the pit pinches toward a pyramid (narrow floor); misaligned:
        # the {111} facets no longer match the mask, so the floor stays wide.
        from soidlc import koh
        d = CR.RateDiagram.from_condition("KOH_30_70")
        sq = lambda x, y: 18 <= x <= 42 and 18 <= y <= 42
        kw = dict(dx=0.6, diagram=d, orientation="100",
                  recipe=koh.WetEtch(steps=130))
        aligned = koh.simulate_3d(sq, 60, 60, 30, misalign_deg=0.0, **kw)
        skew = koh.simulate_3d(sq, 60, 60, 30, misalign_deg=30.0, **kw)
        self.assertGreater(self._floor_width(skew, 30),
                           2.0 * self._floor_width(aligned, 30))


if __name__ == "__main__":
    unittest.main(verbosity=2)
