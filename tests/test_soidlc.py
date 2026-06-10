"""Unit tests for the SOIDL compiler (stdlib unittest, no deps)."""

import math
import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soidlc import compile_source                      # noqa: E402
from soidlc.parser import parse                          # noqa: E402
from soidlc.units import (DimensionError, Quantity,      # noqa: E402
                          make_quantity, parse_unit)
from soidlc import mesh as M                             # noqa: E402
from soidlc import geometry as G                         # noqa: E402

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "examples")


class TestUnits(unittest.TestCase):
    def test_length_conversion(self):
        q = make_quantity(200, "um")
        self.assertAlpha(q.um, 200.0)

    def assertAlpha(self, a, b):
        self.assertTrue(abs(a - b) < 1e-9, f"{a} != {b}")

    def test_dimension_add_mismatch(self):
        with self.assertRaises(DimensionError):
            _ = make_quantity(1, "um") + make_quantity(1, "kHz")

    def test_zero_is_dimension_agnostic(self):
        r = make_quantity(0, None) - make_quantity(5, "um")
        self.assertAlpha(r.um, -5.0)

    def test_compound_units(self):
        f, _ = parse_unit("kg/m^3")
        self.assertAlpha(f, 1.0)
        f2, _ = parse_unit("ohm*cm")
        self.assertAlpha(f2, 1e-2)

    def test_stiffness_dimension(self):
        E = make_quantity(169, "GPa")
        t = make_quantity(25, "um")
        w = make_quantity(4, "um")
        L = make_quantity(200, "um")
        k = E * t * w ** Quantity(3.0) / L ** Quantity(3.0)
        self.assertEqual(k.dim, (0, 1, -2, 0))  # stiffness = N/m


class TestParser(unittest.TestCase):
    def test_parse_examples(self):
        for name in ("comb_resonator.soidl", "accelerometer.soidl"):
            with open(os.path.join(EX, name)) as f:
                ast = parse(f.read())
            self.assertTrue(ast.decls)


class TestGeometryMesh(unittest.TestCase):
    def _manifold(self, mesh):
        edges = Counter()
        for (a, b, c) in mesh.triangles:
            for e in ((a, b), (b, c), (c, a)):
                edges[frozenset(e)] += 1
        return sum(1 for v in edges.values() if v != 2)

    def test_simple_prism_is_watertight(self):
        poly = G.rect(10, 20)
        mesh = M.extrude_polygon(poly, 0, 5, "DEVICE")
        self.assertEqual(self._manifold(mesh), 0)
        self.assertGreater(len(mesh.triangles), 0)

    def test_holed_plate_is_watertight(self):
        ext = G.rect(100, 100).exterior
        holes = [G.rect(5, 5, x, y).exterior
                 for x in (-20, 0, 20) for y in (-20, 0, 20)]
        poly = G.Polygon(ext, holes)
        mesh = M.extrude_polygon(poly, 0, 25, "DEVICE")
        self.assertEqual(self._manifold(mesh), 0)


class TestEndToEnd(unittest.TestCase):
    def test_comb_resonator(self):
        with open(os.path.join(EX, "comb_resonator.soidl")) as f:
            art = compile_source(f.read())
        self.assertGreater(len(art.mesh.triangles), 1000)
        # 4 device layers in the stack
        self.assertEqual(len(art.process.layers), 4)
        # model extraction produced a positive resonant frequency
        f0 = art.model.get("f0")
        self.assertIsNotNone(f0)
        self.assertGreater(f0.value, 1e3)
        self.assertLess(f0.value, 1e6)

    def test_mesh_watertight_end_to_end(self):
        with open(os.path.join(EX, "accelerometer.soidl")) as f:
            art = compile_source(f.read())
        edges = Counter()
        for (a, b, c) in art.mesh.triangles:
            for e in ((a, b), (b, c), (c, a)):
                edges[frozenset(e)] += 1
        self.assertEqual(sum(1 for v in edges.values() if v != 2), 0)


class TestConnectivity(unittest.TestCase):
    def _compile(self, src):
        return compile_source(src)

    def test_examples_have_no_errors(self):
        for name in ("comb_resonator.soidl", "accelerometer.soidl"):
            with open(os.path.join(EX, name)) as f:
                art = compile_source(f.read())
            self.assertEqual(art.errors, [], f"{name}: {art.errors}")

    def test_nets_map_to_distinct_islands(self):
        with open(os.path.join(EX, "comb_resonator.soidl")) as f:
            art = compile_source(f.read())
        nets = [l for l in art.report if l.startswith("net    ")]
        islands = {l.rsplit("#", 1)[1] for l in nets}
        self.assertEqual(len(nets), 3)
        self.assertEqual(len(islands), 3)   # DRIVE, SENSE, GND all distinct
        self.assertTrue(any("isolate" in l and "ok" in l for l in art.report))

    def test_short_is_detected(self):
        src = """
        device bad {
          inst A = anchor(20 um, 20 um) at (0, 0);
          inst B = anchor(20 um, 20 um) at (10 um, 0);   // overlaps A
          net N1 = A;
          net N2 = B;
          isolate N1 from N2 by trench;
        }
        """
        art = self._compile(src)
        self.assertTrue(any("shorted" in e for e in art.errors), art.errors)
        self.assertTrue(any("isolate violated" in e for e in art.errors))

    def test_floating_island_is_detected(self):
        src = """
        device floaty {
          inst M = plate(100 um, 100 um);
          net X = M;
        }
        """
        art = self._compile(src)
        self.assertTrue(any("no anchor" in e for e in art.errors), art.errors)

    def test_split_net_is_detected(self):
        src = """
        device split {
          inst A = anchor(20 um, 20 um) at (0, 0);
          inst B = anchor(20 um, 20 um) at (100 um, 0);  // not connected
          net N = A | B;
        }
        """
        art = self._compile(src)
        self.assertTrue(any("split across" in e for e in art.errors),
                        art.errors)

    def test_comb_fingers_not_shorted(self):
        # interdigitated fingers must remain on two distinct islands
        from soidlc.primitives import prim_comb, PrimitiveCtx
        from soidlc import connectivity
        shapes = prim_comb([], {}, PrimitiveCtx())
        comp = connectivity.components([s.polygon for s in shapes])
        rotor = {c for s, c in zip(shapes, comp) if "rotor" in s.label}
        stator = {c for s, c in zip(shapes, comp) if "stator" in s.label}
        self.assertEqual(len(rotor), 1)
        self.assertEqual(len(stator), 1)
        self.assertNotEqual(rotor, stator)


if __name__ == "__main__":
    unittest.main(verbosity=2)
