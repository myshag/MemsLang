"""Tests for the SOIDL standard library: imports, path geometry, new elements."""

import math
import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soidlc import compile_source                      # noqa: E402
from soidlc import geometry as G                       # noqa: E402
from soidlc import mesh as M                           # noqa: E402
from soidlc import sast as A                           # noqa: E402
from soidlc.parser import parse, ParseError            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")


class TestImportParsing(unittest.TestCase):
    def test_import_becomes_an_ast_node(self):
        ast = parse('import "flexures.soidl";\ndevice d { }')
        imports = [d for d in ast.decls if isinstance(d, A.Import)]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].path, "flexures.soidl")

    def test_import_must_be_top_level(self):
        with self.assertRaises(ParseError):
            parse('device d { import "flexures.soidl"; }')



class TestImportResolution(unittest.TestCase):
    PROC = """
      process p {
        stack {
          layer DEVICE { thickness = 25 um; material = Si;
                         E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
          layer BOX    { thickness = 2 um; material = SiO2 }
          layer HANDLE { thickness = 400 um; material = Si }
        }
        rules { release { hole_size = 6 um; hole_pitch = 30 um;
                          max_solid_span = 40 um } }
      }
    """

    def test_bundled_library_resolves_without_a_base_dir(self):
        src = self.PROC + """
          import "flexures.soidl";
          device d {
            inst M = plate(200 um, 200 um) at (0, 0);
            inst S = array(guided_beam(L = 200 um, w = 4 um), count = 4,
                           place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        self.assertGreater(len(art.mesh.triangles), 0)

    def test_local_component_shadows_the_library_one(self):
        src = self.PROC + """
          import "flexures.soidl";
          component guided_beam(L = 100 um, w = 4 um) {
            mech port fixed, shuttle;
            geometry { anchor(500 um, 7 um) at (0, 0); }
          }
          device d {
            inst S = guided_beam() at (0, 0);
            net GND = S;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        x0, y0, x1, y1 = _bbox(art)
        self.assertAlmostEqual(x1 - x0, 500.0, delta=1.0)

    def test_missing_import_is_a_clean_diagnostic(self):
        src = self.PROC + 'import "nope.soidl";\ndevice d { }'
        with self.assertRaises(Exception) as cm:
            compile_source(src)
        self.assertIn("nope.soidl", str(cm.exception))

    def test_cyclic_import_terminates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.soidl")
            b = os.path.join(tmp, "b.soidl")
            with open(a, "w") as f:
                f.write('import "b.soidl";\ncomponent ca(w = 4 um) '
                        '{ geometry { anchor(20 um, 20 um) at (0, 0); } }\n')
            with open(b, "w") as f:
                f.write('import "a.soidl";\ncomponent cb(w = 4 um) '
                        '{ geometry { anchor(20 um, 20 um) at (0, 0); } }\n')
            src = self.PROC + ('import "a.soidl";\n'
                               'device d { inst S = cb() at (0, 0); net G = S; }')
            art = compile_source(src, base_dir=tmp)
            self.assertEqual(art.errors, [], art.errors)

    def test_library_file_may_not_declare_a_device(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            lib = os.path.join(tmp, "bad.soidl")
            with open(lib, "w") as f:
                f.write("device sneaky { }\n")
            src = self.PROC + 'import "bad.soidl";\ndevice d { }'
            with self.assertRaises(Exception) as cm:
                compile_source(src, base_dir=tmp)
            self.assertIn("bad.soidl", str(cm.exception))


def _bbox(art):
    xs, ys = [], []
    for s in art.result.shapes:
        x0, y0, x1, y1 = s.polygon.bbox()
        xs += [x0, x1]
        ys += [y0, y1]
    return min(xs), min(ys), max(xs), max(ys)



class TestWire(unittest.TestCase):
    def test_straight_wire_is_a_rectangle(self):
        p = G.wire([(0.0, 0.0), (10.0, 0.0)], 4.0)
        self.assertAlmostEqual(p.area(), 40.0, places=6)
        x0, y0, x1, y1 = p.bbox()
        self.assertAlmostEqual(x0, 0.0, places=6)
        self.assertAlmostEqual(x1, 10.0, places=6)
        self.assertAlmostEqual(y0, -2.0, places=6)
        self.assertAlmostEqual(y1, 2.0, places=6)

    def test_right_angle_wire_is_mitred_not_truncated(self):
        """A mitred 90-degree corner runs the outer edges out to their crossing.

        Path (0,0)->(10,0)->(10,10) at width 4: the outer offsets meet at
        (12,-2) and the inner ones at (8,2), so the stroke is the 12x12 box
        [0,12]x[-2,10] minus the 8x8 notch [0,8]x[2,10]  ->  144 - 64 = 80.
        A truncated (bevelled) join would cut the outer corner and give less.
        """
        p = G.wire([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], 4.0)
        self.assertAlmostEqual(p.area(), 80.0, delta=0.01)
        self.assertAlmostEqual(p.bbox()[2], 12.0, places=6)   # outer miter x
        self.assertAlmostEqual(p.bbox()[1], -2.0, places=6)   # outer miter y

    def test_sharp_corner_bevel_clip_keeps_the_polygon_bounded(self):
        """A near-180-degree reversal must not spike off to infinity."""
        p = G.wire([(0.0, 0.0), (50.0, 0.0), (0.0, 1.0)], 4.0)
        x0, y0, x1, y1 = p.bbox()
        self.assertLess(x1, 70.0, "miter spike was not clipped")
        self.assertGreater(p.area(), 0.0)

    def test_wire_rejects_a_degenerate_path(self):
        with self.assertRaises(ValueError):
            G.wire([(0.0, 0.0)], 4.0)



class TestCircularGeometry(unittest.TestCase):
    def test_circle_area_converges(self):
        self.assertAlmostEqual(G.circle(10.0, 256).area(), math.pi * 100.0,
                               delta=0.05)

    def test_annulus_area(self):
        R, w = 50.0, 10.0
        p = G.annulus(R, w, 256)
        expected = math.pi * ((R + w / 2) ** 2 - (R - w / 2) ** 2)
        self.assertAlmostEqual(p.area(), expected, delta=expected * 0.001)
        self.assertEqual(len(p.holes), 1)
        self.assertTrue(p.band)

    def test_annulus_rings_pair_one_to_one(self):
        p = G.annulus(50.0, 10.0, 32)
        self.assertEqual(len(p.exterior), len(p.holes[0]))

    def test_arc_is_a_quarter_of_the_band(self):
        R, w = 50.0, 10.0
        full = math.pi * ((R + w / 2) ** 2 - (R - w / 2) ** 2)
        q = G.arc(R, w, 0.0, 90.0, 256).area()
        self.assertAlmostEqual(q, full / 4.0, delta=full * 0.002)

    def test_band_flag_survives_placement(self):
        """elaborate translates and rotates every shape; a ring that lost the
        flag would silently fall back to the non-manifold bridging path."""
        p = G.annulus(50.0, 10.0, 32)
        self.assertTrue(p.translated(5.0, 5.0).band)
        self.assertTrue(p.rotated(90).band)
        self.assertTrue(p.normalized().band)
        self.assertTrue(p.mirrored(True, False).band)


if __name__ == "__main__":
    unittest.main()
