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


if __name__ == "__main__":
    unittest.main()
