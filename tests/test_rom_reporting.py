"""The ROM stage and the FEM report must say what they cannot do.

Both produce plausible output for devices they do not actually model, which
is the failure mode that costs the most: a file that looks like an answer.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soidlc import compile_source                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")


def _compile(name, **kw):
    with open(os.path.join(EX, f"{name}.soidl")) as f:
        return compile_source(f.read(), base_dir=EX, **kw)


class TestTorsionalFemReporting(unittest.TestCase):
    """A plane-stress solver has no out-of-plane degree of freedom, so a
    torsional device's operating mode is invisible to it.  It still prints
    the in-plane modes it did find, and 38.61 kHz reads exactly like a
    resonance unless it is labelled."""

    def test_in_plane_modes_are_labelled_as_such(self):
        art = _compile("micromirror", fem=True, fem_h=14.0)
        modes = [l for l in art.report if "modes:" in l]
        self.assertTrue(modes, art.report)
        self.assertIn("in-plane", modes[0].lower(),
                      "torsional device's 2D modes are printed unlabelled")

    def test_the_invisible_operating_mode_is_named(self):
        art = _compile("micromirror", fem=True, fem_h=14.0)
        note = [l for l in art.report + art.warnings
                if "torsional" in l.lower() and "solid3d" in l]
        self.assertTrue(note,
                        "nothing tells the reader the operating mode is "
                        f"missing; report was {art.report}")

    def test_a_translational_device_is_unchanged(self):
        art = _compile("comb_resonator", fem=True, fem_h=15.0)
        modes = [l for l in art.report if "modes:" in l]
        self.assertTrue(modes)
        self.assertNotIn("in-plane", modes[0].lower())
        self.assertTrue([l for l in art.report if "vs lumped" in l])


class TestRomSkipIsExplained(unittest.TestCase):
    def test_torsional_device_says_why_it_has_no_rom(self):
        art = _compile("micromirror", fem=True, fem_h=14.0)
        said = [w for w in art.warnings if w.startswith("rom:")]
        self.assertTrue(said,
                        f"ROM produced nothing and said nothing; "
                        f"warnings were {art.warnings}")
        self.assertIn("transducer", said[0])

    def test_a_device_with_a_comb_still_builds_a_rom(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            art = _compile("folded_flexure_resonator", fem=True, fem_h=14.0,
                           out_prefix=os.path.join(tmp, "d"))
            self.assertIn("rom_cir", art.files)
            self.assertTrue(os.path.exists(art.files["rom_cir"]))
            self.assertEqual([w for w in art.warnings
                              if w.startswith("rom:")], [])


class TestGapClosingRomIsQualified(unittest.TestCase):
    """A BVD model is linear.  A gap-closing pair is not -- its capacitance
    goes as 1/(g-x), and an RF switch is driven deliberately PAST pull-in.
    The model is a small-signal linearisation of a regime the device leaves.
    """

    def test_plate_driven_rom_warns_about_its_validity_limit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            art = _compile("rf_switch", fem=True, fem_h=14.0,
                           out_prefix=os.path.join(tmp, "d"))
            self.assertIn("rom_cir", art.files, "the ROM should still build")
            qual = [w for w in art.warnings
                    if w.startswith("rom:") and "pull-in" in w]
            self.assertTrue(qual,
                            f"gap-closing ROM shipped unqualified; "
                            f"warnings were {art.warnings}")

    def test_comb_driven_rom_is_not_qualified(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            art = _compile("folded_flexure_resonator", fem=True, fem_h=14.0,
                           out_prefix=os.path.join(tmp, "d"))
            self.assertEqual([w for w in art.warnings if "pull-in" in w], [])


if __name__ == "__main__":
    unittest.main()
