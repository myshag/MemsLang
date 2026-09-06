"""Connectivity: holes must not conduct, and a bypassed suspension must warn."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soidlc import compile_source                      # noqa: E402
from soidlc import connectivity                        # noqa: E402
from soidlc import geometry as G                       # noqa: E402

PROC = """
  process p {
    stack {
      layer METAL  { thickness = 1 um; material = Au }
      layer DEVICE { thickness = 25 um; material = Si;
                     E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
      layer BOX    { thickness = 2 um; material = SiO2 }
      layer HANDLE { thickness = 400 um; material = Si }
    }
    rules { release { hole_size = 6 um; hole_pitch = 30 um;
                      max_solid_span = 40 um } }
  }
"""


class TestHolesDoNotConduct(unittest.TestCase):
    """A shape sitting inside a hole is separated from the silicon around it.

    _point_in tests only the exterior ring, so anything inside a release hole
    or inside a ring's bore was reported as electrically connected to the
    shape whose hole it sits in.  That silently fuses islands, which is the
    one thing the LVS extractor exists to get right.
    """

    def test_shape_inside_a_ring_bore_is_separate(self):
        ring = G.annulus(200.0, 20.0, 96)
        hub = G.rect(40.0, 40.0, 0.0, 0.0)
        self.assertFalse(connectivity.touches(hub, ring))
        self.assertFalse(connectivity.touches(ring, hub))

    def test_shape_inside_a_release_hole_is_separate(self):
        plate = G.Polygon(G.rect(200.0, 200.0).exterior,
                          [G.rect(20.0, 20.0).exterior])
        dot = G.rect(6.0, 6.0, 0.0, 0.0)
        self.assertFalse(connectivity.touches(dot, plate))
        self.assertFalse(connectivity.touches(plate, dot))

    def test_a_shape_straddling_a_hole_rim_still_touches(self):
        """Only what is wholly inside the hole is separate; silicon that
        reaches the rim is still continuous."""
        plate = G.Polygon(G.rect(200.0, 200.0).exterior,
                          [G.rect(20.0, 20.0).exterior])
        straddler = G.rect(30.0, 6.0, 0.0, 0.0)   # spans the hole and beyond
        self.assertTrue(connectivity.touches(straddler, plate))

    def test_ordinary_overlap_still_touches(self):
        a = G.rect(100.0, 10.0, 0.0, 0.0)
        b = G.rect(10.0, 100.0, 0.0, 0.0)
        self.assertTrue(connectivity.touches(a, b))

    def test_ordinary_separation_still_does_not(self):
        a = G.rect(10.0, 10.0, 0.0, 0.0)
        b = G.rect(10.0, 10.0, 100.0, 0.0)
        self.assertFalse(connectivity.touches(a, b))

    def test_an_island_in_a_bore_is_reported_as_its_own_island(self):
        """End to end: an electrode inside a ring must not be fused with it,
        or an `isolate` violation would go unreported."""
        src = PROC + """
          device d {
            inst R  = ring(R = 200 um, w = 20 um) at (0, 0);
            inst SP = beam(180 um, 4 um, dir = y) at (0, 100 um);
            inst A  = anchor(30 um, 30 um) at (0, 0);
            inst EL = anchor(20 um, 20 um) at (100 um, 0);
            net GND = R | SP;
            net SIG = EL;
            isolate SIG from GND by trench;
            constraint anchored(A);
          }
        """
        art = compile_source(src)
        shorts = [e for e in art.errors if "shorted" in e]
        self.assertEqual(shorts, [],
                         "an electrode inside the ring bore was fused with "
                         "the ring")


class TestBypassedSuspension(unittest.TestCase):
    """A declared suspension that the geometry short-circuits.

    This is the class of defect that cost the most during the element-library
    work: an anchor a few um out of place welds the proof mass straight to the
    substrate.  The device stays one legal electrical island with an anchor
    and no shorts, so every existing check passes it, and `derive k.x` keeps
    reporting a compliant spring because it is arithmetic that never sees the
    silicon.  Measured on the real case: the compiler reported 24.1 kHz where
    FEM measured 220.1 kHz.

    The contradiction is checkable: the device declares a stiffness AND its
    proof mass is in direct contact with an anchor.  A warning, not an error --
    a clamped membrane rim is the same geometry with none of the intent.
    """

    BROKEN_FLEXURE = """
      component bad_flexure(L = 200 um, w = 4 um) {
        mech port fixed, shuttle;
        derive k.x = (process.DEVICE.E * process.DEVICE.thickness * w^3) / L^3;
        geometry {
          beam(L, w, dir = y) at (0, 2 um - L/2);
          // the defect: the anchor lands on the mass side, not outboard
          anchor(20 um, 20 um) at (0, 12 um);
        }
      }
    """

    def test_welded_mass_warns(self):
        src = PROC + self.BROKEN_FLEXURE + """
          device d {
            inst M = plate(300 um, 300 um) at (0, 0);
            inst S = array(bad_flexure(L = 200 um, w = 4 um), count = 4,
                           place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [],
                         "this must stay a warning, not an error")
        welded = [w for w in art.warnings if "bypassed" in w]
        self.assertTrue(welded,
                        f"no bypassed-suspension warning; got {art.warnings}")

    def test_a_buried_torsion_bar_warns(self):
        """The other way this happened: torsion_bar grows in -x, so placing
        the right-hand one with `at` instead of `attach` buries it in the
        plate with its anchor inside -- which is how a mirror ends up rigidly
        held while reporting an unchanged f0_theta."""
        src = PROC + """
          import "flexures.soidl";
          device mirror {
            inst MIR = plate(500 um, 400 um, holes = none) at (0, 0);
            inst TL  = torsion_bar(L = 140 um, w = 6 um) at (0 - 250 um, 0);
            inst TR  = torsion_bar(L = 140 um, w = 6 um) at (250 um, 0);
            net GND = MIR | TL.fixed | TR.fixed;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        welded = [w for w in art.warnings if "bypassed" in w]
        self.assertTrue(welded, f"buried bar not caught; got {art.warnings}")
        self.assertIn("TR", welded[0], "the warning must name the culprit")

    def test_a_correct_suspension_is_silent(self):
        src = PROC + """
          import "flexures.soidl";
          device d {
            inst M = plate(300 um, 300 um) at (0, 0);
            inst S = array(guided_beam(L = 200 um, w = 4 um), count = 4,
                           place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        self.assertEqual([w for w in art.warnings if "bypassed" in w], [])

    def test_a_device_declaring_no_stiffness_is_silent(self):
        """A clamped membrane is welded on purpose and claims no spring."""
        src = PROC + """
          import "membranes.soidl";
          device d {
            inst MB = membrane(W = 400 um, H = 400 um, margin = 40 um)
                      at (0, 0);
            net GND = MB;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        self.assertEqual([w for w in art.warnings if "bypassed" in w], [])

    def test_every_shipped_example_is_silent(self):
        """A warning that fires on correct geometry is one nobody keeps on."""
        import glob
        EX = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "examples")
        for path in sorted(glob.glob(os.path.join(EX, "*.soidl"))):
            name = os.path.basename(path)
            with self.subTest(example=name):
                with open(path) as f:
                    art = compile_source(f.read(), base_dir=EX)
                self.assertEqual(
                    [w for w in art.warnings if "bypassed" in w], [],
                    f"{name} triggers the bypassed-suspension warning")


if __name__ == "__main__":
    unittest.main()
