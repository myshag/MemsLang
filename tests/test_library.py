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



class TestRingMesh(unittest.TestCase):
    """The annulus extrusion defect and its fix.

    Measured before the fix: exactly 4 non-manifold edges, constant across
    n_seg in {8,16,64,128}.  They were not boundary edges (count 1) but edges
    shared by FOUR triangles -- _bridge_holes cuts a zero-width slit from the
    hole to the exterior and traverses it twice, and the extruder's vertex
    cache merges the two sides into one edge.
    """

    def _bad_edges(self, mesh):
        edges = Counter()
        for (a, b, c) in mesh.triangles:
            for e in ((a, b), (b, c), (c, a)):
                edges[frozenset(e)] += 1
        return sum(1 for v in edges.values() if v != 2)

    def test_annulus_extrudes_watertight(self):
        for n_seg in (8, 16, 64):
            with self.subTest(n_seg=n_seg):
                poly = G.annulus(50.0, 10.0, n_seg)
                mesh = M.extrude_polygon(poly, 0.0, 25.0, "DEVICE")
                self.assertEqual(self._bad_edges(mesh), 0)
                self.assertGreater(len(mesh.triangles), 0)

    def test_annulus_survives_placement_transforms(self):
        """elaborate mirrors and rotates shapes when resolving `attach`.

        Mirroring flips ring winding, so a cap that paired vertices by raw
        index would silently produce twisted triangles.  Placement must not
        change watertightness.
        """
        base = G.annulus(50.0, 10.0, 32)
        for name, poly in (
            ("translated", base.translated(120.0, -40.0)),
            ("rotated90", base.rotated(90)),
            ("rotated37", base.rotated(37)),
            ("mirrored_x", base.mirrored(True, False)),
            ("mirrored_xy", base.mirrored(True, True)),
            ("normalized_mirror", base.mirrored(True, False).normalized()),
        ):
            with self.subTest(transform=name):
                mesh = M.extrude_polygon(poly, 0.0, 25.0, "DEVICE")
                self.assertEqual(self._bad_edges(mesh), 0)

    def test_annulus_cap_area_matches_the_polygon(self):
        """Watertight is necessary but not sufficient: twisted pairing can
        still be closed.  The cap triangles must also cover the real area."""
        poly = G.annulus(50.0, 10.0, 64)
        tris = M.triangulate_with_holes(poly)
        area = 0.0
        for (a, b, c) in tris:
            area += abs((b[0] - a[0]) * (c[1] - a[1])
                        - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
        self.assertAlmostEqual(area, poly.area(), delta=poly.area() * 0.01)

    def test_disk_extrudes_watertight(self):
        mesh = M.extrude_polygon(G.circle(50.0, 64), 0.0, 25.0, "DEVICE")
        self.assertEqual(self._bad_edges(mesh), 0)

    def test_tilted_bar_extrudes_watertight(self):
        # the chevron case: a rectangle rotated off-axis leaves the voxel path
        mesh = M.extrude_polygon(G.rect(100.0, 6.0).rotated(8.0),
                                 0.0, 25.0, "DEVICE")
        self.assertEqual(self._bad_edges(mesh), 0)



class TestFrame(unittest.TestCase):
    def test_frame_is_a_hollow_square(self):
        src = TestImportResolution.PROC + """
          import "flexures.soidl";
          device d {
            inst F = frame(W = 680 um, H = 680 um, bar = 40 um) at (0, 0);
            inst A = anchor(30 um, 30 um) at (0, 320 um);
            net GND = F | A;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        x0, y0, x1, y1 = _bbox(art)
        self.assertAlmostEqual(x1 - x0, 680.0, delta=1.0)
        self.assertAlmostEqual(y1 - y0, 680.0, delta=1.0)
        area = sum(s.polygon.area() for s in art.result.shapes
                   if s.layer == "DEVICE")
        self.assertLess(area, 680.0 * 680.0 * 0.5, "frame is not hollow")

    def test_frame_corners_are_counted_once(self):
        """Horizontal bars span the full width and vertical ones are shortened
        by 2*bar, so corner silicon is covered exactly once -- but the bars
        must still abut, or the ring is four islands instead of one."""
        src = TestImportResolution.PROC + """
          import "flexures.soidl";
          device d {
            inst F = frame(W = 400 um, H = 400 um, bar = 20 um) at (0, 0);
            inst A = anchor(30 um, 30 um) at (0, 190 um);
            net GND = F | A;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        bars = [s for s in art.result.shapes
                if s.layer == "DEVICE" and s.label == "beam"]
        exact = 2 * (400.0 * 20.0) + 2 * ((400.0 - 40.0) * 20.0)
        self.assertAlmostEqual(sum(b.polygon.area() for b in bars), exact,
                               delta=1.0)



class TestFoldedFlexure(unittest.TestCase):
    def _k(self, component):
        src = TestImportResolution.PROC + f"""
          import "flexures.soidl";
          device d {{
            inst M = plate(200 um, 200 um) at (0, 0);
            inst S = array({component}, count = 4, place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }}
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        return art.model["k"].value

    def test_folded_is_softer_than_guided_by_the_series_ratio(self):
        """Two beams in series, but NOT two equal beams.

        The fold's anchor has to stand `gap` clear of the mass edge, so the
        outboard beam is L - gap - 2 long while the inboard one is L.  Adding
        compliances gives k_folded/k_guided = L^3 / (L^3 + L_out^3), which is
        0.63 at the defaults -- softer than a guided beam, but not the 0.5 the
        equal-length idealisation would predict.  Asserting 0.5 here would
        force the formula back to a lie the FEM check then has to catch.
        """
        L, gap = 200.0, 30.0
        L_out = L - gap - 2.0
        expected = L ** 3 / (L ** 3 + L_out ** 3)
        k_guided = self._k("guided_beam(L = 200 um, w = 4 um, n_beams = 2)")
        k_folded = self._k("folded_flexure(L = 200 um, w = 4 um, n_folds = 2)")
        self.assertLess(k_folded, k_guided)
        self.assertAlmostEqual(k_folded / k_guided, expected, delta=0.02)

    def test_stiffness_scales_with_fold_count(self):
        k2 = self._k("folded_flexure(L = 200 um, w = 4 um, n_folds = 2)")
        k4 = self._k("folded_flexure(L = 200 um, w = 4 um, n_folds = 4)")
        self.assertAlmostEqual(k4 / k2, 2.0, delta=0.05)

    def test_anchor_does_not_short_the_spring(self):
        """The failure `derive` cannot see, and LVS will not report.

        An anchor wide enough to touch its fold's INBOARD beam (or a tie bar
        drawn across the folds) turns the flexure into a rigid block while
        `derive k.x` keeps reporting a compliant spring.  One island is
        electrically legal, so connectivity passes and the device silently
        resonates at the wrong frequency.  Each anchor must therefore touch
        exactly one beam: its own outboard beam.
        """
        from soidlc import connectivity
        src = TestImportResolution.PROC + """
          import "flexures.soidl";
          device d {
            inst S = folded_flexure(L = 200 um, w = 4 um, n_folds = 3)
                     at (0, 0);
            net GND = S;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        dev = [s for s in art.result.shapes if s.layer == "DEVICE"]
        anchors = [s for s in dev if s.label == "anchor"]
        beams = [s for s in dev if s.label == "beam"]
        self.assertEqual(len(anchors), 3)
        for a in anchors:
            touched = [b for b in beams
                       if connectivity.touches(a.polygon, b.polygon)]
            self.assertEqual(
                len(touched), 1,
                f"anchor at {a.polygon.bbox()} touches {len(touched)} beams; "
                f"it must touch only its own outboard beam")



class TestFoldedFlexureGeometry(unittest.TestCase):
    """Two shorts that no existing check can see.

    Both leave the device with one legal electrical island, so connectivity
    passes; both leave `derive k.x` reporting a compliant spring while the
    silicon is rigid.  Each cost a wrong FEM answer during development
    (192 kHz and 220 kHz against an intended 26 kHz) before being found.
    """

    SRC = TestImportResolution.PROC + """
      import "flexures.soidl";
      device d {
        inst M = plate(300 um, 300 um) at (0, 0);
        inst S = array(folded_flexure(L = 200 um, w = 4 um, n_folds = 2),
                       count = 4, place = corners(M));
        net GND = M | S.fixed;
        constraint anchored(S.fixed);
      }
    """

    def _shapes(self):
        art = compile_source(self.SRC)
        self.assertEqual(art.errors, [], art.errors)
        return art, [s for s in art.result.shapes if s.layer == "DEVICE"]

    def test_no_anchor_is_bolted_to_the_proof_mass(self):
        from soidlc import connectivity
        art, dev = self._shapes()
        plate = [s for s in dev if s.label == "plate"][0]
        shorted = [a for a in dev if a.label == "anchor"
                   and connectivity.touches(a.polygon, plate.polygon)]
        self.assertEqual(
            shorted, [],
            "an anchor touching the proof mass ties it straight to the "
            "substrate: the suspension stops being a suspension")

    def test_each_anchor_grips_exactly_one_beam(self):
        from soidlc import connectivity
        art, dev = self._shapes()
        beams = [s for s in dev if s.label == "beam"]
        anchors = [s for s in dev if s.label == "anchor"]
        self.assertEqual(len(anchors), 8)
        for a in anchors:
            n = sum(1 for b in beams
                    if connectivity.touches(a.polygon, b.polygon))
            self.assertEqual(n, 1,
                             f"anchor at {a.polygon.bbox()} grips {n} beams; "
                             f"only its own outboard beam may be gripped")

    def test_mass_reaches_the_substrate_only_through_two_beams(self):
        """The series path: mass -> inboard beam -> truss -> outboard beam ->
        anchor.  If the outboard beam also reached the mass, the mass would
        hang off the anchor through a few um of silicon and be rigid."""
        from soidlc import connectivity
        art, dev = self._shapes()
        plate = [s for s in dev if s.label == "plate"][0]
        anchors = [s for s in dev if s.label == "anchor"]
        for a in anchors:
            beams_on_anchor = [b for b in dev if b.label == "beam"
                               and connectivity.touches(a.polygon, b.polygon)]
            for b in beams_on_anchor:
                self.assertFalse(
                    connectivity.touches(b.polygon, plate.polygon),
                    "an anchored beam also touches the mass: the suspension "
                    "is short-circuited")


class TestFoldedFlexureFEM(unittest.TestCase):
    """Validates the folded-flexure stiffness formula against plane-stress FEM.

    This is the check that caught both geometry shorts above -- neither the
    parser, the unit checker nor the LVS extractor could see them, because a
    rigid block is a perfectly legal device.
    """

    def test_lumped_f0_agrees_with_fem(self):
        with open(os.path.join(EX, "folded_flexure_resonator.soidl")) as f:
            art = compile_source(f.read(), fem=True, fem_h=12.0, base_dir=EX)
        self.assertEqual(art.errors, [], art.errors)
        cmp = [l for l in art.report
               if l.startswith("fem") and "vs lumped" in l]
        self.assertTrue(cmp, art.report)
        pct = float(cmp[0].rsplit("(", 1)[1].rstrip("%)"))
        self.assertLess(abs(pct), 15.0, cmp[0])



class TestCompliantFlexures(unittest.TestCase):
    def _k(self, component):
        src = TestImportResolution.PROC + f"""
          import "flexures.soidl";
          device d {{
            inst M = plate(300 um, 300 um) at (0, 0);
            inst S = array({component}, count = 4, place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }}
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        return art.model["k"].value

    def test_more_turns_is_softer(self):
        k2 = self._k("serpentine(L = 120 um, w = 3 um, n_turns = 2)")
        k6 = self._k("serpentine(L = 120 um, w = 3 um, n_turns = 6)")
        self.assertLess(k6, k2)
        self.assertAlmostEqual(k2 / k6, 3.0, delta=0.2)

    def test_crab_leg_reports_no_stiffness_rather_than_a_wrong_one(self):
        """crab_leg ships without a closed-form k on purpose.

        The cantilever-style expression for it measured 38.7% low against
        plane-stress FEM (67.4 kHz vs 48.6 kHz predicted), because for motion
        along x the thigh is loaded axially and contributes almost no bending
        compliance.  A device on crab legs alone therefore has no lumped f0 --
        no number beats a confidently wrong one in `solve` and `require`.
        """
        src = TestImportResolution.PROC + """
          import "flexures.soidl";
          device d {
            inst M = plate(300 um, 300 um) at (0, 0);
            inst S = array(crab_leg(Lx = 150 um, Ly = 60 um, w = 4 um),
                           count = 4, place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        self.assertIn("m", art.model)          # mass is measurable
        self.assertNotIn("k", art.model)       # stiffness is not claimed
        self.assertNotIn("f0", art.model)

    def test_crab_leg_is_a_real_spring_geometrically(self):
        """No formula, but the silicon must still be a suspension: a single
        chain mass -> thigh -> shin -> anchor, with the anchor off the mass."""
        from soidlc import connectivity
        src = TestImportResolution.PROC + """
          import "flexures.soidl";
          device d {
            inst M = plate(300 um, 300 um) at (0, 0);
            inst S = array(crab_leg(Lx = 150 um, Ly = 60 um, w = 4 um),
                           count = 4, place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src)
        dev = [s for s in art.result.shapes if s.layer == "DEVICE"]
        plate = [s for s in dev if s.label == "plate"][0]
        for a in [s for s in dev if s.label == "anchor"]:
            self.assertFalse(connectivity.touches(a.polygon, plate.polygon))
            gripped = [b for b in dev if b.label == "beam"
                       and connectivity.touches(a.polygon, b.polygon)]
            self.assertEqual(len(gripped), 1)
            self.assertFalse(
                connectivity.touches(gripped[0].polygon, plate.polygon),
                "the anchored shin also reaches the mass: spring shorted")

    def test_compliant_flexures_do_not_short_their_springs(self):
        """The folded-flexure lesson, applied to the other two: an anchor that
        touches the proof mass makes a rigid device that nothing complains
        about."""
        from soidlc import connectivity
        for comp in ("serpentine(L = 120 um, w = 3 um, n_turns = 4)",
                     "crab_leg(Lx = 150 um, Ly = 60 um, w = 4 um)"):
            with self.subTest(component=comp):
                src = TestImportResolution.PROC + f"""
                  import "flexures.soidl";
                  device d {{
                    inst M = plate(300 um, 300 um) at (0, 0);
                    inst S = array({comp}, count = 4, place = corners(M));
                    net GND = M | S.fixed;
                    constraint anchored(S.fixed);
                  }}
                """
                art = compile_source(src)
                self.assertEqual(art.errors, [], art.errors)
                dev = [s for s in art.result.shapes if s.layer == "DEVICE"]
                plate = [s for s in dev if s.label == "plate"][0]
                shorted = [a for a in dev if a.label == "anchor"
                           and connectivity.touches(a.polygon, plate.polygon)]
                self.assertEqual(shorted, [],
                                 "anchor bolts the proof mass to the substrate")



class TestSerpentineFEM(unittest.TestCase):
    """The serpentine's series behaviour, checked on a small probe device.

    low_g_accel itself meshes to ~37k elements (the 3 um spans are kept
    fine-meshed on purpose), past the solver's 30k guard, so the component is
    validated on a device small enough to mesh rather than not at all.
    """

    def test_serpentine_lumped_f0_agrees_with_fem(self):
        src = TestImportResolution.PROC + """
          import "flexures.soidl";
          device s_probe {
            inst M = plate(200 um, 120 um) at (0, 0);
            inst S = array(serpentine(L = 150 um, w = 4 um, n_turns = 3),
                           count = 4, place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src, fem=True, fem_h=10.0)
        self.assertEqual(art.errors, [], art.errors)
        cmp = [l for l in art.report
               if l.startswith("fem") and "vs lumped" in l]
        self.assertTrue(cmp, art.report)
        pct = float(cmp[0].rsplit("(", 1)[1].rstrip("%)"))
        self.assertLess(abs(pct), 15.0, cmp[0])

    def test_meander_is_one_connected_path(self):
        """The property that makes the parallel-spans short impossible."""
        from soidlc import primitives as P
        from soidlc.units import Quantity
        um = lambda v: Quantity(v * 1e-6, (1, 0, 0, 0))
        shapes = P.PRIMITIVES["meander"](
            [], {"L": um(150), "w": um(4), "n_turns": 5, "pitch": um(14)},
            P.PrimitiveCtx())
        paths = [s for s in shapes if s.label == "meander"]
        self.assertEqual(len(paths), 1, "the spring must be a single polygon")
        self.assertGreater(paths[0].polygon.area(), 0.0)



class TestDETF(unittest.TestCase):
    """A DETF is distributed, not a mass on a spring.

    Every other resonator here has an obvious proof mass; a tuning fork's
    frequency comes from the tine itself, so squeezing it into
    f0 = sqrt(k/m) is the least favourable case for the lumped model in this
    repo.  Measured agreement is +1.8%, which is better than the design
    expected -- worth pinning so a future change to _extract_device_model
    cannot quietly degrade it.
    """

    def test_lumped_f0_agrees_with_fem(self):
        with open(os.path.join(EX, "tuning_fork_detf.soidl")) as f:
            art = compile_source(f.read(), fem=True, fem_h=8.0, base_dir=EX)
        self.assertEqual(art.errors, [], art.errors)
        cmp = [l for l in art.report
               if l.startswith("fem") and "vs lumped" in l]
        self.assertTrue(cmp, art.report)
        pct = float(cmp[0].rsplit("(", 1)[1].rstrip("%)"))
        self.assertLess(abs(pct), 15.0, cmp[0])

    def test_drive_comb_is_anchored_not_floating(self):
        """A DETF has no shuttle, so the comb rotor needs a paddle on a tine.
        Without it the rotor is a released island with nothing holding it and
        would detach during release."""
        with open(os.path.join(EX, "tuning_fork_detf.soidl")) as f:
            art = compile_source(f.read(), base_dir=EX)
        self.assertEqual(art.errors, [], art.errors)



def _um(v):
    from soidlc.units import Quantity
    return Quantity(v * 1e-6, (1, 0, 0, 0))


class TestParallelPlate(unittest.TestCase):
    def test_emits_a_released_and_an_anchored_plate(self):
        from soidlc import primitives as P
        shapes = P.PRIMITIVES["parallel_plate"](
            [], {"W": _um(100), "H": _um(40), "g": _um(3), "n": 2},
            P.PrimitiveCtx())
        self.assertEqual(
            sorted({s.label for s in shapes}),
            ["plate_rotor_bar", "plate_stator_bar", "rotor_plate",
             "stator_plate"])
        mechs = {s.label: s.mech for s in shapes}
        self.assertEqual(mechs["rotor_plate"], "released")
        self.assertEqual(mechs["stator_plate"], "anchored")
        self.assertEqual(sum(1 for s in shapes if s.label == "rotor_plate"), 2)

    def test_gap_is_respected(self):
        from soidlc import primitives as P
        shapes = P.PRIMITIVES["parallel_plate"](
            [], {"W": _um(100), "H": _um(40), "g": _um(3), "n": 1},
            P.PrimitiveCtx())
        rot = [s for s in shapes if s.label == "rotor_plate"][0]
        sta = [s for s in shapes if s.label == "stator_plate"][0]
        self.assertAlmostEqual(sta.polygon.bbox()[1] - rot.polygon.bbox()[3],
                               3.0, places=3)

    def test_plates_do_not_touch(self):
        """A rotor touching its stator is a short, and at zero gap the
        electrostatics are meaningless."""
        from soidlc import primitives as P
        from soidlc import connectivity
        shapes = P.PRIMITIVES["parallel_plate"](
            [], {"W": _um(100), "H": _um(40), "g": _um(3), "n": 3},
            P.PrimitiveCtx())
        rotors = [s for s in shapes if s.label == "rotor_plate"]
        stators = [s for s in shapes if s.label == "stator_plate"]
        for r in rotors:
            for st in stators:
                self.assertFalse(
                    connectivity.touches(r.polygon, st.polygon),
                    "rotor and stator plates must not touch")



class TestThermalActuators(unittest.TestCase):
    def _bad_edges(self, mesh):
        edges = Counter()
        for (a, b, c) in mesh.triangles:
            for e in ((a, b), (b, c), (c, a)):
                edges[frozenset(e)] += 1
        return sum(1 for v in edges.values() if v != 2)

    def test_chevron_beams_are_inclined_and_paired(self):
        from soidlc import primitives as P
        shapes = P.PRIMITIVES["chevron"](
            [], {"n": 3, "L": _um(200), "w": _um(6), "angle": 6.0},
            P.PrimitiveCtx())
        beams = [s for s in shapes if s.label == "chevron_beam"]
        self.assertEqual(len(beams), 6)          # n pairs
        self.assertEqual(len([s for s in shapes
                              if s.label == "chevron_shuttle"]), 1)
        self.assertFalse(M._is_rectilinear(beams[0].polygon.exterior),
                         "a chevron beam must be inclined, not axis-aligned")

    def test_chevron_extrudes_watertight(self):
        """Inclined beams leave the voxel mesher for the general path, which
        is exactly the path the ring fix had to repair."""
        from soidlc import primitives as P
        shapes = P.PRIMITIVES["chevron"](
            [], {"n": 2, "L": _um(200), "w": _um(6), "angle": 6.0},
            P.PrimitiveCtx())
        for s in shapes:
            with self.subTest(shape=s.label):
                mesh = M.extrude_polygon(s.polygon, 0.0, 25.0, "DEVICE")
                self.assertEqual(self._bad_edges(mesh), 0)

    def test_chevron_shuttle_is_connected_to_every_beam(self):
        """The shuttle is what the beams push; a gap there and the actuator
        pushes nothing while still looking like a legal device."""
        from soidlc import primitives as P
        from soidlc import connectivity
        shapes = P.PRIMITIVES["chevron"](
            [], {"n": 3, "L": _um(200), "w": _um(6), "angle": 6.0},
            P.PrimitiveCtx())
        shuttle = [s for s in shapes if s.label == "chevron_shuttle"][0]
        beams = [s for s in shapes if s.label == "chevron_beam"]
        for b in beams:
            self.assertTrue(
                connectivity.touches(shuttle.polygon, b.polygon),
                "a chevron beam does not reach the shuttle")

    def test_chevron_anchors_hold_the_outer_beam_ends(self):
        from soidlc import primitives as P
        from soidlc import connectivity
        shapes = P.PRIMITIVES["chevron"](
            [], {"n": 3, "L": _um(200), "w": _um(6), "angle": 6.0},
            P.PrimitiveCtx())
        anchors = [s for s in shapes if s.label == "chevron_anchor"]
        beams = [s for s in shapes if s.label == "chevron_beam"]
        self.assertEqual(len(anchors), 2)
        for b in beams:
            self.assertTrue(
                any(connectivity.touches(a.polygon, b.polygon)
                    for a in anchors),
                "a chevron beam has no anchored end")

    def test_hot_arm_is_asymmetric(self):
        from soidlc import primitives as P
        shapes = P.PRIMITIVES["hot_arm"](
            [], {"L": _um(200), "w_hot": _um(3), "w_cold": _um(12),
                 "g": _um(4)}, P.PrimitiveCtx())
        hot = [s for s in shapes if s.label == "hot_arm"][0]
        cold = [s for s in shapes if s.label == "cold_arm"][0]
        # the thin arm carries the higher resistance per length, so it runs
        # hotter and expands more -- the asymmetry IS the actuator
        self.assertLess(hot.polygon.area(), cold.polygon.area())



class TestActuatorMetrics(unittest.TestCase):
    PROC = TestImportResolution.PROC

    def _pull_in(self, gap_um):
        from soidlc import metrics
        src = self.PROC + f"""
          import "flexures.soidl";
          device d {{
            inst M = plate(200 um, 100 um) at (0, 0);
            inst S = array(guided_beam(L = 250 um, w = 4 um), count = 4,
                           place = corners(M));
            inst PP = parallel_plate(W = 60 um, H = 30 um,
                                     g = {gap_um} um, n = 2)
                      attach (rotor -> M.top);
            net DRIVE = PP.stator;
            net GND   = M | S.fixed;
            isolate DRIVE from GND by trench;
            constraint anchored(S.fixed, PP.stator);
          }}
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        env = metrics.build_env(art.elab, art.result)
        return env["pull_in"]().value

    def test_pull_in_is_computed_from_the_geometry(self):
        v = self._pull_in(3.0)
        self.assertGreater(v, 0.0)
        self.assertLess(v, 1000.0, "implausible collapse voltage")

    def test_pull_in_scales_with_gap_to_the_three_halves(self):
        """V_pi ~ g^(3/2). Two devices differing only in g, so the ratio
        isolates the gap dependence -- and it goes through the compiler and
        the transducer extractor, not a formula retyped in the test."""
        self.assertAlmostEqual(self._pull_in(6.0) / self._pull_in(3.0),
                               2 ** 1.5, delta=0.05)

    def test_pull_in_refuses_a_device_with_no_gap_closing_pair(self):
        from soidlc import metrics
        src = self.PROC + """
          import "flexures.soidl";
          device d {
            inst M = plate(200 um, 100 um) at (0, 0);
            inst S = array(guided_beam(L = 250 um, w = 4 um), count = 4,
                           place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src)
        env = metrics.build_env(art.elab, art.result)
        with self.assertRaises(metrics.MetricError):
            env["pull_in"]()

    def test_thermal_stroke_grows_with_temperature(self):
        from soidlc import metrics
        d1 = metrics.chevron_stroke(200e-6, 6.0, 50.0)
        d2 = metrics.chevron_stroke(200e-6, 6.0, 200.0)
        self.assertGreater(d2, d1)
        self.assertGreater(d1, 0.0)

    def test_shallower_chevron_gives_more_stroke(self):
        """A shallower V converts the same expansion into more motion -- the
        force-for-stroke trade that sets a chevron's angle."""
        from soidlc import metrics
        shallow = metrics.chevron_stroke(200e-6, 3.0, 100.0)
        steep = metrics.chevron_stroke(200e-6, 12.0, 100.0)
        self.assertGreater(shallow, steep)

    def test_exact_stroke_matches_small_angle_where_that_is_valid(self):
        """Cross-check the exact form against the textbook approximation in
        the regime where the approximation holds."""
        import math as _m
        from soidlc import metrics
        L, ang, dT = 200e-6, 6.0, 100.0
        exact = metrics.chevron_stroke(L, ang, dT)
        approx = L * metrics.ALPHA_SI * dT / _m.sin(_m.radians(ang))
        self.assertAlmostEqual(exact / approx, 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
