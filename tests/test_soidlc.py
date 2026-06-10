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

    def test_split_net_across_released_islands_is_detected(self):
        # two separate *released* masses cannot be one silicon node
        src = """
        device split {
          inst A = plate(80 um, 80 um) at (0, 0);
          inst B = plate(80 um, 80 um) at (400 um, 0);   // not connected
          net N = A | B;
        }
        """
        art = self._compile(src)
        self.assertTrue(any("split across" in e and "released" in e
                            for e in art.errors), art.errors)

    def test_metal_routed_anchored_islands_allowed(self):
        # two separate *anchored* stators wired to one pad via METAL is legal
        src = """
        device routed {
          inst A = anchor(20 um, 20 um) at (0, 0);
          inst B = anchor(20 um, 20 um) at (100 um, 0);
          net N = A | B;
        }
        """
        art = self._compile(src)
        self.assertFalse(any("split" in e for e in art.errors), art.errors)
        self.assertTrue(any("METAL-routed" in l for l in art.report),
                        art.report)

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


class TestClosure(unittest.TestCase):
    """Design closure: solve targets are synthesised from the spec."""

    def test_resonator_meets_spec(self):
        with open(os.path.join(EX, "comb_resonator.soidl")) as f:
            art = compile_source(f.read())
        self.assertEqual(art.errors, [], art.errors)
        # f0_target = 20 kHz, within 1%
        f0 = art.model["f0"].value
        self.assertLess(abs(f0 - 20e3) / 20e3, 0.015, f0)
        solved = [l for l in art.report if l.startswith("closure: solved")]
        self.assertEqual(len(solved), 2, art.report)
        self.assertTrue(any("S.L" in l for l in solved))
        self.assertTrue(any("D1.N" in l for l in solved))
        ok = [l for l in art.report if l.startswith("require") and ": ok" in l]
        self.assertEqual(len(ok), 3, art.report)

    def test_violated_require_is_an_error(self):
        src = """
        component flex_suspension(L = 200 um, w = 4 um, n_beams = 1) {
          derive k.x = n_beams * (process.DEVICE.E * process.DEVICE.thickness * w^3) / L^3;
          geometry {
            beam(L, w, dir = y) at (0, 2 um - L/2);
            anchor(20 um, 20 um) at (0, 2 um - L - 8 um);
          }
        }
        device d {
          inst M = plate(100 um, 100 um) at (0, 0);
          inst S = array(flex_suspension(L = 200 um), count = 4, place = corners(M));
          net GND = M | S.fixed;
          require f_res(M, S) >= 1 MHz;   // impossible for this geometry
        }
        """
        art = compile_source(src)
        self.assertTrue(any("require violated" in e for e in art.errors),
                        art.errors)

    def test_devices_without_solve_unaffected(self):
        with open(os.path.join(EX, "accelerometer.soidl")) as f:
            art = compile_source(f.read())
        self.assertFalse([l for l in art.report if "closure" in l])
        self.assertEqual(art.errors, [])

    def test_fem_in_the_loop_calibration(self):
        """With --fem-closure the spec must be met by the FEM-predicted
        frequency, not the lumped estimate."""
        with open(os.path.join(EX, "comb_resonator.soidl")) as f:
            art = compile_source(f.read(), fem=True, fem_closure=True,
                                 fem_h=20.0)
        self.assertEqual(art.errors, [], art.errors)
        cal = [l for l in art.report if "FEM calibration" in l]
        self.assertTrue(cal, art.report)
        # final FEM verification: mode 1 within 2% of the 20 kHz target
        line = [l for l in art.report if "mode 1 vs lumped" in l][0]
        f_fem_khz = float(line.split("vs lumped f0: ")[1].split(" kHz")[0])
        self.assertLess(abs(f_fem_khz - 20.0) / 20.0, 0.02, line)
        # and the lumped f0 must now sit *below* target (retargeted)
        self.assertLess(art.model["f0"].value, 19.6e3)


class TestFEM(unittest.TestCase):
    """Validate the built-in plane-stress solver against beam theory."""

    E, NU, RHO, T = 169e9, 0.22, 2330.0, 25e-6
    L_UM, H_UM = 100.0, 10.0

    def _cantilever(self, h):
        from soidlc import fem2d
        shapes = [
            G.Shape("DEVICE", G.rect_corner(-20, 0, 20, self.H_UM),
                    "anchor", "anchored"),
            G.Shape("DEVICE", G.rect_corner(0, 0, self.L_UM, self.H_UM),
                    "beam", "released"),
        ]
        return fem2d.build_mesh(shapes, h)

    def test_cantilever_static_deflection(self):
        from soidlc import fem2d
        mesh = self._cantilever(2.5)
        tip = [n for n, (x, y) in enumerate(mesh.nodes)
               if abs(x - self.L_UM) < 1e-6 and n not in mesh.fixed]
        self.assertTrue(tip)
        F = 1e-6                                  # 1 uN, shared by tip nodes
        loads = {n: (0.0, F / len(tip)) for n in tip}
        u = fem2d.static_solve(mesh, self.E, self.NU, self.T, loads)
        dy = sum(u[n][1] for n in tip) / len(tip)
        L, hgt = self.L_UM * 1e-6, self.H_UM * 1e-6
        I = self.T * hgt ** 3 / 12.0
        euler = F * L ** 3 / (3.0 * self.E * I)
        self.assertLess(abs(dy - euler) / euler, 0.12,
                        f"FEM {dy:.3e} vs Euler {euler:.3e}")

    def test_cantilever_first_mode(self):
        from soidlc import fem2d
        mesh = self._cantilever(2.5)
        f = fem2d.modal(mesh, self.E, self.NU, self.RHO, self.T, n_modes=1)
        self.assertTrue(f)
        L, hgt = self.L_UM * 1e-6, self.H_UM * 1e-6
        I = self.T * hgt ** 3 / 12.0
        A = self.T * hgt
        analytic = (1.8751 ** 2 / (2 * math.pi * L ** 2)) \
            * math.sqrt(self.E * I / (self.RHO * A))
        self.assertLess(abs(f[0] - analytic) / analytic, 0.12,
                        f"FEM {f[0]:.0f} Hz vs analytic {analytic:.0f} Hz")

    def test_device_fem_end_to_end(self):
        with open(os.path.join(EX, "accelerometer.soidl")) as f:
            art = compile_source(f.read(), fem=True, fem_h=20.0)
        fem_lines = [l for l in art.report if l.startswith("fem")]
        self.assertTrue(fem_lines, art.report)
        # mode 1 must agree with the lumped estimate within 25%
        cmp = [l for l in fem_lines if "vs lumped" in l]
        self.assertTrue(cmp)
        pct = float(cmp[0].rsplit("(", 1)[1].rstrip("%)"))
        self.assertLess(abs(pct), 25.0, cmp[0])

    def test_deformation_plots_written(self):
        import struct
        import tempfile
        with open(os.path.join(EX, "accelerometer.soidl")) as f:
            src = f.read()
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "dev")
            art = compile_source(src, out_prefix=prefix, fem=True,
                                 fem_h=20.0)
            for key in ("fem_modes_1", "fem_3d_1"):
                path = art.files.get(key)
                self.assertTrue(path and os.path.exists(path), key)
                with open(path, "rb") as fh:
                    self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")
                    fh.read(8)
                    w, h = struct.unpack(">II", fh.read(8))
                self.assertGreater(w, 100)
                self.assertGreater(h, 100)


class TestROM(unittest.TestCase):
    """The generated behavioural models must agree with the FEM they came
    from: ring-down of the ODE model reproduces mode 1, and the SPICE BVD
    series resonance matches it."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        prefix = os.path.join(cls._tmp.name, "res")
        with open(os.path.join(EX, "comb_resonator.soidl")) as f:
            cls.art = compile_source(f.read(), out_prefix=prefix,
                                     fem=True, fem_h=20.0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _load_model(self):
        import importlib.util
        path = self.art.files["rom_py"]
        spec = importlib.util.spec_from_file_location("rom_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_ode_ringdown_matches_mode1(self):
        mod = self._load_model()
        f_est = mod.estimate_resonance()
        self.assertGreater(f_est, 0)
        self.assertLess(abs(f_est - mod.F_MODES[0]) / mod.F_MODES[0], 0.02)

    def test_ode_static_deflection_consistent(self):
        mod = self._load_model()
        F = mod.force(mod.DRIVE_PORT, mod.V_DC)
        x_static = abs(mod.freq_response(1.0)) * F
        self.assertLess(abs(x_static - F / mod.K_EFF) / (F / mod.K_EFF),
                        0.05)

    def test_spice_bvd_resonance(self):
        with open(self.art.files["rom_cir"]) as f:
            txt = f.read()
        Lm = float([l for l in txt.splitlines()
                    if l.startswith("Lm")][0].split()[-1])
        Cm = float([l for l in txt.splitlines()
                    if l.startswith("Cm")][0].split()[-1])
        f_series = 1.0 / (2 * math.pi * math.sqrt(Lm * Cm))
        mod = self._load_model()
        self.assertLess(abs(f_series - mod.F_MODES[0]) / mod.F_MODES[0],
                        0.01)

    def test_arw_in_require_and_report(self):
        ok = [l for l in self.art.report
              if l.startswith("require arw") and ": ok" in l]
        self.assertTrue(ok, self.art.report)
        rom = [l for l in self.art.report if "ARW (Brownian" in l]
        self.assertTrue(rom, self.art.report)
        # parse the deg/sqrt(h) value: plausible MEMS range in air
        val = float(rom[0].split(":")[1].split("deg")[0])
        self.assertGreater(val, 1e-3)
        self.assertLess(val, 1.0)

    def test_allan_experiment_matches_analytic_arw(self):
        mod = self._load_model()
        pts, est = mod.arw_experiment(n=60000, dt=1e-3, seed=4)
        self.assertLess(abs(est - mod.ARW_RADS) / mod.ARW_RADS, 0.15,
                        (est, mod.ARW_RADS))
        # white noise: sigma(tau) ~ tau^-1/2 over the first decades
        (t0, s0), (t1, s1) = pts[0], pts[6]
        slope = math.log(s1 / s0) / math.log(t1 / t0)
        self.assertLess(abs(slope + 0.5), 0.1, slope)

    def test_thermal_equipartition(self):
        # Langevin forcing and damping must satisfy <x^2> = kB*T/k
        mod = self._load_model()
        dt, xs = mod.simulate_thermal(0.2, n_modes=1, seed=5)
        rms = math.sqrt(sum(x * x for x in xs) / len(xs))
        expected = mod.thermal_x_rms()
        self.assertLess(abs(rms - expected) / expected, 0.5,
                        (rms, expected))

    def test_veriloga_well_formed(self):
        with open(self.art.files["rom_va"]) as f:
            txt = f.read()
        self.assertIn("module comb_resonator_rom", txt)
        self.assertIn("endmodule", txt)
        self.assertIn("DCDXD", txt)
        self.assertEqual(txt.count("ddt(V(q"), 3)   # one per mode


if __name__ == "__main__":
    unittest.main(verbosity=2)
