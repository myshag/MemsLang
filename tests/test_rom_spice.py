"""The exported SPICE netlist must be a netlist a simulator will actually run.

Nothing else checks this. reduce.py writes a .cir by string formatting, so a
malformed node list, a bad subckt header or a mistyped element would ship
silently -- the file looks like a model either way. These tests hand it to
ngspice and require the resonance back.

Skipped when ngspice is not installed; it is not a project dependency.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soidlc import compile_source                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")
NGSPICE = shutil.which("ngspice") or shutil.which(
    "/opt/homebrew/bin/ngspice")

TESTBENCH = """
* --- generated testbench ---------------------------------------------
Vin   in 0 DC 0 AC 1
Rs    in drive 50
X1    drive sense 0 {subckt}
Vamp  sense 0 DC 0
.control
  ac lin 2001 {f_lo:g} {f_hi:g}
  let mag = abs(i(vamp))
  meas ac ipk MAX mag
  meas ac ioff FIND mag AT={f_off:g}
  print ipk ioff
.endc
.end
"""


@unittest.skipIf(NGSPICE is None, "ngspice not installed")
class TestSpiceExportIsSimulatable(unittest.TestCase):
    EXAMPLE = "folded_flexure_resonator"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(EX, f"{cls.EXAMPLE}.soidl")) as f:
            cls.art = compile_source(
                f.read(), base_dir=EX, fem=True, fem_h=14.0,
                out_prefix=os.path.join(cls._tmp.name, "dut"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _cir(self):
        path = self.art.files.get("rom_cir")
        self.assertTrue(path and os.path.exists(path),
                        "no SPICE netlist was exported")
        return path

    def _fem_mode1_hz(self):
        line = next(l for l in self.art.report if "modes:" in l)
        first = re.search(r"([0-9.]+)\s*kHz", line)
        self.assertTrue(first, line)
        return float(first.group(1)) * 1e3

    def _subckt_name(self, text):
        m = re.search(r"^\.subckt\s+(\S+)", text, re.M)
        self.assertTrue(m, "netlist has no .subckt line")
        return m.group(1)

    def test_ngspice_accepts_the_netlist(self):
        """A syntax error here means the export ships an unusable file."""
        text = open(self._cir()).read()
        f0 = self._fem_mode1_hz()
        bench = text + TESTBENCH.format(
            subckt=self._subckt_name(text), f_lo=f0 * 0.99,
            f_hi=f0 * 1.01, f_off=f0 * 0.992)
        with tempfile.NamedTemporaryFile("w", suffix=".cir",
                                         delete=False) as fh:
            fh.write(bench)
            path = fh.name
        try:
            r = subprocess.run([NGSPICE, "-b", path], capture_output=True,
                               text=True, timeout=180)
        finally:
            os.unlink(path)
        combined = r.stdout + r.stderr
        self.assertNotIn("Error on line", combined, combined[-2000:])
        self.assertNotRegex(combined, r"(?i)\bfatal\b", combined[-2000:])

    def test_simulated_resonance_matches_the_fem_mode(self):
        """The whole chain in one assertion: FEM mode -> ROM -> netlist ->
        an independent circuit simulator."""
        text = open(self._cir()).read()
        f0 = self._fem_mode1_hz()
        bench = text + TESTBENCH.format(
            subckt=self._subckt_name(text), f_lo=f0 * 0.99,
            f_hi=f0 * 1.01, f_off=f0 * 0.992)
        with tempfile.NamedTemporaryFile("w", suffix=".cir",
                                         delete=False) as fh:
            fh.write(bench)
            path = fh.name
        try:
            r = subprocess.run([NGSPICE, "-b", path], capture_output=True,
                               text=True, timeout=180)
        finally:
            os.unlink(path)
        out = r.stdout + r.stderr
        m = re.search(r"ipk\s*=\s*([0-9.eE+-]+)\s*at=\s*([0-9.eE+-]+)", out)
        self.assertTrue(m, f"ngspice printed no measurement:\n{out[-2000:]}")
        i_pk, f_pk = float(m.group(1)), float(m.group(2))
        rel = abs(f_pk - f0) / f0
        self.assertLess(rel, 0.01,
                        f"ngspice peaked at {f_pk:.1f} Hz, FEM mode 1 is "
                        f"{f0:.1f} Hz ({rel * 100:.2f}%)")

        # There must be a RESONANCE, not merely a maximum.  A corrupted
        # motional branch still produces a peak -- at the sweep edge, barely
        # above the feedthrough floor -- and the frequency assertion above
        # only just catches it (1.07% against a 1% bound).  Measured
        # peak/off-resonance ratio at 0.992*f0:
        #     intact netlist            6.93
        #     Lm missing a node         1.02
        #     Cm wrong by 10x           1.02
        #     Rm wrong by 100x          1.00
        # A threshold of 3 sits 2.3x below the good case and 3x above every
        # broken one.
        off = re.search(r"ioff\s*=\s*([0-9.eE+-]+)", out)
        self.assertTrue(off, f"no off-resonance measurement:\n{out[-2000:]}")
        i_off = float(off.group(1))
        self.assertGreater(
            i_pk, 3.0 * i_off,
            f"peak {i_pk:.3e} A is only {i_pk / i_off:.1f}x the "
            f"off-resonance current {i_off:.3e} A — that is not a resonance")

    def test_the_netlist_states_its_bias(self):
        """A BVD branch is a small-signal model; without the bias it was
        linearised about, the numbers cannot be interpreted."""
        text = open(self._cir()).read()
        self.assertRegex(text, r"(?i)bias\s+V_dc\s*=")


if __name__ == "__main__":
    unittest.main()
