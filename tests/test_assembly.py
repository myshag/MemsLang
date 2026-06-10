import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soidlc import compile_source, assembly
from soidlc.assembly import Cap

EX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "examples")

class TestAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(EX, "comb_resonator.soidl")) as f:
            cls.art = compile_source(f.read())

    def test_cap_mesh_above_device(self):
        res, proc = self.art.result, self.art.process
        from soidlc import build3d
        dm = build3d.build_mesh(res, proc)
        full = assembly.assemble(dm, res, proc, Cap(gap=6))
        self.assertGreater(len(full.triangles), len(dm.triangles))
        # the cap sits above the device top
        (_, _, zmin), (_, _, zmax) = full.bounds()
        self.assertGreater(zmax, proc.device().z1 + 6)

    def test_vacuum_cap_raises_Q(self):
        res, proc, model = self.art.result, self.art.process, self.art.model
        q_open = assembly.capped_quality(res, proc, model,
                                         None)["uncapped_air"]
        q_vac = assembly.capped_quality(res, proc, model,
                                        Cap(gap=6, pressure=1.0))["capped"]
        self.assertGreater(q_vac, q_open)         # vacuum packaging helps

    def test_trapped_gas_cap_lowers_Q(self):
        res, proc, model = self.art.result, self.art.process, self.art.model
        P = 1.01325e5
        q_open = assembly.capped_quality(res, proc, model,
                                         None)["uncapped_air"]
        q_cap = assembly.capped_quality(res, proc, model,
                                        Cap(gap=6, pressure=P))["capped"]
        self.assertLess(q_cap, q_open)            # trapped-gas top film
        # smaller gap -> more squeeze film -> even lower Q
        q3 = assembly.capped_quality(res, proc, model,
                                     Cap(gap=3, pressure=P))["capped"]
        self.assertLess(q3, q_cap)

if __name__ == "__main__":
    unittest.main(verbosity=2)
