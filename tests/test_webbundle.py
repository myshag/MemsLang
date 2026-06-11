import math
from soidlc.webbundle import build_bundle


def test_bundle_geometry_valid():
    b = build_bundle("examples/comb_resonator.soidl", n_modes=0)
    g = b["geometry"]
    assert len(g["positions"]) % 3 == 0
    assert len(g["positions"]) // 3 == len(g["vertexLayer"])
    assert g["indices"] and max(g["indices"]) < len(g["vertexLayer"])
    assert b["meta"]["device"]
    assert len(b["meta"]["layers"]) >= 1
    assert len(b["meta"]["bbox"]) == 6
