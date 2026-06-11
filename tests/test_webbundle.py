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


def test_bundle_modal_results():
    b = build_bundle("examples/comb_resonator.soidl", n_modes=2)
    res = b["results"]
    assert len(res) >= 1
    npos = len(b["geometry"]["positions"])
    r = res[0]
    assert r["type"] == "modal" and r["freq_hz"] > 0 and r["animate"] is True
    assert len(r["disp"]) == npos
    mags = [math.hypot(r["disp"][i], r["disp"][i+1])
            for i in range(0, npos, 3)]
    assert abs(max(mags) - 1.0) < 1e-6
    assert max(r["fields"]["disp_mag"]) <= 1.0 + 1e-9
    assert min(r["fields"]["disp_mag"]) >= 0.0
