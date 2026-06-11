import json
import math
import subprocess
import sys

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


def test_web_cli_writes_json(tmp_path):
    out = tmp_path / "comb.web.json"
    subprocess.run([sys.executable, "-m", "soidlc.cli",
                    "examples/comb_resonator.soidl", "-o",
                    str(tmp_path / "comb"), "--web", str(out)],
                   check=True)
    b = json.loads(out.read_text())
    assert b["geometry"]["positions"] and b["results"]


def test_bundle_static_case():
    from soidlc.webbundle import _compile, _suspended_islands
    from soidlc import fem
    elab, result, mesh = _compile("examples/comb_resonator.soidl")
    _cid, ss = _suspended_islands(elab)[0]
    fm = fem.build_mesh(ss, 12.0)
    free = [n for n in range(len(fm.nodes)) if n not in fm.fixed]
    tip = max(free, key=lambda n: fm.nodes[n][0])
    b = build_bundle("examples/comb_resonator.soidl", n_modes=0,
                     static_cases=[{"label": "tip load",
                                    "forces": {tip: (0.0, -1.0)}}])
    st = [r for r in b["results"] if r["type"] == "static"]
    assert len(st) == 1 and st[0]["animate"] is False
    assert len(st[0]["disp"]) == len(b["geometry"]["positions"])


def test_static_layers_have_zero_disp():
    b = build_bundle("examples/comb_resonator.soidl", n_modes=1)
    g, meta = b["geometry"], b["meta"]
    names = [l["name"] for l in meta["layers"]]
    r = b["results"][0]
    bad = 0
    for v in range(len(g["vertexLayer"])):
        if names[g["vertexLayer"][v]] in ("BOX", "HANDLE"):
            if abs(r["disp"][3*v]) > 0 or abs(r["disp"][3*v+1]) > 0:
                bad += 1
    assert bad == 0


def test_bundle_from_source_roundtrip():
    from soidlc.webbundle import build_bundle_from_source
    src = open("examples/comb_resonator.soidl").read()
    b = build_bundle_from_source(src, n_modes=1)
    assert b["geometry"]["positions"] and b["results"]


def test_bundle_from_source_broken_raises():
    import pytest
    from soidlc.webbundle import build_bundle_from_source
    with pytest.raises(Exception):
        build_bundle_from_source("this is not valid soidl !!!")
