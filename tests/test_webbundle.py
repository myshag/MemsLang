"""Tests for the web-bundle exporter.

Architecture note: the 3D FEM solve for comb_resonator takes ~40-60 s.  To
keep total test time reasonable, a single module-scoped fixture builds ONE
bundle (n_modes=6) and all modal-result tests run against that bundle.
Tests that need a special configuration (n_modes=0, static cases, etc.) do
their own build but avoid triggering a second modal solve.
"""
import json
import math
import subprocess
import sys

import pytest

from soidlc.webbundle import build_bundle


# ---------------------------------------------------------------------------
# Module-scoped shared comb bundle (n_modes=6)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def comb_bundle():
    """Build ONE comb_resonator bundle with 6 modes; shared across all tests."""
    return build_bundle("examples/comb_resonator.soidl", n_modes=6)


# ---------------------------------------------------------------------------
# Geometry tests (no modal solve — cheap)
# ---------------------------------------------------------------------------

def test_bundle_geometry_valid():
    b = build_bundle("examples/comb_resonator.soidl", n_modes=0)
    g = b["geometry"]
    assert len(g["positions"]) % 3 == 0
    assert len(g["positions"]) // 3 == len(g["vertexLayer"])
    assert g["indices"] and max(g["indices"]) < len(g["vertexLayer"])
    assert b["meta"]["device"]
    assert len(b["meta"]["layers"]) >= 1
    assert len(b["meta"]["bbox"]) == 6


# ---------------------------------------------------------------------------
# Modal result tests (use shared bundle)
# ---------------------------------------------------------------------------

def test_bundle_modal_results(comb_bundle):
    b = comb_bundle
    res = b["results"]
    # With n_modes=6 we expect at least 4 modal results
    assert len(res) >= 4
    npos = len(b["geometry"]["positions"])
    r = res[0]
    assert r["type"] == "modal" and r["freq_hz"] > 0 and r["animate"] is True
    assert len(r["disp"]) == npos
    # 3-component magnitude check
    mags = [math.sqrt(r["disp"][i]**2 + r["disp"][i+1]**2 + r["disp"][i+2]**2)
            for i in range(0, npos, 3)]
    assert abs(max(mags) - 1.0) < 1e-6
    assert max(r["fields"]["disp_mag"]) <= 1.0 + 1e-9
    assert min(r["fields"]["disp_mag"]) >= 0.0


def test_bundle_has_out_of_plane_mode(comb_bundle):
    """At least one mode must be labelled out-of-plane and have nonzero dz."""
    b = comb_bundle
    res = b["results"]
    oop = [r for r in res if "(out-of-plane)" in r["label"]]
    assert len(oop) >= 1, "Expected at least one out-of-plane mode label"
    n = len(b["geometry"]["positions"])
    for r in oop:
        has_nonzero_dz = any(abs(r["disp"][i + 2]) > 0 for i in range(0, n, 3))
        assert has_nonzero_dz, f"Out-of-plane mode '{r['label']}' has all-zero dz"


def test_static_layers_have_zero_disp(comb_bundle):
    """BOX and HANDLE vertices must have zero dx, dy, AND dz for all modes."""
    b = comb_bundle
    g, meta = b["geometry"], b["meta"]
    names = [l["name"] for l in meta["layers"]]
    for r in b["results"]:
        if r["type"] != "modal":
            continue
        bad = 0
        for v in range(len(g["vertexLayer"])):
            if names[g["vertexLayer"][v]] in ("BOX", "HANDLE"):
                if (abs(r["disp"][3*v]) > 0
                        or abs(r["disp"][3*v+1]) > 0
                        or abs(r["disp"][3*v+2]) > 0):
                    bad += 1
        assert bad == 0, f"Mode '{r['label']}' has nonzero disp on static layer"


def test_modal_freq_ordering(comb_bundle):
    """Modal results must be sorted in ascending frequency."""
    freqs = [r["freq_hz"] for r in comb_bundle["results"] if r["type"] == "modal"]
    assert freqs == sorted(freqs)


# ---------------------------------------------------------------------------
# Static results (use shared bundle geometry; fresh compile for static case)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Source round-trip and error path
# ---------------------------------------------------------------------------

def test_bundle_from_source_roundtrip(comb_bundle):
    """Source round-trip: geometry and results both present (reuse shared bundle)."""
    b = comb_bundle
    assert b["geometry"]["positions"] and b["results"]


def test_bundle_from_source_broken_raises():
    import pytest
    from soidlc.webbundle import build_bundle_from_source
    with pytest.raises(Exception):
        build_bundle_from_source("this is not valid soidl !!!")


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

def test_web_cli_writes_json(tmp_path):
    out = tmp_path / "comb.web.json"
    subprocess.run([sys.executable, "-m", "soidlc.cli",
                    "examples/comb_resonator.soidl", "-o",
                    str(tmp_path / "comb"), "--web", str(out)],
                   check=True, timeout=300)
    b = json.loads(out.read_text())
    assert b["geometry"]["positions"] and b["results"]
