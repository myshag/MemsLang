import pytest
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from webapp.backend.app import app

client = TestClient(app)


def test_examples_listed():
    r = client.get("/api/examples")
    assert r.status_code == 200
    names = [e["name"] for e in r.json()]
    assert "comb_resonator" in names


def test_bundle_ok():
    r = client.get("/api/bundle/comb_resonator")
    assert r.status_code == 200
    b = r.json()
    assert b["geometry"]["positions"] and b["results"]


def test_bundle_cached():
    import time
    t0 = time.time(); client.get("/api/bundle/comb_resonator"); t1 = time.time()
    assert t1 - t0 < 1.0          # second hit comes from cache


def test_bundle_unknown_404():
    assert client.get("/api/bundle/nope_not_real").status_code == 404


def test_source_endpoint():
    r = client.get("/api/source/comb_resonator")
    assert r.status_code == 200
    assert "process" in r.json()["source"] or len(r.json()["source"]) > 100


def test_source_unknown_404():
    assert client.get("/api/source/nope").status_code == 404


def test_compile_endpoint_ok():
    src = client.get("/api/source/comb_resonator").json()["source"]
    r = client.post("/api/compile", json={"source": src})
    assert r.status_code == 200
    assert r.json()["results"]


def test_compile_endpoint_broken_422():
    r = client.post("/api/compile", json={"source": "garbage %%%"})
    assert r.status_code == 422
    assert r.json()["detail"]


def test_bundle_cache_invalidated_by_mtime(tmp_path, monkeypatch):
    # point EXAMPLES at a temp dir with a copy; touch -> rebuild
    import shutil, time
    from webapp.backend import app as backend
    ex = tmp_path / "examples"; ex.mkdir()
    shutil.copy("examples/comb_resonator.soidl", ex / "tiny.soidl")
    monkeypatch.setattr(backend, "EXAMPLES", str(ex))
    backend._BUNDLES.clear()
    b1 = backend._bundle_for("tiny")
    t0 = time.time()
    b2 = backend._bundle_for("tiny")
    assert time.time() - t0 < 1.0 and b2 is b1     # cache hit
    time.sleep(0.02)
    (ex / "tiny.soidl").touch()
    b3 = backend._bundle_for("tiny")
    assert b3 is not b1                             # rebuilt after mtime change
