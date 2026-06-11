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
