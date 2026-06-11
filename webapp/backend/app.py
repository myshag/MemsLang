"""FastAPI backend: lists examples, serves web-viewer bundles, serves the SPA."""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from soidlc import webbundle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(ROOT, "examples")
DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

app = FastAPI(title="soidlc viewer")


def _example_names():
    return sorted(f[:-6] for f in os.listdir(EXAMPLES) if f.endswith(".soidl"))


@app.get("/api/examples")
def examples():
    return [{"name": n, "title": n.replace("_", " ")} for n in _example_names()]


@lru_cache(maxsize=32)
def _cached_bundle(name: str) -> dict:
    return webbundle.build_bundle(os.path.join(EXAMPLES, f"{name}.soidl"))


@app.get("/api/bundle/{name}")
def bundle(name: str):
    if name not in _example_names():
        raise HTTPException(status_code=404, detail=f"unknown example {name}")
    try:
        return _cached_bundle(name)
    except Exception as e:  # surface compile/FEM failure, don't swallow
        raise HTTPException(status_code=422, detail=str(e))


if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="spa")
