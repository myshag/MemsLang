"""FastAPI backend: lists examples, serves web-viewer bundles, serves the SPA.

Compiles run in a throwaway child process: the FEM stack (scipy/ARPACK,
gmsh) can crash natively, and a segfault in-process would take the whole
server down. A fresh child per compile also keeps native-library state clean.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from soidlc import webbundle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(ROOT, "examples")
DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

app = FastAPI(title="soidlc viewer")

_BUNDLES: dict = {}            # name -> (mtime, bundle)


def _example_names():
    return sorted(f[:-6] for f in os.listdir(EXAMPLES) if f.endswith(".soidl"))


def _run_isolated(fn, *args) -> dict:
    """Run a compile in a single-use child process.

    A native crash (segfault in ARPACK/gmsh) only kills the child; we turn
    it into an HTTPException instead of losing the server.
    """
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=ctx) as pool:
        try:
            return pool.submit(fn, *args).result(timeout=300)
        except concurrent.futures.process.BrokenProcessPool:
            raise HTTPException(
                status_code=500,
                detail="compiler crashed (native error in FEM solver); "
                       "the server is still up — try again or simplify "
                       "the design")
        except concurrent.futures.TimeoutError:
            raise HTTPException(status_code=504,
                                detail="compile timed out (300s)")


def _bundle_for(name: str) -> dict:
    path = os.path.join(EXAMPLES, f"{name}.soidl")
    mtime = os.path.getmtime(path)
    hit = _BUNDLES.get(name)
    if hit and hit[0] == mtime:
        return hit[1]
    b = _run_isolated(webbundle.build_bundle, path)
    _BUNDLES[name] = (mtime, b)
    return b


@app.get("/api/examples")
def examples():
    return [{"name": n, "title": n.replace("_", " ")} for n in _example_names()]


@app.get("/api/bundle/{name}")
def bundle(name: str):
    if name not in _example_names():
        raise HTTPException(status_code=404, detail=f"unknown example {name}")
    try:
        return _bundle_for(name)
    except HTTPException:
        raise
    except Exception as e:  # surface compile/FEM failure, don't swallow
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/source/{name}")
def source(name: str):
    if name not in _example_names():
        raise HTTPException(status_code=404, detail=f"unknown example {name}")
    with open(os.path.join(EXAMPLES, f"{name}.soidl")) as f:
        return {"name": name, "source": f.read()}


class CompileRequest(BaseModel):
    source: str


@app.post("/api/compile")
def compile_source(req: CompileRequest):
    try:
        return _run_isolated(webbundle.build_bundle_from_source, req.source)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e) or repr(e))


if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="spa")
