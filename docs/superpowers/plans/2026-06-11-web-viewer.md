# Web Viewer (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser app that loads a compiled soidlc device and post-processes its deformation results (animated modal shapes + static deformation), built on a universal `{geometry, results[]}` JSON contract.

**Architecture:** Python exporter (`soidlc/webbundle.py`) turns the existing 3D mesh + FEM results into a generic bundle; a thin FastAPI backend compiles examples and serves bundles + the SPA; a React + react-three-fiber frontend renders any bundle agnostically (morphs vertices for deformation, colours by a scalar field).

**Tech Stack:** Python (soidlc, FastAPI, uvicorn); React 18 + Vite + three + @react-three/fiber + @react-three/drei.

---

## Reference: existing soidlc API (read before starting)
- `from soidlc import compile_source, compile_file` → `Artifacts(process, result, mesh, report, warnings, errors, files, model)`.
- `art.mesh` is a `soidlc.mesh.Mesh`: `.vertices` (list of (x,y,z) floats, um), `.triangles` (list of (a,b,c) int), `.tri_group` (per-triangle layer-name string, parallel to `.triangles`).
- `build3d.layer_colors(process)` → `{layer_name: (r,g,b)}` floats 0..1.
- FEM: `from soidlc import fem`; `fem.build_mesh(shapes, h)` → FemMesh; `fem.modal(femmesh, E, nu, rho, t, n_modes)` → `(freqs_hz, vecs, dof_of)`; `fem.static_solve(femmesh, E, nu, t, forces)` → `{node:(ux,uy)}`.
- Suspended islands: `elab.islands` is a list of `(cid, shapes)`; a suspended island has at least one `s.mech=="anchored"` and one `s.mech=="released"` shape (mirror `soidlc/fem/analyze.py`).
- Device material: `dev = elab.process.device()`; `dev.E, dev.nu, dev.rho, dev.thickness` (thickness in um; pass `dev.thickness*1e-6` to FEM).
- `_node_disp(femmesh, vec, dof_of)` per-node (ux,uy) logic lives in `soidlc/femplot.py`; replicate it (don't import a private helper).

Run all python via `.venv/bin/python`. Frontend via `npm` inside `webapp/frontend`.

## File Structure
- Create `soidlc/webbundle.py` — the exporter (one public `build_bundle`).
- Modify `soidlc/cli.py` — add `--web <path>` flag.
- Create `webapp/backend/app.py` — FastAPI app.
- Create `webapp/frontend/` — Vite React app (`package.json`, `vite.config.js`, `index.html`, `src/main.jsx`, `src/App.jsx`, `src/DeviceCanvas.jsx`, `src/DeformableMesh.jsx`, `src/ResultPanel.jsx`, `src/Controls.jsx`, `src/ExampleSelector.jsx`, `src/lib/viridis.js`).
- Create `webapp/README.md`.
- Modify `pyproject.toml` — add optional `[project.optional-dependencies] web`.
- Create `tests/test_webbundle.py`, `tests/test_webapp_backend.py`.

---

## Task 1: Exporter — geometry + meta

**Files:** Create `soidlc/webbundle.py`; Test `tests/test_webbundle.py`.

- [ ] **Step 1: Failing test** — `tests/test_webbundle.py`:

```python
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
```

- [ ] **Step 2: Run, expect FAIL** (no module): `.venv/bin/python -m pytest tests/test_webbundle.py::test_bundle_geometry_valid -v`

- [ ] **Step 3: Implement geometry + meta** — `soidlc/webbundle.py`:

```python
"""Export a compiled soidlc device to a generic web-viewer JSON bundle.

Contract: { meta, geometry:{positions,indices,vertexLayer}, results:[...] }.
The bundle is device-agnostic; the frontend renders any result without
soidlc-specific knowledge.
"""
from __future__ import annotations

import math
from typing import List, Optional

from . import build3d
from .parser import parse
from .elaborate import Elaborator


def _compile(path: str, fem_h: float = 12.0):
    """Elaborate a .soidl file and build its 3D mesh; return (elab, result, mesh)."""
    with open(path) as f:
        ast = parse(f.read())
    elab = Elaborator(ast)
    from . import closure
    overrides = closure.run(elab, None, fem_calibrate=False, fem_h=20.0)
    result = elab.elaborate_device(None, overrides=overrides)
    mesh = build3d.build_mesh(result, elab.process, include_handle=True)
    return elab, result, mesh


def _geometry(mesh, process):
    colors = build3d.layer_colors(process)
    layer_names = sorted({g for g in mesh.tri_group})
    layer_idx = {name: i for i, name in enumerate(layer_names)}
    positions: List[float] = []
    for (x, y, z) in mesh.vertices:
        positions.extend((float(x), float(y), float(z)))
    indices: List[int] = []
    vertex_layer = [0] * len(mesh.vertices)
    seen = [False] * len(mesh.vertices)
    for (a, b, c), grp in zip(mesh.triangles, mesh.tri_group):
        indices.extend((a, b, c))
        li = layer_idx[grp]
        for v in (a, b, c):           # first-writer-wins per shared vertex
            if not seen[v]:
                vertex_layer[v] = li
                seen[v] = True
    layers = [{"name": n,
               "color": [float(c) for c in colors.get(n, (0.6, 0.6, 0.6))]}
              for n in layer_names]
    x0, y0, z0, x1, y1, z1 = mesh.bounds()
    meta = {"units": "um",
            "bbox": [x0, y0, z0, x1, y1, z1],
            "layers": layers}
    return {"positions": positions, "indices": indices,
            "vertexLayer": vertex_layer}, meta


def build_bundle(path: str, n_modes: int = 3,
                 static_cases: Optional[list] = None) -> dict:
    """Compile `path` and return the web-viewer bundle dict."""
    elab, result, mesh = _compile(path)
    geometry, meta = _geometry(mesh, elab.process)
    meta["device"] = getattr(getattr(elab, "device_ast", None), "name",
                             "device")
    f0 = result.model.get("f0")
    if f0 is not None and hasattr(f0, "value"):
        meta["lumped_f0_hz"] = f0.value
    results = _modal_results(elab, mesh, n_modes) if n_modes else []
    return {"meta": meta, "geometry": geometry, "results": results}


def _modal_results(elab, mesh, n_modes):   # filled in Task 2
    return []
```

- [ ] **Step 4: Run, expect PASS:** `.venv/bin/python -m pytest tests/test_webbundle.py::test_bundle_geometry_valid -v`

- [ ] **Step 5: Commit:**
```bash
git add soidlc/webbundle.py tests/test_webbundle.py
git commit -m "feat(web): web-bundle exporter — geometry + meta

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Exporter — modal results (per-vertex deformation)

**Files:** Modify `soidlc/webbundle.py`; Test `tests/test_webbundle.py`.

- [ ] **Step 1: Failing test** — append:

```python
def test_bundle_modal_results():
    b = build_bundle("examples/comb_resonator.soidl", n_modes=2)
    res = b["results"]
    assert len(res) >= 1
    npos = len(b["geometry"]["positions"])
    r = res[0]
    assert r["type"] == "modal" and r["freq_hz"] > 0 and r["animate"] is True
    assert len(r["disp"]) == npos                 # per-vertex (x,y,z)
    mags = [math.hypot(r["disp"][i], r["disp"][i+1])
            for i in range(0, npos, 3)]
    assert abs(max(mags) - 1.0) < 1e-6            # normalised to 1
    assert min(b["geometry"]["positions"]) is not None
    assert max(r["fields"]["disp_mag"]) <= 1.0 + 1e-9
    assert min(r["fields"]["disp_mag"]) >= 0.0    # static parts -> 0
```

(add `import math` at top of the test file if missing.)

- [ ] **Step 2: Run, expect FAIL** (results empty): `.venv/bin/python -m pytest tests/test_webbundle.py::test_bundle_modal_results -v`

- [ ] **Step 3: Implement `_modal_results`** — replace the stub:

```python
def _suspended_islands(elab):
    out = []
    for cid, ss in getattr(elab, "islands", []):
        if any(s.mech == "anchored" for s in ss) and \
           any(s.mech == "released" for s in ss):
            out.append((cid, ss))
    return out


def _nearest_disp(vx, vy, femmesh, vec, dof_of, max_dist):
    """In-plane (ux,uy) at the FEM node nearest (vx,vy), or (0,0) if the
    nearest node is farther than max_dist (vertex not on this island)."""
    best_d = None
    best = (0.0, 0.0)
    for n, (px, py) in enumerate(femmesh.nodes):
        d = (px - vx) ** 2 + (py - vy) ** 2
        if best_d is None or d < best_d:
            best_d = d
            base = dof_of.get(n, -1)
            best = (vec[base], vec[base + 1]) if base >= 0 else (0.0, 0.0)
    if best_d is None or best_d > max_dist * max_dist:
        return 0.0, 0.0
    return best


def _modal_results(elab, mesh, n_modes):
    dev = elab.process.device()
    if dev is None:
        return []
    E, nu, rho, t = dev.E, dev.nu, dev.rho, dev.thickness * 1e-6
    from . import fem
    verts = mesh.vertices
    out = []
    for cid, ss in _suspended_islands(elab):
        fm = fem.build_mesh(ss, 12.0)
        if not fm.cells or fm.n_cells > fem.MAX_ELEMENTS:
            continue
        freqs, vecs, dof_of = fem.modal(fm, E, nu, rho, t, n_modes=n_modes)
        # restrict displacement to vertices near this island's footprint
        xs = [p[0] for p in fm.nodes]
        ys = [p[1] for p in fm.nodes]
        max_dist = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0) + 12.0
        for mi, (f, vec) in enumerate(zip(freqs, vecs), 1):
            disp = []
            mags = []
            for (vx, vy, vz) in verts:
                ux, uy = _nearest_disp(vx, vy, fm, vec, dof_of, max_dist)
                disp.extend((ux, uy, 0.0))
                mags.append(math.hypot(ux, uy))
            mmax = max(mags) or 1.0
            disp = [d / mmax for d in disp]
            dmag = [m / mmax for m in mags]
            out.append({
                "type": "modal",
                "label": (f"Island {cid} — Mode {mi}"
                          if len(_suspended_islands(elab)) > 1
                          else f"Mode {mi}"),
                "freq_hz": float(f),
                "animate": True,
                "disp": disp,
                "dmax_um": float(mmax),
                "fields": {"disp_mag": dmag},
            })
    return out
```

- [ ] **Step 4: Run, expect PASS:** `.venv/bin/python -m pytest tests/test_webbundle.py -v`

- [ ] **Step 5: Commit:**
```bash
git add soidlc/webbundle.py tests/test_webbundle.py
git commit -m "feat(web): modal deformation results in web-bundle

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Exporter — static case + `--web` CLI flag

**Files:** Modify `soidlc/webbundle.py`, `soidlc/cli.py`; Test `tests/test_webbundle.py`.

- [ ] **Step 1: Failing test** — append:

```python
import json, subprocess, sys, os


def test_web_cli_writes_json(tmp_path):
    out = tmp_path / "comb.web.json"
    subprocess.run([sys.executable, "-m", "soidlc.cli",
                    "examples/comb_resonator.soidl", "-o",
                    str(tmp_path / "comb"), "--web", str(out)],
                   check=True)
    b = json.loads(out.read_text())
    assert b["geometry"]["positions"] and b["results"]
```

- [ ] **Step 2: Run, expect FAIL** (no `--web`): `.venv/bin/python -m pytest tests/test_webbundle.py::test_web_cli_writes_json -v`

- [ ] **Step 3a: Add CLI flag** — in `soidlc/cli.py`, add to the argparser (near the other flags):
```python
    ap.add_argument("--web", metavar="PATH",
                    help="also write a web-viewer JSON bundle to PATH")
```
and after the normal compile completes (after `art = compile_file(...)`), before returning:
```python
    if args.web:
        import json
        from . import webbundle
        bundle = webbundle.build_bundle(args.input, n_modes=3)
        with open(args.web, "w") as f:
            json.dump(bundle, f)
        print(f"web    {args.web}")
```
(Place this where other outputs are reported; keep it guarded by `args.web`.)

- [ ] **Step 3b: Optional static case support** — extend `build_bundle` so a caller can request a static tip-load case. Add to `build_bundle` after `results = _modal_results(...)`:
```python
    if static_cases:
        results += _static_results(elab, mesh, static_cases)
```
and implement (forces are `{node:(Fx,Fy)}` in N/m per spec; a case is a dict `{"label":str,"forces":{node:(fx,fy)}}` — but node indices are FEM-mesh-specific, so Phase-1 keeps `static_cases=None` from the CLI; this function exists for programmatic/test use):
```python
def _static_results(elab, mesh, cases):
    dev = elab.process.device()
    E, nu, t = dev.E, dev.nu, dev.thickness * 1e-6
    from . import fem
    out = []
    isl = _suspended_islands(elab)
    if not isl:
        return out
    _cid, ss = isl[0]
    fm = fem.build_mesh(ss, 12.0)
    for case in cases:
        disp_map = fem.static_solve(fm, E, nu, t, case["forces"])
        # reuse the same vertex-sampling path as modal by faking a (vec,dof_of):
        verts = mesh.vertices
        disp, mags = [], []
        for (vx, vy, vz) in verts:
            best_d, uxuy = None, (0.0, 0.0)
            for n, (px, py) in enumerate(fm.nodes):
                d = (px - vx) ** 2 + (py - vy) ** 2
                if best_d is None or d < best_d:
                    best_d, uxuy = d, disp_map.get(n, (0.0, 0.0))
            disp.extend((uxuy[0], uxuy[1], 0.0))
            mags.append(math.hypot(uxuy[0], uxuy[1]))
        mmax = max(mags) or 1.0
        out.append({"type": "static", "label": case["label"], "freq_hz": None,
                    "animate": False, "disp": [d / mmax for d in disp],
                    "dmax_um": float(mmax),
                    "fields": {"disp_mag": [m / mmax for m in mags]}})
    return out
```

- [ ] **Step 4: Run, expect PASS:** `.venv/bin/python -m pytest tests/test_webbundle.py -v`

- [ ] **Step 5: Commit:**
```bash
git add soidlc/webbundle.py soidlc/cli.py tests/test_webbundle.py
git commit -m "feat(web): static case support + --web CLI flag

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: FastAPI backend

**Files:** Create `webapp/backend/app.py`; Modify `pyproject.toml`; Test `tests/test_webapp_backend.py`.

- [ ] **Step 1: Add optional deps** — in `pyproject.toml` add:
```toml
[project.optional-dependencies]
web = ["fastapi>=0.110", "uvicorn[standard]>=0.29"]
```
Install: `.venv/bin/python -m pip install -e ".[web]"`

- [ ] **Step 2: Failing test** — `tests/test_webapp_backend.py`:
```python
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


def test_bundle_unknown_404():
    assert client.get("/api/bundle/nope_not_real").status_code == 404
```

- [ ] **Step 3: Run, expect FAIL** (no app): `.venv/bin/python -m pytest tests/test_webapp_backend.py -v`

- [ ] **Step 4: Implement** — `webapp/backend/app.py`:
```python
"""FastAPI backend: lists examples, serves web-viewer bundles, serves the SPA."""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from soidlc import webbundle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

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
```

- [ ] **Step 5: Run, expect PASS:** `.venv/bin/python -m pytest tests/test_webapp_backend.py -v`

- [ ] **Step 6: Commit:**
```bash
git add webapp/backend/app.py pyproject.toml tests/test_webapp_backend.py
git commit -m "feat(web): FastAPI backend serving examples + bundles

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Frontend scaffold (Vite + React + R3F)

**Files:** Create `webapp/frontend/{package.json,vite.config.js,index.html,src/main.jsx,src/App.jsx,src/lib/viridis.js}`.

- [ ] **Step 1: `webapp/frontend/package.json`:**
```json
{
  "name": "soidlc-viewer",
  "private": true,
  "type": "module",
  "scripts": { "dev": "vite", "build": "vite build", "preview": "vite preview" },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "three": "^0.169.0",
    "@react-three/fiber": "^8.17.0",
    "@react-three/drei": "^9.114.0"
  },
  "devDependencies": { "@vitejs/plugin-react": "^4.3.0", "vite": "^5.4.0" }
}
```

- [ ] **Step 2: `webapp/frontend/vite.config.js`** (proxy /api to backend):
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
```

- [ ] **Step 3: `index.html`:**
```html
<!doctype html>
<html><head><meta charset="utf-8"/><title>soidlc viewer</title>
<style>html,body,#root{height:100%;margin:0;font-family:system-ui,sans-serif}</style>
</head><body><div id="root"></div>
<script type="module" src="/src/main.jsx"></script></body></html>
```

- [ ] **Step 4: `src/main.jsx`:**
```javascript
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
createRoot(document.getElementById('root')).render(<App />)
```

- [ ] **Step 5: `src/lib/viridis.js`** (compact viridis ramp):
```javascript
// 8-stop viridis approximation; t in [0,1] -> [r,g,b] 0..1
const STOPS = [
  [0.267,0.005,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],
  [0.207,0.372,0.553],[0.164,0.471,0.558],[0.128,0.567,0.551],
  [0.135,0.659,0.518],[0.267,0.749,0.441],[0.478,0.821,0.318],
  [0.741,0.873,0.150],[0.993,0.906,0.144],
]
export function viridis(t) {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1)
  const i = Math.floor(t), f = t - i
  const a = STOPS[i], b = STOPS[Math.min(i + 1, STOPS.length - 1)]
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f]
}
```

- [ ] **Step 6: minimal `src/App.jsx`** (fetch examples; full UI in Task 7):
```javascript
import React, { useEffect, useState } from 'react'

export default function App() {
  const [examples, setExamples] = useState([])
  useEffect(() => { fetch('/api/examples').then(r => r.json()).then(setExamples) }, [])
  return <div style={{padding:20}}>
    <h2>soidlc viewer</h2>
    <ul>{examples.map(e => <li key={e.name}>{e.title}</li>)}</ul>
  </div>
}
```

- [ ] **Step 7: Install + build** (verifies the toolchain):
```bash
cd webapp/frontend && npm install && npm run build
```
Expected: `dist/` produced, no build errors. (If `npm` is unavailable, report BLOCKED.)

- [ ] **Step 8: Commit:**
```bash
git add webapp/frontend/package.json webapp/frontend/vite.config.js webapp/frontend/index.html webapp/frontend/src
git commit -m "feat(web): Vite+React frontend scaffold

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: DeviceCanvas + DeformableMesh (undeformed render)

**Files:** Create `webapp/frontend/src/{DeviceCanvas.jsx,DeformableMesh.jsx}`; update `App.jsx`.

- [ ] **Step 1: `src/DeformableMesh.jsx`** — builds geometry, applies deformation+colour each frame:
```javascript
import React, { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { viridis } from './lib/viridis.js'

export default function DeformableMesh({ geometry, result, settings }) {
  const meshRef = useRef()
  const base = useMemo(() => Float32Array.from(geometry.positions), [geometry])
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(base.slice(), 3))
    g.setAttribute('color', new THREE.BufferAttribute(
      new Float32Array(base.length), 3))
    g.setIndex(geometry.indices)
    g.computeVertexNormals()
    return g
  }, [geometry, base])

  const layerColor = useMemo(() => {
    const lc = settings.layers.map(l => l.color)
    return geometry.vertexLayer.map(i => lc[i] || [0.6,0.6,0.6])
  }, [geometry, settings.layers])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    const amp = (result && result.animate)
      ? Math.sin(2*Math.PI*settings.speed*t) : 1
    const s = settings.scale * amp
    const pos = geom.attributes.position.array
    const col = geom.attributes.color.array
    const disp = result ? result.disp : null
    const mag = result ? result.fields.disp_mag : null
    for (let v = 0; v < base.length/3; v++) {
      const k = v*3
      if (disp) { pos[k]=base[k]+disp[k]*s; pos[k+1]=base[k+1]+disp[k+1]*s; pos[k+2]=base[k+2]+disp[k+2]*s }
      else { pos[k]=base[k]; pos[k+1]=base[k+1]; pos[k+2]=base[k+2] }
      let c = layerColor[v]
      if (disp && settings.colorField && mag) {
        const vc = viridis(mag[v] * Math.abs(amp))
        c = mag[v] > 1e-6 ? vc : c
      }
      col[k]=c[0]; col[k+1]=c[1]; col[k+2]=c[2]
    }
    geom.attributes.position.needsUpdate = true
    geom.attributes.color.needsUpdate = true
    geom.computeVertexNormals()
  })

  return <mesh ref={meshRef} geometry={geom}>
    <meshStandardMaterial vertexColors wireframe={settings.wireframe}
      flatShading metalness={0.1} roughness={0.8} side={THREE.DoubleSide}/>
  </mesh>
}
```

- [ ] **Step 2: `src/DeviceCanvas.jsx`:**
```javascript
import React from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Bounds } from '@react-three/drei'
import DeformableMesh from './DeformableMesh.jsx'

export default function DeviceCanvas({ bundle, result, settings }) {
  return <Canvas camera={{ position: [400, 400, 600], far: 100000 }}
                 style={{ background: '#0b0d12' }}>
    <ambientLight intensity={0.6}/>
    <directionalLight position={[1, 2, 3]} intensity={1.0}/>
    <directionalLight position={[-2, -1, -1]} intensity={0.3}/>
    <Bounds fit clip observe margin={1.2}>
      <DeformableMesh geometry={bundle.geometry} result={result}
        settings={{ ...settings, layers: bundle.meta.layers }}/>
    </Bounds>
    <OrbitControls makeDefault/>
  </Canvas>
}
```

- [ ] **Step 3: Wire into `App.jsx`** so selecting an example loads `/api/bundle/{name}` and shows the canvas with default settings (`{scale: 40, speed: 1.5, playing: true, wireframe: false, colorField: 'disp_mag'}`) and no result selected yet (undeformed). Verify `npm run build` passes.

- [ ] **Step 4: Commit:**
```bash
git add webapp/frontend/src
git commit -m "feat(web): R3F canvas + deformable mesh (undeformed render)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Result panel, controls, animation wiring

**Files:** Create `src/{ResultPanel.jsx,Controls.jsx,ExampleSelector.jsx}`; finish `App.jsx`.

- [ ] **Step 1: `src/ResultPanel.jsx`:**
```javascript
import React from 'react'
export default function ResultPanel({ results, active, onSelect }) {
  return <div style={{minWidth:220}}>
    <h4>Results</h4>
    <button onClick={() => onSelect(-1)}
      style={{fontWeight: active===-1?'bold':'normal'}}>Undeformed</button>
    {results.map((r, i) =>
      <div key={i} onClick={() => onSelect(i)}
        style={{cursor:'pointer', padding:'4px 0',
                fontWeight: active===i?'bold':'normal'}}>
        {r.label}{r.freq_hz ? ` — ${(r.freq_hz/1e3).toFixed(2)} kHz` : ''}
      </div>)}
  </div>
}
```

- [ ] **Step 2: `src/Controls.jsx`** — sliders/toggles bound to settings:
```javascript
import React from 'react'
export default function Controls({ settings, set, result }) {
  const f = (k) => (e) => set({ ...settings, [k]: parseFloat(e.target.value) })
  const b = (k) => () => set({ ...settings, [k]: !settings[k] })
  return <div style={{display:'flex', gap:16, alignItems:'center',
                      flexWrap:'wrap', padding:8, color:'#ddd'}}>
    <label>scale {settings.scale}
      <input type="range" min="0" max="200" value={settings.scale} onChange={f('scale')}/></label>
    <label>speed {settings.speed}
      <input type="range" min="0" max="5" step="0.1" value={settings.speed} onChange={f('speed')}/></label>
    <label><input type="checkbox" checked={settings.wireframe} onChange={b('wireframe')}/> wireframe</label>
    {result && <span>dmax = {result.dmax_um?.toExponential(2)} um</span>}
  </div>
}
```

- [ ] **Step 3: `src/ExampleSelector.jsx`:**
```javascript
import React from 'react'
export default function ExampleSelector({ examples, value, onChange }) {
  return <select value={value || ''} onChange={e => onChange(e.target.value)}>
    <option value="" disabled>Choose a device…</option>
    {examples.map(e => <option key={e.name} value={e.name}>{e.title}</option>)}
  </select>
}
```

- [ ] **Step 4: Final `src/App.jsx`** — compose everything:
```javascript
import React, { useEffect, useState } from 'react'
import DeviceCanvas from './DeviceCanvas.jsx'
import ResultPanel from './ResultPanel.jsx'
import Controls from './Controls.jsx'
import ExampleSelector from './ExampleSelector.jsx'

const DEFAULTS = { scale: 40, speed: 1.5, wireframe: false, colorField: 'disp_mag' }

export default function App() {
  const [examples, setExamples] = useState([])
  const [name, setName] = useState('')
  const [bundle, setBundle] = useState(null)
  const [active, setActive] = useState(-1)
  const [settings, setSettings] = useState(DEFAULTS)

  useEffect(() => { fetch('/api/examples').then(r => r.json()).then(setExamples) }, [])
  useEffect(() => {
    if (!name) return
    setBundle(null); setActive(-1)
    fetch(`/api/bundle/${name}`).then(r => r.json()).then(setBundle)
  }, [name])

  const result = bundle && active >= 0 ? bundle.results[active] : null
  return <div style={{display:'grid', gridTemplateRows:'auto 1fr auto', height:'100%', background:'#0b0d12', color:'#ddd'}}>
    <div style={{display:'flex', gap:16, alignItems:'center', padding:8}}>
      <strong>soidlc viewer</strong>
      <ExampleSelector examples={examples} value={name} onChange={setName}/>
    </div>
    <div style={{display:'grid', gridTemplateColumns:'240px 1fr', minHeight:0}}>
      <div style={{padding:8, overflow:'auto'}}>
        {bundle && <ResultPanel results={bundle.results} active={active} onSelect={setActive}/>}
        {bundle && <Legend layers={bundle.meta.layers}/>}
      </div>
      <div style={{minHeight:0}}>
        {bundle && <DeviceCanvas bundle={bundle} result={result} settings={settings}/>}
      </div>
    </div>
    <Controls settings={settings} set={setSettings} result={result}/>
  </div>
}

function Legend({ layers }) {
  return <div style={{marginTop:16, fontSize:12}}>
    {layers.map((l, i) => <div key={i} style={{display:'flex', gap:6, alignItems:'center'}}>
      <span style={{width:12, height:12, background:`rgb(${l.color.map(c=>Math.round(c*255)).join(',')})`}}/>
      {l.name}</div>)}
  </div>
}
```

- [ ] **Step 5: Build:** `cd webapp/frontend && npm run build` — expect success, no errors.

- [ ] **Step 6: Commit:**
```bash
git add webapp/frontend/src
git commit -m "feat(web): result panel, controls, animation wiring

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: End-to-end run, README, polish

**Files:** Create `webapp/README.md`; final verification.

- [ ] **Step 1: `webapp/README.md`:**
```markdown
# soidlc web viewer (phase 1)

Interactive 3D + FEM deformation post-processor for compiled soidlc devices.

## Run (dev)
1. Backend: `.venv/bin/python -m pip install -e ".[web]"`
   then `.venv/bin/uvicorn webapp.backend.app:app --reload` (serves :8000)
2. Frontend: `cd webapp/frontend && npm install && npm run dev` (serves :5173, proxies /api)
3. Open http://localhost:5173 , pick a device, select a mode, hit play.

## Build (prod)
`cd webapp/frontend && npm run build` then run only the backend; it serves the built SPA at :8000.
```

- [ ] **Step 2: End-to-end smoke** — start backend, curl a bundle, confirm it serves:
```bash
.venv/bin/uvicorn webapp.backend.app:app --port 8000 &
sleep 3
curl -s localhost:8000/api/examples | head -c 200
curl -s localhost:8000/api/bundle/comb_resonator | python -c "import sys,json; b=json.load(sys.stdin); print('verts', len(b['geometry']['positions'])//3, 'results', len(b['results']))"
kill %1
```
Expected: examples list + `verts <N> results <M>` with M>=1.

- [ ] **Step 3: Full python test suite:** `.venv/bin/python -m pytest -q` — expect all green (existing + new webbundle/backend tests).

- [ ] **Step 4: Commit:**
```bash
git add webapp/README.md
git commit -m "docs(web): viewer README + phase-1 complete

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer
- Run python via `.venv/bin/python`; never pip globally.
- The frontend needs `npm`. If unavailable, implement all files but report the build step as BLOCKED — the Python side (Tasks 1-4) is independently testable and valuable.
- The animation/colour code in `DeformableMesh` is the one piece to verify visually ("fix in reality"); tests only assert it builds and mounts. Keep per-frame work O(vertices) — the example meshes are a few thousand vertices.
- Do not add features beyond the spec (no upload, no compile-on-demand, no stress field) — those are later phases.
```
