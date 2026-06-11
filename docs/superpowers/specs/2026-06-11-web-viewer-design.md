# Web Viewer — Phase 1: 3D + universal FEM/deformation post-processor

**Date:** 2026-06-11
**Status:** Design — pending user review
**Part of:** the "open MEMS suite" roadmap (web front-end; see memory `memslang-north-star`). This is the FIRST web sub-project.

## 1. Goal

A browser app that loads a compiled soidlc device and lets the user inspect it
in interactive 3D and **post-process its deformation results** — modal mode
shapes (animated) and static load cases (deformed state) — colour-mapped by a
scalar field. The viewer is **universal over soidlc results**: it consumes a
generic `{ geometry, results[] }` bundle and renders any result type
(modal/static, and future stress/thermal/electrostatic fields) without
device-specific code.

### Scope (Phase 1)
- Backend compiles a chosen example with FEM and serves a JSON **web-bundle**.
- Frontend: 3D viewer (orbit), result selector, animated modal shapes, static
  deformed shapes, displacement colour-map, deformation-scale slider,
  play/pause, layer legend, wireframe toggle.
- Multiple suspended islands per device supported.

### Non-goals (explicit YAGNI for Phase 1)
- **No compile-on-demand from user-edited source** — that is Phase 2 (the
  agreed "both in phases" split). Phase 1 serves a fixed set of bundled
  examples (backend may compile them lazily and cache).
- **No ingestion of standard formats** (glTF/STL/VTU) and **no browser file
  upload** — explicitly deferred (user chose "generic over soidlc results").
- No editor, no SOIDL syntax highlighting, no auth, no persistence/DB.
- No stress field yet (the contract reserves a slot for it; the FEM does not
  compute von Mises today — `fields` simply omits it until a later milestone).

## 2. Architecture

```
soidlc (Python)                backend (FastAPI)              frontend (React + R3F)
 webbundle exporter   ──▶   GET /api/examples           ──▶   <App>
 (new: soidlc/webbundle.py)  GET /api/bundle/{name}            ├─ <DeviceCanvas> (R3F)
                             serves built SPA + bundles         ├─ <ResultPanel>
                                                                └─ <Controls>
```

Three independently testable units:
1. **Exporter** (Python) — turns the existing 3D mesh + FEM results into the
   web-bundle JSON. Pure data transform; unit-testable without a browser.
2. **Backend** (FastAPI) — lists examples, returns bundles (compile+cache),
   serves the built SPA. Thin; all heavy work is in `soidlc`.
3. **Frontend** (React + R3F) — agnostic renderer of `{ geometry, results[] }`.

Data flows one way: soidlc → bundle JSON → backend → frontend. The bundle is
the only contract between Python and JS.

## 3. Data contract — the web-bundle JSON (the universal interface)

```jsonc
{
  "meta": {
    "device": "comb_resonator",
    "units": "um",
    "bbox": [x0, y0, z0, x1, y1, z1],
    "layers": [ { "name": "DEVICE", "color": [0.2,0.4,0.9] }, ... ],
    "lumped_f0_hz": 20000            // optional, for comparison display
  },
  "geometry": {
    "positions": [x,y,z, ...],       // flat float32, model coords (um)
    "indices":   [i,j,k, ...],       // triangle vertex indices
    "vertexLayer": [layerIdx, ...]   // one layer index per vertex (for colour)
  },
  "results": [
    {
      "type": "modal",              // "modal" | "static" | (future) "field"
      "label": "Mode 1",
      "freq_hz": 21490,             // present for modal; null otherwise
      "animate": true,              // modal -> oscillate; static -> false
      "disp": [dx,dy,dz, ...],      // per-vertex, SAME order/length as positions,
                                    //   normalised so max|disp| == 1 (unitless)
      "dmax_um": 0.012,             // physical max displacement at unit amplitude
      "fields": {                   // per-vertex scalar fields for colour-mapping
        "disp_mag": [m, ...]        //   normalised 0..1 (always present)
        // "stress_vm": [...]       //   reserved; omitted until FEM computes it
      }
    }
    // ... more modes, and static cases when present
  ]
}
```

Key properties:
- **Universality:** the frontend never references "mode" or "device" specially.
  It renders `geometry`, lists `results[].label`, animates when `animate` is
  true, and colours by a chosen key in `fields`. New physics = new `results`
  entries / `fields` keys; no viewer change.
- **Static parts** (HANDLE substrate, anchored silicon) have `disp = 0` and
  `disp_mag = 0`, so they stay put and render neutral.
- **Animation:** modal → `pos(t) = base + disp · scale · sin(2π·speed·t)`;
  static → `pos = base + disp · scale` (no time term).
- `disp` is normalised (max magnitude 1) so the scale slider is device- and
  result-independent; `dmax_um` carries the real number for the legend.

## 4. Exporter — `soidlc/webbundle.py`

Single new module. One public function:

```python
def build_bundle(elab, art, n_modes: int = 3,
                 static_cases: list | None = None) -> dict
```

- **Geometry:** reuse the existing 3D mesh that `soidlc/exporters.py` writes for
  STL/OBJ (a `soidlc.mesh.Mesh` with `.vertices` Pt3, `.triangles` (a,b,c),
  `.tri_group` per-triangle layer name). Flatten to `positions`/`indices`;
  derive `vertexLayer` from `tri_group` (first-writer-wins per shared vertex);
  `layers[].color` from `elab.process.layers[name].color`.
- **Results:** for each suspended island, run `fem.build_mesh` + `fem.modal`
  (reusing `soidlc/fem`); for each mode build per-3D-vertex `disp` by sampling
  the in-plane FEM displacement at the nearest FEM node (the exact technique in
  `femplot.render_deformed_3d`/`_node_disp`), `dz = 0`. Normalise to max
  magnitude 1, record `dmax_um`. `disp_mag` = per-vertex `hypot(ux,uy)`
  normalised 0..1.
- **Static cases (optional):** if `static_cases` is given (e.g. a tip load), run
  `fem.static_solve` and emit a `type:"static"` result the same way.
- Returns a plain dict (JSON-serialisable). A thin CLI flag `--web <path>`
  (in `cli.py`) writes it to disk for offline/testing use.

This keeps all soidlc-specific knowledge in Python; the bundle is generic.

## 5. Backend — FastAPI (`webapp/backend/`)

- `app.py`:
  - `GET /api/examples` → `[{ "name": "comb_resonator", "title": "..." }, ...]`
    (derived from `examples/*.soidl`).
  - `GET /api/bundle/{name}` → the web-bundle dict (JSON). Compiles the example
    via `soidlc` with FEM, calls `webbundle.build_bundle`, caches the result in
    memory (and/or on disk) keyed by name so repeat loads are instant.
  - In production, serves the built SPA (`webapp/frontend/dist`) as static files
    at `/`. In dev, the Vite dev server proxies `/api` to this backend.
- Errors: a compile/FEM failure returns HTTP 422 with the soidlc
  error/warning report in the body (surfaced in the UI, not swallowed).
- Deps: `fastapi`, `uvicorn[standard]`. Backend is a separate optional install
  (`pip install -e ".[web]"`) so the core compiler stays dependency-light.

## 6. Frontend — React + Vite + react-three-fiber (`webapp/frontend/`)

Components (each one responsibility):
- **`<App>`** — fetches `/api/examples`, holds selected example + selected
  result index + view settings (scale, speed, playing, colourField, wireframe).
- **`<DeviceCanvas>`** — R3F `<Canvas>` with `OrbitControls`, lighting, and a
  `<DeformableMesh>`. Agnostic: receives `geometry` + the active `result` +
  settings.
- **`<DeformableMesh>`** — builds a `BufferGeometry` once from
  `geometry.positions`/`indices`; per-vertex base colour from `vertexLayer` →
  `layers[].color`. On each frame (`useFrame`): writes
  `position = base + disp·scale·(animate ? sin(2π·speed·t) : 1)` into the
  position attribute; sets vertex colours by blending the layer colour with the
  chosen `fields[colourField]` through a viridis ramp (precomputed per result,
  scaled live by the magnitude factor). Marks attributes `needsUpdate`.
- **`<ResultPanel>`** — lists `results[]` with label + `freq_hz` (kHz);
  selecting one sets the active result.
- **`<Controls>`** — deformation-scale slider, animation speed, play/pause,
  colour-field dropdown (keys of `fields`), wireframe toggle, reset-view,
  layer-colour legend, and a `dmax_um` readout.
- **`<ExampleSelector>`** — dropdown bound to `/api/examples`.

Stack: React 18, Vite, `three`, `@react-three/fiber`, `@react-three/drei`
(OrbitControls/Bounds/Html). No state library needed (React state suffices).

## 7. Repo layout & dev workflow

```
soidlc/webbundle.py              # exporter (Python)
webapp/
  backend/app.py                 # FastAPI
  backend/requirements.txt
  frontend/                      # Vite + React app
    src/{App,DeviceCanvas,DeformableMesh,ResultPanel,Controls,ExampleSelector}.jsx
    src/lib/{bundle.js,viridis.js}
    package.json, vite.config.js  # /api proxied to :8000 in dev
  README.md                      # how to run
```

Dev: `uvicorn webapp.backend.app:app --reload` (:8000) + `npm run dev` in
`webapp/frontend` (:5173, proxies `/api`). Prod: `npm run build` → FastAPI
serves `dist/`.

## 8. Testing

- **Exporter (pytest, `tests/test_webbundle.py`):** for each example, the bundle
  validates structurally — `positions` length %3==0, every `indices` entry in
  range, `len(vertexLayer)*3 == len(positions)`, each `results[i].disp` length
  == `positions` length, `disp_mag` normalised to [0,1], modal results carry a
  positive `freq_hz`, static parts have zero disp. A golden mode-1 frequency
  for `comb_resonator` is within tolerance of the FEM value.
- **Backend (pytest + FastAPI TestClient):** `/api/examples` non-empty;
  `/api/bundle/comb_resonator` returns 200 with a schema-valid bundle; an
  unknown name returns 404; a deliberately broken source returns 422 with a
  report body.
- **Frontend (lightweight):** a Vitest + React Testing Library smoke test that
  `<DeformableMesh>` mounts with a tiny fixture bundle without throwing, and
  that `<ResultPanel>` renders one row per result. (Full WebGL rendering is not
  asserted.)

## 9. Risks / open points

- **3D-vertex → FEM-node sampling cost:** nearest-node lookup per 3D vertex per
  mode is O(V·N). For the example meshes (a few thousand each) this is fine; if
  it becomes slow, bucket FEM nodes on a grid. (Settle in the plan.)
- **Shared vertices across layers:** `vertexLayer` uses first-writer-wins; at
  layer interfaces a vertex may pick either side. Acceptable for colour; if it
  looks wrong, expand to non-indexed geometry in the exporter.
- **Bundle size:** per-vertex disp for N modes ≈ N×geometry. A few thousand
  vertices × 3 modes is well under a megabyte; fine over localhost.
- **Frontend dependency footprint** (node/Vite/React) is new to this repo;
  isolated under `webapp/frontend/` and not required to run the core compiler.
```
