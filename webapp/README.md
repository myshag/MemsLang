# soidlc web viewer (phase 1)

Interactive 3D + FEM deformation post-processor for compiled soidlc devices:
pick a device, select a mode (or static case), watch the animated mode shape
colour-mapped by displacement.

## Run (dev)

1. Backend (from the repo root):

   ```bash
   .venv/bin/python -m pip install -e ".[web]"
   .venv/bin/uvicorn webapp.backend.app:app --reload   # serves :8000
   ```

2. Frontend:

   ```bash
   cd webapp/frontend && npm install && npm run dev    # serves :5173, proxies /api
   ```

3. Open http://localhost:5173 , pick a device, select a mode, adjust scale.

## Run (prod)

```bash
cd webapp/frontend && npm run build
.venv/bin/uvicorn webapp.backend.app:app              # serves SPA + API at :8000
```

## How it works

- `soidlc/webbundle.py` exports a compiled device as a universal JSON bundle
  `{meta, geometry, results[]}` (also available from the CLI: `--web out.json`).
- `webapp/backend/app.py` (FastAPI) compiles examples on demand, caches
  bundles, serves the SPA.
- `webapp/frontend/` (React + react-three-fiber) renders any bundle
  agnostically: modal results oscillate, static results show the deformed
  state; colour = viridis over `fields.disp_mag`.
