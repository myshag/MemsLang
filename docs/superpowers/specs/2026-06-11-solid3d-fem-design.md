# Solid 3D FEM — volumetric modal mechanics for the viewer

**Date:** 2026-06-11
**Status:** Approved (user: "да"); PoC validated in-session
**Roadmap:** advances M1's solver into 3D for visualization; closure/ROM keep 2D

## Goal
Replace the plane-stress (2D) modal results in the **web bundle** with true 3D
solid mechanics so out-of-plane and torsional modes become visible in the
viewer. PoC on the real comb island: 3D finds modes 21.4 / 116.1 / 185.1 /
389.4 / 453.5 kHz where 2D saw only 21.5 / 447.8 — three out-of-plane modes
were invisible to 2D. Mode 1 agrees with 2D within 0.5% (mutual validation).

## Scope
1. **`soidlc/fem/solid3d.py`** (new):
   - `mesh_island_3d(shapes, thickness_um, h=10.0, lump_label="finger")` →
     `Solid3DMesh` (nodes um ×3, tets, fixed node mask, lumped finger masses).
     gmsh OCC: per-solid plane surface → extrude by thickness → fuse → tets at
     `Mesh.MeshSizeMax=h`. Fingers are NOT meshed (2 µm features would explode
     the tet count) — lumped as point mass on the nearest node, like 2D.
   - `modal3d(mesh, E, nu, rho, n_modes=6)` → `(freqs, vecs, dof_of)` with
     per-node `(ux,uy,uz)`; TetP2 vector elements, **3D Lame**
     (`lam = E*nu/((1+nu)(1-2nu))`), eigsh `sigma=0`; clamp = all dofs of
     nodes inside anchored-shape bboxes (full z-column). M-normalised in the
     vertex (P1) metric like the 2D solver.
2. **Exporter** (`soidlc/webbundle.py`): bundle results computed by the 3D
   solver (replaces 2D in this path). Displacement sampling per render vertex:
   nearest 3D FEM node via cKDTree **in xyz** (FEM z is 0..t local — offset by
   the DEVICE layer z0 from the process so coordinates match the render mesh).
   `disp` now carries real dz. BOX/HANDLE static mask unchanged. Mode labels:
   append "(out-of-plane)" when mean|dz| > mean in-plane magnitude over
   unmasked vertices. `n_modes` default 6 for bundles.
3. **Viewer**: no changes required (contract already carries dz).
4. **2D solver remains** for design-closure and ROM (speed: hundreds of
   evaluations). CLI `--fem` report also stays 2D in this phase.

## Non-goals
- No 3D static path in the bundle yet; no anisotropic silicon (isotropic E,nu
  as today); no remeshing of fingers; no closure on 3D.

## Performance budget
comb island: ~6k nodes / 19k tets / 110k dofs → ~37 s (assembly 6 s + eigsh
30 s). Acceptable for bundle builds (cached); editor Compile gets slower —
acceptable per user. Guard: cap tets (e.g. 150k) with a clear warning.

## Testing
- 3D cantilever (100×10×25 µm): mode 1 within 3% of in-plane Euler-Bernoulli;
  some mode within 8% of the out-of-plane analytic estimate.
- Real comb island: mode 1 within 2% of the 2D solver; at least one mode in
  (50, 420) kHz (out-of-plane family 2D cannot see).
- Bundle: some modal result has nonzero dz on unmasked vertices; BOX/HANDLE
  still zero; labels include an "(out-of-plane)" entry for comb.
- Full suite stays green (2D fem tests untouched).
