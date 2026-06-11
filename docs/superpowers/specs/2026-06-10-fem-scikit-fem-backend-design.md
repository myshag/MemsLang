# M1 — Replace the home-grown FEM with scikit-fem

**Date:** 2026-06-10
**Status:** Design approved, ready for implementation planning
**Milestone:** M1 of the "open MEMS suite" roadmap (see memory `memslang-north-star`)

## 1. Goal

Replace the hand-written 2D plane-stress solver in `soidlc/fem2d.py` with a
real, sparse, validated FEM built on **scikit-fem** (numpy/scipy), while
keeping every current consumer working: design closure (`--fem-closure`),
model-order reduction (`reduce.py`), and mode/deformation plotting
(`femplot.py`).

This is the first milestone toward a free, code-first MEMS simulation suite.
The new FEM layer is designed so that later physics (electrostatic actuation,
piezoelectric, damping) can be added as additional scikit-fem weak forms on
the **same mesh and result abstraction** — but those are out of scope here.

### Non-goals (explicit YAGNI for M1)
- No electrostatic/piezo/thermal coupling yet (later milestones).
- No Elmer / external-solver backend. Decision: scikit-fem only
  (Elmer has no native arm64 build and crashes under x86 emulation on the
  target machine — see memory `memslang-fem-backend-decision`).
- No pluggable multi-backend abstraction. One backend.
- No change to the SOIDL language, geometry kernel, exporters, or 3D mesher.

## 2. Background: what `fem2d.py` does today

- `Mesh2D`: tensor-grid of **axis-aligned quads**; `nodes: [(x,y) um]`,
  `elems: [(n0,n1,n2,n3, dx, dy, fill)]`, `fixed: set`, `extra_mass`.
- `build_mesh(shapes, h, lump_label="finger")`: meshes an island's
  rectilinear polygons; release holes are **smeared into a mass fill factor**
  (not meshed); comb **fingers** are **lumped** as point mass onto the nearest
  backbone node.
- `modal(mesh, E, nu, rho, t, n_modes, return_vectors)` →
  `(freqs[Hz], vecs, dof_of)`. Q6 (Wilson incompatible-mode) quads to avoid
  shear locking; banded Cholesky + inverse iteration.
- `static_solve(mesh, E, nu, t, loads)` → nodal displacement.
- `analyze(elab, art, h, plot_prefix)`: pipeline hook — meshes each suspended
  island, runs modal, appends to report, compares mode 1 vs lumped `f0`,
  writes plots, then calls `reduce.build(...)`.
- `MAX_ELEMENTS = 30000`.

### The result contract every consumer depends on
- `mesh.nodes`: `list[(x,y)]` in micrometres.
- `mesh.elems`: cell connectivity — **only `femplot` uses this**, to draw the
  mesh (iterates quads `n0..n3`). Also used for empty/size checks.
- `dof_of`: `dict[node_index -> base]`; a node's displacement is
  `(vec[base], vec[base+1])`; fixed/constrained nodes map to `-1`.
- `vecs`: list of **M-normalised** mode vectors over free dofs.
- `freqs`: list of natural frequencies in Hz.

`reduce._mean_phi` and `femplot._node_disp` both read displacement purely
**per node** via `dof_of`; they are topology-agnostic. Only `femplot`'s mesh
drawing assumes quads.

## 3. Architecture

New package `soidlc/fem/` replacing the single `fem2d.py` module:

```
soidlc/fem/
  __init__.py        # public API: build_mesh, modal, static_solve, analyze,
                     #             MAX_ELEMENTS, FemMesh, FemResult
  result.py          # FemMesh + FemResult: backend-neutral data types
  mesh_build.py      # island polygons -> triangle mesh (gmsh) + mass model
  skfem_solve.py     # assemble K, M; modal (scipy eigsh) + static (spsolve)
  analyze.py         # pipeline hook (was fem2d.analyze)
```

`fem2d.py` is **deleted**. `from .fem2d import Mesh2D` references become
`from .fem import FemMesh`.

### 3.1 Backend-neutral data types (`result.py`)

```python
@dataclass
class FemMesh:
    nodes: list[tuple[float, float]]   # (x, y) um  -- vertices only
    cells: list[tuple[int, ...]]       # vertex-index tuples (triangles: len 3)
    fixed: set[int]                    # constrained node indices
    fill: list[float]                  # per-cell mass fill factor (holes)
    extra_mass: list[tuple[int,float]] # (node, lumped mass um^2) for fingers
    @property
    def n_cells(self) -> int: ...      # replaces len(mesh.elems)
```

- `nodes`/`cells`/`fixed`/`extra_mass` keep the names/semantics consumers use.
- `mesh.elems` (quad 7-tuples) is **removed**; `femplot` switches to iterating
  `cells` (triangles). `n_cells` replaces the `len(mesh.elems)` size checks.

```python
@dataclass
class FemResult:
    freqs: list[float]                 # Hz
    vecs: list[list[float]]            # M-normalised, over free dofs
    dof_of: dict[int, int]             # node -> base dof (or -1)
```

`modal()` returns `(freqs, vecs, dof_of)` exactly as before, so `reduce.build`
and `femplot.plot_modes` signatures are unchanged.

### 3.2 Meshing (`mesh_build.py`)

`build_mesh(shapes, h, lump_label="finger") -> FemMesh`, same signature.

- Split shapes into **solids** (mesh) and **lumped** (`lump_label in label`,
  e.g. fingers) exactly like today.
- Build the island domain as a polygon-with-holes from the solid shapes'
  rectilinear outlines (union of axis-aligned rectangles), then mesh it with
  **gmsh** (already a dependency; native arm64) at target size `h`.
  - Read the `.msh` into `FemMesh` via `meshio` (new lightweight dep).
  - Use **P1 triangle vertices** as `nodes`; `cells` are the triangles.
- **Holes / release perforation:** keep the fem2d simplification for M1 —
  smear holes into a per-cell `fill` mass factor (do not cut them from the
  mesh). Document as a deliberate phase-1 simplification.
- **Fingers:** lump as `extra_mass` on the nearest mesh node, as today.

> Meshing-accuracy note: fem2d used Q6 quads to beat shear locking on slender
> flexures. With triangles we avoid locking by using **quadratic (P2)
> elements** in the solver (§3.3) rather than P1. The mesh stores P1 vertices
> for the public node set; the solver enriches internally. The cantilever PoC
> (P2, 81×9 grid) matched Euler–Bernoulli to **-0.7%**, confirming this.

### 3.3 Solver (`skfem_solve.py`)

- Element: `ElementVector(ElementTriP2())` (quadratic, no locking).
- Plane stress: stiffness via `linear_elasticity(lam_ps, mu)` with
  `mu = E/(2(1+nu))`, `lam_ps = E*nu/(1-nu^2)`.
- Mass: `BilinearForm rho * dot(u,v)`, scaled per cell by `fill`; lumped
  `extra_mass` added to the diagonal at finger nodes.
- Constraints: `fixed` nodes → `condense(..., D=clamped_dofs)`.
- **Modal:** `solve(*condense(K, M, D=...),
  solver=solver_eigen_scipy_sym(k=n_modes, sigma=0.0))`; `f=sqrt(λ)/(2π)`.
  Mode vectors normalised so `phiᵀ M phi = 1`.
- **Static:** `solve(*condense(K, b, D=...))` for applied nodal loads.
- Map the P2 solution down to **per-vertex** `(ux, uy)` and build `dof_of`
  over vertices so the consumer contract (§2) holds. `MAX_ELEMENTS` retained
  as a guard against oversized meshes.

### 3.4 Consumer adaptations

| File | Change |
|---|---|
| `fem/analyze.py` | port `fem2d.analyze` verbatim except `FemMesh`/`n_cells`. |
| `closure.py` | `fem_measure`: `fem2d` → `fem`; `mesh.elems` check → `mesh.n_cells`. Behaviour identical. |
| `reduce.py` | `from .fem2d import Mesh2D` → `from .fem import FemMesh`; node/`dof_of` logic unchanged (topology-agnostic). |
| `femplot.py` | `_node_disp` unchanged; `plot_modes` & `render_deformed_3d` iterate **triangle `cells`** instead of quad `elems` (fill 3 verts instead of 4). |
| `metrics.py` | update any `fem2d` import/reference to `fem`. |
| `__init__.py` / `cli.py` | `from . import fem2d` → `from . import fem`; flags (`--fem`, `--fem-h`, `--fem-closure`) unchanged. |
| `pyproject.toml` | add `scikit-fem`, `gmsh`, `meshio` to `dependencies`. **The "zero dependencies" claim in README/pyproject is retired** — update prose to "scientific deps: numpy/scipy via scikit-fem, gmsh for meshing." |

## 4. Dependencies

Adds runtime deps (was zero): `scikit-fem`, `gmsh`, `meshio` (pull in
`numpy`, `scipy`). All install natively on macOS arm64 via pip. README and
`pyproject.toml` description updated to reflect this; the core compiler
(parser, geometry, exporters, 3D mesher) stays pure-stdlib — only the FEM
layer takes scientific deps.

## 5. Testing

1. **Analytic cantilever** (port the PoC): clamped silicon beam, mode 1 within
   ~2% of Euler–Bernoulli; assert convergence under refinement.
2. **Plate / known modes:** a clamped-clamped beam or square plate vs textbook
   coefficients within tolerance.
3. **Example regression:** for `comb_resonator`, `accelerometer`, `gyroscope`,
   `gyroscope_2mass` — `--fem` runs, mode 1 is finite/positive and within a
   sane band of the lumped `f0` (loose tolerance; documents lumped-vs-FEM
   bias, not exact equality).
4. **Consumer smoke tests:** `--fem` produces mode PNG + 3D PNG; `reduce`
   still writes `_model.py/.cir/.va` and the Allan plot; `--fem-closure`
   converges and reports a calibration factor.
5. **Determinism:** fixed mesh seed → stable frequencies across runs.

Existing 87 tests must still pass (those not touching fem2d internals).

## 6. Risks / open points

- **gmsh meshing of rectilinear unions with holes**: island outlines come from
  unions of axis-aligned rectangles; need a robust polygon-boolean to produce
  outer+hole loops for gmsh. Mitigation: reuse the existing geometry kernel's
  polygon ops if available; else mesh per-rectangle and merge coincident
  nodes. (To be settled in the plan.)
- **P2 → vertex downsampling** for `dof_of`: mode shapes are reported at
  vertices only; midside detail is dropped for plotting/ROM. Acceptable —
  ROM averages over a bbox; plots are qualitative.
- **Hole smearing** keeps fem2d's approximation; revisit if validation (M5)
  shows it matters.
- Tolerances in example regression are intentionally loose (lumped models and
  FEM legitimately differ); tighten against literature in M5.
```
