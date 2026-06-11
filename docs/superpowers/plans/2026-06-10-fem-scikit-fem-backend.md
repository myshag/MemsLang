# scikit-fem FEM Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-written plane-stress solver `soidlc/fem2d.py` with a real sparse FEM built on scikit-fem + gmsh, keeping design-closure, ROM and plotting working.

**Architecture:** New package `soidlc/fem/` with backend-neutral data types (`FemMesh`/`FemResult`), a gmsh mesher (`mesh_build.py`), a scikit-fem solver (`skfem_solve.py`, quadratic P2 triangles, plane stress, `eigsh` modal), and the pipeline hook (`analyze.py`). Consumers are adapted to triangle cells; `fem2d.py` is deleted.

**Tech Stack:** Python, scikit-fem (numpy/scipy), gmsh (native arm64), existing `soidlc` geometry kernel.

---

## File Structure

- Create `soidlc/fem/__init__.py` — public API surface re-exporting `build_mesh`, `modal`, `static_solve`, `analyze`, `MAX_ELEMENTS`, `FemMesh`, `FemResult`.
- Create `soidlc/fem/result.py` — `FemMesh`, `FemResult` dataclasses.
- Create `soidlc/fem/mesh_build.py` — `build_mesh(shapes, h, lump_label) -> FemMesh`.
- Create `soidlc/fem/skfem_solve.py` — `modal(...)`, `static_solve(...)`.
- Create `soidlc/fem/analyze.py` — `analyze(elab, art, h, plot_prefix)`.
- Modify `soidlc/closure.py` — fem import + `n_cells`.
- Modify `soidlc/reduce.py` — import `FemMesh`.
- Modify `soidlc/femplot.py` — iterate triangle `cells`.
- Modify `soidlc/metrics.py`, `soidlc/__init__.py`, `soidlc/cli.py` — fem import.
- Modify `pyproject.toml`, `README.md` — deps + retire "zero dependencies".
- Delete `soidlc/fem2d.py`.
- Create `tests/test_fem.py` — new FEM tests.

Tests run with the project venv: `.venv/bin/python -m pytest`.

---

## Task 1: Dependencies and package skeleton

**Files:**
- Modify: `pyproject.toml:13-15`
- Create: `soidlc/fem/__init__.py`

- [ ] **Step 1: Add runtime deps**

In `pyproject.toml`, replace the zero-deps block:

```toml
# FEM layer uses scientific libraries; the core compiler (parser, geometry,
# 2.5D extrusion, software renderer, exporters) remains pure standard library.
dependencies = [
    "scikit-fem>=10",
    "gmsh>=4.13",
]
```

- [ ] **Step 2: Create the package with a stub API**

`soidlc/fem/__init__.py`:

```python
"""Finite-element layer for soidlc, built on scikit-fem + gmsh."""
from .result import FemMesh, FemResult
from .mesh_build import build_mesh, MAX_ELEMENTS
from .skfem_solve import modal, static_solve
from .analyze import analyze

__all__ = ["FemMesh", "FemResult", "build_mesh", "modal",
           "static_solve", "analyze", "MAX_ELEMENTS"]
```

(This import will fail until later tasks create the modules — that is expected;
do not run it yet.)

- [ ] **Step 3: Install deps into the venv**

Run: `.venv/bin/python -m pip install -e .`
Expected: `Successfully installed ... scikit-fem ... gmsh ...`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml soidlc/fem/__init__.py
git commit -m "feat(fem): add scikit-fem/gmsh deps and package skeleton"
```

---

## Task 2: FemMesh / FemResult data types

**Files:**
- Create: `soidlc/fem/result.py`
- Test: `tests/test_fem.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fem.py`:

```python
from soidlc.fem.result import FemMesh, FemResult


def test_femmesh_basic():
    m = FemMesh(
        nodes=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        cells=[(0, 1, 2)],
        fixed={0},
        fill=[1.0],
        extra_mass=[(2, 5.0)],
    )
    assert m.n_cells == 1
    assert len(m.nodes) == 3
    assert 0 in m.fixed


def test_femresult_basic():
    r = FemResult(freqs=[1.0e3], vecs=[[0.1, 0.2]], dof_of={0: 0})
    assert r.freqs[0] == 1.0e3
    assert r.dof_of[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fem.py -v`
Expected: FAIL — `ModuleNotFoundError: soidlc.fem.result`

- [ ] **Step 3: Implement the data types**

`soidlc/fem/result.py`:

```python
"""Backend-neutral FEM data types shared by mesher, solver and consumers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class FemMesh:
    nodes: List[Tuple[float, float]] = field(default_factory=list)  # um
    cells: List[Tuple[int, ...]] = field(default_factory=list)      # tri verts
    fixed: Set[int] = field(default_factory=set)                    # node idx
    fill: List[float] = field(default_factory=list)                 # per cell
    extra_mass: List[Tuple[int, float]] = field(default_factory=list)  # um^2

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_free_dof(self) -> int:
        return 2 * (len(self.nodes) - len(self.fixed))


@dataclass
class FemResult:
    freqs: List[float] = field(default_factory=list)          # Hz
    vecs: List[List[float]] = field(default_factory=list)     # M-normalised
    dof_of: Dict[int, int] = field(default_factory=dict)      # node -> base
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fem.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add soidlc/fem/result.py tests/test_fem.py
git commit -m "feat(fem): add FemMesh/FemResult data types"
```

---

## Task 3: gmsh mesher — single polygon

**Files:**
- Create: `soidlc/fem/mesh_build.py`
- Test: `tests/test_fem.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fem.py`:

```python
from soidlc import geometry as G
from soidlc.fem import mesh_build


def _rect_shape(x0, y0, w, h, **kw):
    return G.Shape(layer="DEVICE", polygon=G.rect_corner(x0, y0, w, h), **kw)


def test_mesh_single_rect():
    beam = _rect_shape(0.0, 0.0, 100.0, 10.0)
    m = mesh_build.build_mesh([beam], h=3.0)
    assert m.n_cells > 50                      # meshed, not degenerate
    assert all(len(c) == 3 for c in m.cells)   # triangles
    x0 = min(p[0] for p in m.nodes)
    x1 = max(p[0] for p in m.nodes)
    assert x0 < 1.0 and x1 > 99.0              # spans the rectangle
    assert len(m.fill) == m.n_cells
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fem.py::test_mesh_single_rect -v`
Expected: FAIL — `ModuleNotFoundError: soidlc.fem.mesh_build`

- [ ] **Step 3: Implement the mesher core**

`soidlc/fem/mesh_build.py`:

```python
"""Mesh an elaborated island's polygons into a triangle FemMesh via gmsh.

Holes are smeared into a per-cell mass fill factor (not cut from the mesh);
fingers (label contains ``lump_label``) are lumped as point mass onto the
nearest meshed node. These are deliberate phase-1 simplifications inherited
from the original fem2d.
"""
from __future__ import annotations

from typing import List, Tuple

import gmsh

from .. import geometry as G
from .result import FemMesh

MAX_ELEMENTS = 30000


def _point_in_ring(px: float, py: float, ring) -> bool:
    """Ray-cast point-in-polygon for a single ring."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > py) != (yj > py):
            xint = (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
            if px < xint:
                inside = not inside
        j = i
    return inside


def _add_surface(poly: G.Polygon, h: float) -> int:
    """Add one OCC plane surface (exterior + holes) at mesh size h."""
    def loop(ring) -> int:
        pts = [gmsh.model.occ.addPoint(x, y, 0.0, h) for x, y in ring]
        lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                 for i in range(len(pts))]
        return gmsh.model.occ.addCurveLoop(lines)

    outer = loop(poly.exterior)
    inners = [loop(hr) for hr in poly.holes]
    return gmsh.model.occ.addPlaneSurface([outer, *inners])


def build_mesh(shapes: List[G.Shape], h: float,
               lump_label: str = "finger") -> FemMesh:
    solids = [s for s in shapes if lump_label not in s.label]
    lumped = [s for s in shapes if lump_label in s.label]
    mesh = FemMesh()
    if not solids:
        return mesh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("island")
        tags = [(2, _add_surface(s.polygon.normalized(), h)) for s in solids]
        gmsh.model.occ.synchronize()
        # fuse overlapping/adjacent solids into one domain
        if len(tags) > 1:
            gmsh.model.occ.fuse([tags[0]], tags[1:])
            gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

        # nodes
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        idx_of = {int(t): i for i, t in enumerate(node_tags)}
        mesh.nodes = [(coords[3 * i], coords[3 * i + 1])
                      for i in range(len(node_tags))]
        # triangles (type 2)
        etypes, etags, enodes = gmsh.model.mesh.getElements(2)
        for et, conn in zip(etypes, enodes):
            if et != 2:               # only linear triangles
                continue
            for k in range(0, len(conn), 3):
                mesh.cells.append((idx_of[int(conn[k])],
                                   idx_of[int(conn[k + 1])],
                                   idx_of[int(conn[k + 2])]))
    finally:
        gmsh.finalize()

    _mark_fixed(mesh, shapes)
    _assign_fill(mesh, shapes)
    _lump_fingers(mesh, lumped)
    return mesh


def _mark_fixed(mesh: FemMesh, shapes: List[G.Shape]) -> None:
    anchored = [s.polygon for s in shapes if s.mech == "anchored"]
    for n, (px, py) in enumerate(mesh.nodes):
        for poly in anchored:
            if _point_in_ring(px, py, poly.exterior):
                mesh.fixed.add(n)
                break


def _assign_fill(mesh: FemMesh, shapes: List[G.Shape]) -> None:
    """Per-cell mass fill: 1.0 minus hole-area coverage at the centroid."""
    holes = [hr for s in shapes for hr in s.polygon.holes]
    for (a, b, c) in mesh.cells:
        cx = (mesh.nodes[a][0] + mesh.nodes[b][0] + mesh.nodes[c][0]) / 3.0
        cy = (mesh.nodes[a][1] + mesh.nodes[b][1] + mesh.nodes[c][1]) / 3.0
        in_hole = any(_point_in_ring(cx, cy, hr) for hr in holes)
        mesh.fill.append(0.0 if in_hole else 1.0)


def _lump_fingers(mesh: FemMesh, lumped: List[G.Shape]) -> None:
    if not mesh.nodes:
        return
    for s in lumped:
        x0, y0, x1, y1 = s.polygon.bbox()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        best = min(range(len(mesh.nodes)),
                   key=lambda n: (mesh.nodes[n][0] - cx) ** 2
                   + (mesh.nodes[n][1] - cy) ** 2)
        mesh.extra_mass.append((best, s.polygon.area()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fem.py::test_mesh_single_rect -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add soidlc/fem/mesh_build.py tests/test_fem.py
git commit -m "feat(fem): gmsh triangle mesher for island polygons"
```

---

## Task 4: Mesher — union, holes, anchored, fingers

**Files:**
- Test: `tests/test_fem.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fem.py`:

```python
def test_mesh_union_is_connected():
    a = _rect_shape(0.0, 0.0, 50.0, 10.0)
    b = _rect_shape(40.0, 0.0, 50.0, 10.0)   # overlaps a
    m = mesh_build.build_mesh([a, b], h=4.0)
    # one connected component: BFS over cell adjacency reaches every node
    from collections import defaultdict
    adj = defaultdict(set)
    for (i, j, k) in m.cells:
        for u in (i, j, k):
            for v in (i, j, k):
                adj[u].add(v)
    seen, stack = set(), [m.cells[0][0]]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u] - seen)
    used = {n for c in m.cells for n in c}
    assert seen == used                      # fully connected


def test_mesh_anchored_marks_fixed():
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 5.0, 10.0, mech="anchored")
    m = mesh_build.build_mesh([body, anchor], h=3.0)
    assert len(m.fixed) > 0
    assert all(mesh_node_x(m, n) <= 5.5 for n in m.fixed)


def mesh_node_x(m, n):
    return m.nodes[n][0]


def test_mesh_fingers_lumped_not_meshed():
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    finger = _rect_shape(20.0, 10.0, 2.0, 8.0, label="rotor_finger")
    m = mesh_build.build_mesh([body, finger], h=3.0)
    assert len(m.extra_mass) == 1
    # finger tip y=18 is not in the meshed body (height 10)
    assert max(p[1] for p in m.nodes) <= 10.5
```

- [ ] **Step 2: Run tests to verify they fail (or pass)**

Run: `.venv/bin/python -m pytest tests/test_fem.py -k "union or anchored or fingers" -v`
Expected: the Task 3 implementation already handles these; if any FAIL, fix
`mesh_build.py` (e.g. fuse robustness, point-in-ring tolerance) until PASS.

- [ ] **Step 3: Fix any failures inline**

If `test_mesh_union_is_connected` fails because `fuse` left duplicate boundary
nodes, add after `generate(2)`:

```python
        gmsh.model.mesh.removeDuplicateNodes()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fem.py -k "union or anchored or fingers" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add soidlc/fem/mesh_build.py tests/test_fem.py
git commit -m "test(fem): mesher union/holes/anchored/fingers"
```

---

## Task 5: scikit-fem modal solver

**Files:**
- Create: `soidlc/fem/skfem_solve.py`
- Test: `tests/test_fem.py`

- [ ] **Step 1: Write the failing test (analytic cantilever)**

Append to `tests/test_fem.py`:

```python
import math
from soidlc.fem import skfem_solve


def test_modal_cantilever_matches_euler():
    # clamped silicon beam L=100um H=10um; in-plane bending mode 1
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 2.0, 10.0, mech="anchored")
    m = mesh_build.build_mesh([body, anchor], h=2.0)
    E, nu, rho, t = 170.0e9, 0.28, 2330.0, 10.0e-6
    freqs, vecs, dof_of = skfem_solve.modal(m, E, nu, rho, t, n_modes=3)

    L, H = 100e-6, 10e-6
    I = (H ** 3) / 12.0
    A = H
    f1 = (1.8751 ** 2) / (2 * math.pi) * math.sqrt(E * I / (rho * A * L ** 4))
    assert abs(freqs[0] - f1) / f1 < 0.05          # within 5%
    assert len(vecs) == 3 and len(dof_of) == len(m.nodes)
    # a free tip node has nonzero displacement in mode 1
    tip = max(range(len(m.nodes)), key=lambda n: m.nodes[n][0])
    b = dof_of[tip]
    assert b >= 0 and abs(vecs[0][b + 1]) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fem.py::test_modal_cantilever_matches_euler -v`
Expected: FAIL — `ModuleNotFoundError: soidlc.fem.skfem_solve`

- [ ] **Step 3: Implement the modal solver**

`soidlc/fem/skfem_solve.py`:

```python
"""scikit-fem plane-stress solver: modal (eigsh) and static."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from skfem import (Basis, ElementVector, ElementTriP2, ElementTriP0, MeshTri,
                   BilinearForm, condense, solve, solver_eigen_scipy_sym)
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity

from .result import FemMesh

UM = 1.0e-6   # mesh coordinates are in micrometres


def _basis(mesh: FemMesh) -> Basis:
    """Build a P2 vector basis from a FemMesh (coords converted to metres)."""
    p = np.array(mesh.nodes, dtype=float).T * UM        # 2 x Nnodes
    t = np.array(mesh.cells, dtype=np.int64).T          # 3 x Ncells
    return Basis(MeshTri(p, t), ElementVector(ElementTriP2()))


def _assemble(mesh: FemMesh, E: float, nu: float, rho: float):
    basis = _basis(mesh)
    mu = E / (2.0 * (1.0 + nu))
    lam_ps = E * nu / (1.0 - nu * nu)
    K = linear_elasticity(lam_ps, mu).assemble(basis)

    # per-element mass fill (holes -> 0) carried as a DG0 field
    fill = np.array(mesh.fill, dtype=float) if mesh.fill \
        else np.ones(len(mesh.cells))
    fb = Basis(basis.mesh, ElementTriP0())
    fdg = fb.zeros()
    fdg[:] = fill

    @BilinearForm
    def mass(u, v, w):
        return rho * w["f"] * dot(u, v)

    M = mass.assemble(basis, f=fb.interpolate(fdg))
    return basis, K, M


def _vertex_dofs(basis: Basis) -> np.ndarray:
    """Global dof indices (2 x Nvertices): rows = (ux, uy) per vertex."""
    return basis.nodal_dofs   # shape (2, n_vertices) for ElementVector P2


def _clamped_dofs(basis: Basis, mesh: FemMesh) -> np.ndarray:
    vd = _vertex_dofs(basis)
    out = []
    for n in mesh.fixed:
        out.extend([int(vd[0, n]), int(vd[1, n])])
    return np.array(sorted(set(out)), dtype=np.int64)


def modal(mesh: FemMesh, E: float, nu: float, rho: float, t: float,
          n_modes: int = 3, return_vectors: bool = True):
    """Lowest natural frequencies (Hz) and M-normalised mode shapes.

    Returns (freqs, vecs, dof_of). ``t`` (thickness) cancels for plane-stress
    modal frequencies but is accepted for signature parity with consumers.
    """
    if not mesh.cells:
        return ([], [], {})
    basis, K, M = _assemble(mesh, E, nu, rho)
    D = _clamped_dofs(basis, mesh)
    k = min(n_modes, K.shape[0] - len(D) - 1)
    if k < 1:
        return ([], [], {})
    ls, xs = solve(*condense(K, M, D=D),
                   solver=solver_eigen_scipy_sym(k=k, sigma=0.0))
    freqs = [math.sqrt(abs(float(l))) / (2.0 * math.pi) for l in np.atleast_1d(ls)]

    vd = _vertex_dofs(basis)
    nverts = vd.shape[1]
    dof_of: Dict[int, int] = {}
    base = 0
    keep = []
    for n in range(nverts):
        if n in mesh.fixed:
            dof_of[n] = -1
        else:
            dof_of[n] = base
            keep.append((vd[0, n], vd[1, n]))
            base += 2

    vecs: List[List[float]] = []
    for col in range(len(freqs)):
        full = xs[:, col] if xs.ndim == 2 else xs
        v = []
        for (ix, iy) in keep:
            v.append(float(full[ix]))
            v.append(float(full[iy]))
        vecs.append(v)
    return freqs, vecs, dof_of
```

> Implementation note for the worker: the mass-form `f` coefficient must be a
> per-element constant field. If `mass.assemble(basis, f=fill)` raises a shape
> error, pass the fill through an `ElementTriP0` DG basis:
> `fb = Basis(skm, ElementTriP0()); fdg = fb.zeros(); fdg[:] = fill` and use
> `w["f"]` from `mass.assemble(basis, f=fb.interpolate(fdg))`. Verify the
> cantilever test still passes either way.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fem.py::test_modal_cantilever_matches_euler -v`
Expected: PASS (mode 1 ≈ 1.34–1.40 MHz, within 5% of 1.38 MHz)

- [ ] **Step 5: Commit**

```bash
git add soidlc/fem/skfem_solve.py tests/test_fem.py
git commit -m "feat(fem): scikit-fem P2 plane-stress modal solver"
```

---

## Task 6: scikit-fem static solver

**Files:**
- Modify: `soidlc/fem/skfem_solve.py`
- Test: `tests/test_fem.py`

- [ ] **Step 1: Write the failing test (tip-load deflection)**

Append to `tests/test_fem.py`:

```python
def test_static_cantilever_tip_load():
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 2.0, 10.0, mech="anchored")
    m = mesh_build.build_mesh([body, anchor], h=2.0)
    E, nu, t = 170.0e9, 0.28, 10.0e-6
    tip = max(range(len(m.nodes)), key=lambda n: m.nodes[n][0])
    F = 1.0e-6  # N, in -y
    disp = skfem_solve.static_solve(m, E, nu, t, loads={tip: (0.0, -F)})

    L, H = 100e-6, 10e-6
    I = t * (H ** 3) / 12.0
    y_analytic = F * L ** 3 / (3 * E * I)
    uy = abs(disp[tip][1])
    assert 0.5 * y_analytic < uy < 1.6 * y_analytic   # right order, ~beam theory
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fem.py::test_static_cantilever_tip_load -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'static_solve'`

- [ ] **Step 3: Implement static_solve**

Append to `soidlc/fem/skfem_solve.py`:

```python
def static_solve(mesh: FemMesh, E: float, nu: float, t: float,
                 loads: Dict[int, Tuple[float, float]]):
    """Static plane-stress deflection under nodal point loads (N).

    Returns ``{node: (ux, uy)}`` displacement in metres for every node.
    """
    if not mesh.cells:
        return {}
    basis, K, _ = _assemble(mesh, E, nu, 1.0)
    vd = _vertex_dofs(basis)
    b = np.zeros(K.shape[0])
    for n, (fx, fy) in loads.items():
        b[int(vd[0, n])] += fx / t      # plane-stress: force per thickness
        b[int(vd[1, n])] += fy / t
    D = _clamped_dofs(basis, mesh)
    u = solve(*condense(K, b, D=D))
    out: Dict[int, Tuple[float, float]] = {}
    for n in range(vd.shape[1]):
        out[n] = (float(u[int(vd[0, n])]), float(u[int(vd[1, n])]))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fem.py::test_static_cantilever_tip_load -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add soidlc/fem/skfem_solve.py tests/test_fem.py
git commit -m "feat(fem): scikit-fem static plane-stress solver"
```

---

## Task 7: analyze() pipeline hook

**Files:**
- Create: `soidlc/fem/analyze.py`
- Test: `tests/test_fem.py`

- [ ] **Step 1: Port the hook from fem2d**

Create `soidlc/fem/analyze.py` by copying `soidlc/fem2d.py:431-490` (the
`analyze` function) verbatim, then applying these substitutions:
- `mesh = build_mesh(ss, h)` — import from `.mesh_build`.
- `if len(mesh.elems) > MAX_ELEMENTS:` → `if mesh.n_cells > MAX_ELEMENTS:`.
- `if not mesh.elems:` → `if not mesh.cells:`.
- `freqs, vecs, dof_of = modal(...)` — import `modal` from `.skfem_solve`.
- ROM import stays `from .. import reduce as _rom`.
- femplot import stays `from .. import femplot`.

Full file:

```python
"""Pipeline hook: modal FEM for every suspended island; appends to the report,
compares mode 1 against the lumped f0, writes plots, builds the ROM."""
from __future__ import annotations

from typing import Optional

from .mesh_build import build_mesh, MAX_ELEMENTS
from .skfem_solve import modal


def analyze(elab, art, h: float = 12.0,
            plot_prefix: Optional[str] = None) -> None:
    dev = elab.process.device()
    if dev is None:
        return
    E, nu, rho = dev.E, dev.nu, dev.rho
    t = dev.thickness * 1e-6
    for cid, ss in getattr(elab, "islands", []):
        if not any(s.mech == "anchored" for s in ss):
            continue
        if not any(s.mech == "released" for s in ss):
            continue
        mesh = build_mesh(ss, h)
        if not mesh.cells:
            continue
        if mesh.n_cells > MAX_ELEMENTS:
            art.warnings.append(
                f"fem: island #{cid} has {mesh.n_cells} elements "
                f"(> {MAX_ELEMENTS}); increase --fem-h")
            continue
        freqs, vecs, dof_of = modal(mesh, E, nu, rho, t, n_modes=3)
        art.report.append(
            f"fem    island #{cid}: {mesh.n_cells} elements, "
            f"{mesh.n_free_dof} free dof (h = {h:g} um)")
        if not freqs:
            continue
        art.report.append(
            "fem    island #%d modes: %s" % (
                cid, ", ".join(f"{f / 1e3:.2f} kHz" for f in freqs)))
        f0 = art.model.get("f0")
        if f0 is not None:
            d = (freqs[0] - f0.value) / f0.value * 100.0
            art.report.append(
                f"fem    mode 1 vs lumped f0: {freqs[0] / 1e3:.2f} kHz "
                f"vs {f0.value / 1e3:.2f} kHz ({d:+.1f}%)")
        if plot_prefix:
            from .. import femplot
            modes_png = f"{plot_prefix}_fem_island{cid}_modes.png"
            femplot.plot_modes(mesh, freqs, vecs, dof_of, modes_png)
            art.files[f"fem_modes_{cid}"] = modes_png
            island_ids = {id(s) for s in ss}
            others = [s for s in art.result.shapes
                      if id(s) not in island_ids]
            d3_png = f"{plot_prefix}_fem_island{cid}_mode1_3d.png"
            femplot.render_deformed_3d(ss, others, elab.process, mesh,
                                       vecs[0], dof_of, d3_png)
            art.files[f"fem_3d_{cid}"] = d3_png
            art.report.append(
                f"fem    island #{cid} plots: {modes_png}, {d3_png}")
        try:
            from .. import reduce as _rom
            _rom.build(elab, art, ss, mesh, freqs, vecs, dof_of,
                       out_prefix=plot_prefix)
        except Exception as e:  # noqa: BLE001 - ROM is best-effort
            art.warnings.append(f"rom: extraction failed: {e}")
```

- [ ] **Step 2: Verify the package imports cleanly**

Run: `.venv/bin/python -c "import soidlc.fem; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add soidlc/fem/analyze.py
git commit -m "feat(fem): port analyze() pipeline hook to fem package"
```

---

## Task 8: Adapt closure, reduce, metrics, cli, __init__

**Files:**
- Modify: `soidlc/closure.py:198`, `soidlc/closure.py:217-222`
- Modify: `soidlc/reduce.py:33`
- Modify: `soidlc/metrics.py`, `soidlc/__init__.py:67-69`, `soidlc/cli.py`

- [ ] **Step 1: Update closure.py**

In `fem_measure` (around line 198):
- `from . import connectivity, fem2d` → `from . import connectivity, fem`
- `mesh = fem2d.build_mesh(island, fem_h)` → `mesh = fem.build_mesh(island, fem_h)`
- `if not mesh.elems or len(mesh.elems) > fem2d.MAX_ELEMENTS:` →
  `if not mesh.cells or mesh.n_cells > fem.MAX_ELEMENTS:`
- `freqs = fem2d.modal(mesh, d.E, d.nu, d.rho, d.thickness * 1e-6, n_modes=1)`
  → `freqs, _v, _d = fem.modal(mesh, d.E, d.nu, d.rho, d.thickness * 1e-6, n_modes=1)`

- [ ] **Step 2: Update reduce.py import**

`soidlc/reduce.py:33`: `from .fem2d import Mesh2D` → `from .fem import FemMesh as Mesh2D`

(Keeping the local alias `Mesh2D` avoids touching the type hints throughout
`reduce.py`; they are structural — `.nodes` and node indices only.)

- [ ] **Step 3: Update remaining imports**

- `soidlc/__init__.py:67-69`: `from . import fem2d` → `from . import fem`;
  `fem2d.analyze(...)` → `fem.analyze(...)`.
- `soidlc/cli.py`: no fem2d reference (flags only) — verify with
  `grep -n fem2d soidlc/cli.py` (expect no output).
- `soidlc/metrics.py`: `grep -n fem2d soidlc/metrics.py`; replace any
  `fem2d` with `fem` keeping call signatures (modal now returns a 3-tuple).

- [ ] **Step 4: Smoke test the comb example end to end**

Run: `.venv/bin/python -m soidlc.cli examples/comb_resonator.soidl -o /tmp/comb --fem`
Expected: report contains `fem    island #... modes:` and `rom    ...` lines;
exit code 0.

- [ ] **Step 5: Commit**

```bash
git add soidlc/closure.py soidlc/reduce.py soidlc/__init__.py soidlc/metrics.py
git commit -m "refactor(fem): point closure/reduce/metrics at fem package"
```

---

## Task 9: Adapt femplot to triangle cells

**Files:**
- Modify: `soidlc/femplot.py:33` (import), `:199-201`, `:209-211`
- Test: `tests/test_fem.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fem.py`:

```python
import os
from soidlc import femplot


def test_plot_modes_triangles(tmp_path):
    body = _rect_shape(0.0, 0.0, 100.0, 10.0)
    anchor = _rect_shape(0.0, 0.0, 2.0, 10.0, mech="anchored")
    m = mesh_build.build_mesh([body, anchor], h=3.0)
    freqs, vecs, dof_of = skfem_solve.modal(m, 170e9, 0.28, 2330.0, 10e-6,
                                            n_modes=2)
    out = str(tmp_path / "modes.png")
    femplot.plot_modes(m, freqs, vecs, dof_of, out)
    assert os.path.exists(out) and os.path.getsize(out) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fem.py::test_plot_modes_triangles -v`
Expected: FAIL — `femplot` iterates quad `mesh.elems`; `FemMesh` has no
`elems` (AttributeError) or unpacking error.

- [ ] **Step 3: Update femplot.py**

- `soidlc/femplot.py:33`: `from .fem2d import Mesh2D` →
  `from .fem import FemMesh as Mesh2D`.
- In `plot_modes`, replace the quad loops (around lines 199 and 209):

```python
        for (n0, n1, n2) in mesh.cells:
            tri = [mesh.nodes[i] for i in (n0, n1, n2)]
            # ... existing per-edge / fill logic over the 3 vertices
```

Concretely change each `for (n0, n1, n2, n3, _dx, _dy, _f) in mesh.elems:`
loop body: drop `n3`/`_dx`/`_dy`/`_f`, iterate vertices `(n0, n1, n2)`, and
where the old code drew a 4-vertex quad outline, draw the 3-vertex triangle
(`_line` between the three pairs; `_fill_tri` already takes 3 points).
- In `render_deformed_3d`, apply the same `mesh.elems` → `mesh.cells`
  (triangle) substitution for the deformed surface tessellation.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fem.py::test_plot_modes_triangles -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add soidlc/femplot.py tests/test_fem.py
git commit -m "refactor(femplot): draw triangle cells from FemMesh"
```

---

## Task 10: Delete fem2d, full regression

**Files:**
- Delete: `soidlc/fem2d.py`
- Test: all examples + `tests/`

- [ ] **Step 1: Confirm no remaining references**

Run: `grep -rn "fem2d" soidlc/ tests/`
Expected: no output. (Fix any stragglers before deleting.)

- [ ] **Step 2: Delete the module**

```bash
git rm soidlc/fem2d.py
```

- [ ] **Step 3: Run every example with --fem**

Run:
```bash
for f in examples/*.soidl; do
  .venv/bin/python -m soidlc.cli "$f" -o "/tmp/$(basename "$f" .soidl)" --fem -q \
    || echo "FAILED: $f"
done
```
Expected: no `FAILED:` lines; each produces `_fem_island*_modes.png`.

- [ ] **Step 4: Run --fem-closure on the comb resonator**

Run: `.venv/bin/python -m soidlc.cli examples/comb_resonator.soidl -o /tmp/cc --fem-closure`
Expected: report shows `closure: FEM calibration pass ...` and a factor near 1.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (the prior 87 plus the new `test_fem.py`). Investigate and
fix any regression before committing.

- [ ] **Step 6: Update README**

In `README.md`, change the "pure standard-library Python — zero dependencies"
claim to reflect the FEM layer's scientific dependencies, e.g.:

> The core compiler (parser, geometry kernel, 2.5D extruder, software renderer
> and exporters) is pure standard-library Python. The optional FEM layer
> (`--fem`) uses scikit-fem and gmsh.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(fem): remove fem2d, scikit-fem backend is the FEM engine"
```

---

## Notes for the implementer

- Run everything through `.venv/bin/python` (Homebrew Python is PEP-668
  externally managed; never `pip install` globally).
- gmsh must be `initialize()`d and `finalize()`d around each mesh build; never
  leave it initialised across calls (state leaks between islands).
- The scikit-fem API specifics most likely to need a small fix at runtime:
  the mass-form fill coefficient (see note in Task 5) and `basis.nodal_dofs`
  shape for `ElementVector(ElementTriP2())` — confirm it is `(2, n_vertices)`;
  if it is flattened, reshape accordingly. The cantilever test is the oracle.
- Tolerances in the analytic tests (5% modal, 0.5–1.6× static) absorb
  shear/mesh effects; do not tighten them to make a run pass — fix the model.
```
