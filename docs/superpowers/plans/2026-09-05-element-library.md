# Element Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow SOIDL from 8 rectangle-only primitives and 4 in-plane examples into a real element library: an `import` statement plus a standard library written in SOIDL, seven new Python primitives, backside-trench support so out-of-plane devices become expressible, and 11 new example devices.

**Architecture:** Hybrid library (option C from the spec). Irreducible parametric geometry (`comb`, `chevron`, `ring`, …) stays as Python generators in `soidlc/primitives.py`. Everything expressible as a composition of primitives lives in `soidlc/lib/*.soidl` and is pulled in with a new `import` statement, so the library of a language for describing MEMS is itself written in that language.

**Tech Stack:** Python 3.9+ standard library for the compiler core (parser, geometry, 2.5D extrusion, renderer, exporters — no third-party imports allowed here). scikit-fem 12 + gmsh 4.15 (with numpy/scipy transitively) for the FEM validation tasks only.

**Spec:** `docs/superpowers/specs/2026-09-05-element-library-design.md`

## Global Constraints

- **Compiler core stays dependency-free.** `soidlc/{lexer,parser,sast,units,elaborate,geometry,primitives,mesh,build3d,connectivity,render,svg,exporters}.py` must not import numpy, scipy, skfem or gmsh. Only `soidlc/fem/*`, `soidlc/reduce.py` and `soidlc/metrics.py` may.
- **Environment.** FEM dependencies are required for a green baseline, and Homebrew Python is externally managed (PEP 668), so work in a venv:
  ```bash
  python3 -m venv .venv && .venv/bin/python -m pip install -e . && .venv/bin/python -m pip install pytest
  ```
  All `Run:` commands below assume `.venv/bin/python`.
- **Baseline (verified 2026-09-05):** `.venv/bin/python -m unittest tests.test_soidlc -q` → **32 tests, OK**. Every task must leave it OK.
- **Out of scope, pre-existing, do not fix here:** `tests/test_etch.py`, `test_koh.py`, `test_corner.py`, `test_recipe.py`, `test_crystal.py` fail because the C core self-compiles with `-fopenmp` (`soidlc/etch.py:42`), unsupported by Apple clang without libomp. Unrelated to the element library.
- **Closure needs gmsh even without `--fem`.** Without it, `solve`/`require` silently degrade to the warning `closure: spec metrics could not be evaluated`. If a `solve` in a new example appears to do nothing, check the venv first.
- **Units.** Every length in `.soidl` source carries a unit (`200 um`). In Python, `Quantity.um` converts to micrometres; primitives return micrometre coordinates.
- **SOIDL has no unary minus on literals in all positions.** Existing examples write `0 - 320 um`, not `-320 um`. Follow that idiom in every new `.soidl` file.
- **Commit after every task.** Message style follows the repo: `feat(lib): …`, `fix(mesh): …`, `docs(lib): …`, ending with the two attribution trailers used on recent commits.

## File Structure

**Created:**
- `soidlc/lib/flexures.soidl` — `guided_beam`, `folded_flexure`, `serpentine`, `crab_leg`, `frame`, `torsion_bar`
- `soidlc/lib/membranes.soidl` — `membrane`
- `examples/*.soidl` — 11 new devices
- `tests/test_library.py` — all tests for `import`, path geometry, new primitives, new metrics

**Modified:**
- `soidlc/lexer.py` — one keyword
- `soidlc/sast.py` — one AST node
- `soidlc/parser.py` — top-level `import` production
- `soidlc/elaborate.py` — import resolution, torsional lumped model
- `soidlc/geometry.py` — `wire`, `circle`, `arc`, `annulus`
- `soidlc/mesh.py` — hole-bridging fix
- `soidlc/primitives.py` — 7 new primitives, `prim_trench` gains geometry
- `soidlc/build3d.py` — trench subtraction from HANDLE/BOX
- `soidlc/metrics.py` — `pull_in`, `stroke_thermal`, `lorentz_force`
- `soidlc/reduce.py` — recognise new transducer labels
- `soidlc/__init__.py`, `soidlc/webbundle.py` — thread `base_dir`
- `soidlc/cli.py` — no change needed (verified: `compile_file` already has the path)
- `pyproject.toml` — ship `soidlc/lib/*.soidl` as package data
- `tests/test_soidlc.py` — glob the examples directory

**Spec corrections found during planning (already verified against the code):**
1. The spec says `test_examples_have_no_errors` iterates `examples/*.soidl`. It does not — `tests/test_soidlc.py:112` hardcodes two filenames. Task 1 fixes this *first*, so the safety net exists before any example is added.
2. The spec assumes a file path is available for import resolution. `compile_source(src)` takes source **text** (`soidlc/__init__.py:35`); `base_dir` must be threaded explicitly. Task 3 does this.
3. The spec treats `_extrude_general` watertightness as an open risk with a fallback. It is now a **confirmed defect**, measured: an annulus extrudes with exactly 4 non-manifold edges, constant across `n_seg` ∈ {8, 16, 64, 128} — two bridge-seam edges per cap. Task 6 fixes it against that measurement.

---

## Phase 0 — core: safety net, `import`, path geometry

### Task 1: Cover every example, not two of them

**Files:**
- Modify: `tests/test_soidlc.py:111-115`

**Interfaces:**
- Consumes: nothing.
- Produces: a test that fails the moment any `examples/*.soidl` stops compiling cleanly. Every later task that adds an example relies on this.

- [ ] **Step 1: Replace the hardcoded tuple with a glob**

In `tests/test_soidlc.py`, replace `test_examples_have_no_errors` with:

```python
    def test_examples_have_no_errors(self):
        import glob
        paths = sorted(glob.glob(os.path.join(EX, "*.soidl")))
        self.assertGreaterEqual(len(paths), 4, "examples went missing")
        for path in paths:
            name = os.path.basename(path)
            with self.subTest(example=name):
                with open(path) as f:
                    art = compile_source(f.read())
                self.assertEqual(art.errors, [], f"{name}: {art.errors}")
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m unittest tests.test_soidlc.TestConnectivity.test_examples_have_no_errors -v`
Expected: PASS, now covering 4 files instead of 2. (Verified during planning: all four current examples compile with `errors == []`.)

- [ ] **Step 3: Prove the net actually catches something**

Temporarily append a broken device to `examples/gyroscope.soidl`:

```soidl
device broken_probe {
  inst P = plate(50 um, 50 um) at (0, 0);
  net A = P;
  net B = P;
}
```

Run the same command. Expected: FAIL — two nets on one island is a short. Then `git checkout examples/gyroscope.soidl` to revert.

- [ ] **Step 4: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc -q`
Expected: `Ran 32 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_soidlc.py
git commit -m "test: cover every example, not two hardcoded ones"
```

### Task 2: `import` — keyword, AST node, parser production

**Files:**
- Modify: `soidlc/lexer.py:16-24`, `soidlc/sast.py`, `soidlc/parser.py:55-72`
- Test: `tests/test_library.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `sast.Import(path: str)` nodes appearing in `sast.File.decls`. Task 3 resolves them. `parse(src)` signature is unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_library.py`:

```python
"""Tests for the SOIDL standard library: imports, path geometry, new elements."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soidlc import compile_source                      # noqa: E402
from soidlc import sast as A                           # noqa: E402
from soidlc.parser import parse, ParseError            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")


class TestImportParsing(unittest.TestCase):
    def test_import_becomes_an_ast_node(self):
        ast = parse('import "flexures.soidl";\ndevice d { }')
        imports = [d for d in ast.decls if isinstance(d, A.Import)]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].path, "flexures.soidl")

    def test_import_must_be_top_level(self):
        with self.assertRaises(ParseError):
            parse('device d { import "flexures.soidl"; }')


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_library -v`
Expected: FAIL — `AttributeError: module 'soidlc.sast' has no attribute 'Import'`.

- [ ] **Step 3: Add the keyword**

In `soidlc/lexer.py`, add `"import"` to the `KEYWORDS` set (line 17, alongside `"process", "component", "device", "chip"`).

- [ ] **Step 4: Add the AST node**

In `soidlc/sast.py`, next to the other declaration dataclasses:

```python
@dataclass
class Import:
    path: str
```

- [ ] **Step 5: Add the parser production**

In `soidlc/parser.py`, inside `parse_file`'s dispatch loop, before the `process` branch:

```python
            if self.at("KEYWORD", "import"):
                decls.append(self.parse_import())
                continue
```

and add the method:

```python
    def parse_import(self) -> A.Import:
        self.eat("KEYWORD", "import")
        path = self.eat("STRING").text
        self.accept("PUNCT", ";")
        return A.Import(path)
```

The token kind is `STRING` and the lexer already strips the surrounding quotes (`soidlc/lexer.py:84-92` accumulates only the characters between them), so `path` needs no further cleaning.

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_library -v`
Expected: PASS, 2 tests.

Note on `test_import_must_be_top_level`: it passes because `parse_item` (the component/device body parser) has no `import` branch and raises `ParseError` on an unexpected keyword. Confirm the raised type is `ParseError` and not a bare `Exception`; if it is not, add an explicit rejection in `parse_item`.

- [ ] **Step 7: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`.

- [ ] **Step 8: Commit**

```bash
git add soidlc/lexer.py soidlc/sast.py soidlc/parser.py tests/test_library.py
git commit -m "feat(lang): parse top-level import declarations"
```

### Task 3: Resolve imports in the elaborator

**Files:**
- Modify: `soidlc/elaborate.py:117-141`, `soidlc/__init__.py:35,75`, `soidlc/webbundle.py:20-34`, `pyproject.toml`
- Create: `soidlc/lib/` (directory, with a placeholder library file)
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `sast.Import` from Task 2.
- Produces:
  - `Elaborator(file, base_dir: Optional[str] = None)` — new second parameter.
  - `compile_source(src, ..., base_dir: Optional[str] = None)` and `compile_file(path, **kw)` which passes `base_dir=os.path.dirname(os.path.abspath(path))`.
  - `webbundle._compile_source(src, base_dir=None)`, `webbundle._compile(path)` passing the dirname.
  - Library components resolvable by name from any `.soidl` file that imports them. Every Phase 1+ task depends on this.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_library.py`:

```python
class TestImportResolution(unittest.TestCase):
    PROC = """
      process p {
        stack {
          layer DEVICE { thickness = 25 um; material = Si;
                         E = 169 GPa; rho = 2330 kg/m^3 }
          layer BOX    { thickness = 2 um; material = SiO2 }
          layer HANDLE { thickness = 400 um; material = Si }
        }
        rules { release { hole_size = 6 um; hole_pitch = 30 um;
                          max_solid_span = 40 um } }
      }
    """

    def test_bundled_library_resolves_without_a_base_dir(self):
        src = self.PROC + """
          import "flexures.soidl";
          device d {
            inst M = plate(200 um, 200 um) at (0, 0);
            inst S = array(guided_beam(L = 200 um, w = 4 um), count = 4,
                           place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        self.assertGreater(len(art.mesh.triangles), 0)

    def test_local_component_shadows_the_library_one(self):
        src = self.PROC + """
          import "flexures.soidl";
          component guided_beam(L = 100 um, w = 4 um) {
            mech port fixed, shuttle;
            geometry { anchor(500 um, 7 um) at (0, 0); }
          }
          device d {
            inst S = guided_beam() at (0, 0);
            net GND = S;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        # the local stub is one 500x7 anchor; the library component is not
        x0, y0, x1, y1 = _bbox(art)
        self.assertAlmostEqual(x1 - x0, 500.0, delta=1.0)

    def test_missing_import_is_a_clean_diagnostic(self):
        src = self.PROC + 'import "nope.soidl";\ndevice d { }'
        with self.assertRaises(Exception) as cm:
            compile_source(src)
        self.assertIn("nope.soidl", str(cm.exception))

    def test_cyclic_import_terminates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.soidl")
            b = os.path.join(tmp, "b.soidl")
            with open(a, "w") as f:
                f.write('import "b.soidl";\ncomponent ca(w = 4 um) '
                        '{ geometry { anchor(20 um, 20 um) at (0, 0); } }\n')
            with open(b, "w") as f:
                f.write('import "a.soidl";\ncomponent cb(w = 4 um) '
                        '{ geometry { anchor(20 um, 20 um) at (0, 0); } }\n')
            src = self.PROC + ('import "a.soidl";\n'
                               'device d { inst S = cb() at (0, 0); net G = S; }')
            art = compile_source(src, base_dir=tmp)
            self.assertEqual(art.errors, [], art.errors)

    def test_library_file_may_not_declare_a_device(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            lib = os.path.join(tmp, "bad.soidl")
            with open(lib, "w") as f:
                f.write("device sneaky { }\n")
            src = self.PROC + 'import "bad.soidl";\ndevice d { }'
            with self.assertRaises(Exception) as cm:
                compile_source(src, base_dir=tmp)
            self.assertIn("bad.soidl", str(cm.exception))


def _bbox(art):
    xs, ys = [], []
    for s in art.result.shapes:
        x0, y0, x1, y1 = s.polygon.bbox()
        xs += [x0, x1]
        ys += [y0, y1]
    return min(xs), min(ys), max(xs), max(ys)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_library -v`
Expected: FAIL — `compile_source() got an unexpected keyword argument 'base_dir'`, and the bundled-library test fails because `soidlc/lib/flexures.soidl` does not exist.

- [ ] **Step 3: Create the library directory with a minimal `guided_beam`**

Create `soidlc/lib/flexures.soidl`:

```soidl
// SOIDL standard library: suspension elements.
// Imported as:  import "flexures.soidl";
//
// Local origin of every flexure = its attachment point on the moving mass.
// Each beam reaches 2 um into the mass and runs outboard to its anchor, so
// an instance placed exactly on a mass edge is silicon-continuous with it.

// n clamped-guided beams in parallel, stiff along y, compliant along x.
// k = n * E * t * w^3 / L^3  (clamped-guided, both ends restrained in slope)
component guided_beam(L = 200 um, w = 4 um, n_beams = 2, pitch = 12 um) {
  mech port fixed, shuttle;

  derive k.x = n_beams * (process.DEVICE.E * process.DEVICE.thickness * w^3) / L^3;

  geometry {
    repeat i in 0..n_beams-1 {
      beam(L, w, dir = y) at (i * pitch, 2 um - L/2);
    }
    // truss tying the beam tops together, spanning them exactly
    beam((n_beams - 1) * pitch + w, w, dir = x)
        at ((n_beams - 1) * pitch/2, 0);
    anchor((n_beams - 1) * pitch + w + 12 um, 20 um)
        at ((n_beams - 1) * pitch/2, 2 um - L - 8 um);
  }

  check w >= 2 um;
  check L/w <= 100 warn "very compliant beam, check buckling";
}
```

The anchor overlaps the beam bottoms by 2 um (the beam spans `2 um - L` to `2 um`; the anchor is 20 um tall centred at `2 um - L - 8 um`, so it spans `2 um - L - 18 um` to `2 um - L + 2 um`). That overlap is what keeps the flexure from being a floating island — the connectivity check enforces it.

- [ ] **Step 4: Thread `base_dir` through the compile entry points**

In `soidlc/__init__.py`:

```python
def compile_source(src: str, device: Optional[str] = None,
                   out_prefix: Optional[str] = None,
                   include_handle: bool = True,
                   fem: bool = False, fem_h: float = 12.0,
                   fem_closure: bool = False,
                   base_dir: Optional[str] = None) -> Artifacts:
    ast = parse(src)
    elab = Elaborator(ast, base_dir=base_dir)
```

and:

```python
def compile_file(path: str, **kw) -> Artifacts:
    kw.setdefault("base_dir", os.path.dirname(os.path.abspath(path)))
    with open(path) as f:
        return compile_source(f.read(), **kw)
```

In `soidlc/webbundle.py`, mirror it:

```python
def _compile_source(src: str, base_dir: Optional[str] = None):
    """Elaborate SOIDL source text and build its 3D mesh; return (elab, result, mesh)."""
    ast = parse(src)
    elab = Elaborator(ast, base_dir=base_dir)
    from . import closure
    overrides = closure.run(elab, None, fem_calibrate=False, fem_h=20.0)
    result = elab.elaborate_device(None, overrides=overrides)
    mesh = build3d.build_mesh(result, elab.process, include_handle=True)
    return elab, result, mesh


def _compile(path: str):
    with open(path) as f:
        return _compile_source(f.read(),
                               base_dir=os.path.dirname(os.path.abspath(path)))
```

Add `import os` and `Optional` to `webbundle.py`'s imports if absent. `soidlc/cli.py` needs no change — it calls `compile_file`, which now supplies `base_dir` itself.

- [ ] **Step 5: Resolve imports in the elaborator**

In `soidlc/elaborate.py`, add near the top:

```python
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")


class ImportError_(Exception):
    """A SOIDL `import` could not be resolved."""
```

(`import os` at the top of the module if not already present; use a distinct name so the builtin `ImportError` is not shadowed.)

Change the constructor and `_index`:

```python
    def __init__(self, file: A.File, base_dir: Optional[str] = None):
        self.file = file
        self.base_dir = base_dir
        self._loaded_imports: set = set()
        self.components: Dict[str, A.Component] = {}
        ...        # rest unchanged
        self._index()

    def _index(self) -> None:
        proc = None
        # imports first, so a local declaration of the same name shadows a
        # library one (same rule as a user component shadowing a primitive)
        for d in self.file.decls:
            if isinstance(d, A.Import):
                self._load_import(d, self.base_dir)
        for d in self.file.decls:
            if isinstance(d, A.Process):
                proc = d
            elif isinstance(d, A.Component):
                self.components[d.name] = d
            elif isinstance(d, A.Device) and d.name != "__chip__":
                self.devices[d.name] = d
        if proc is not None:
            self.process = self._build_process(proc)

    def _resolve_import(self, path: str, base_dir) -> str:
        cands = []
        if base_dir:
            cands.append(os.path.join(base_dir, path))
        cands.append(os.path.join(_LIB_DIR, path))
        for c in cands:
            if os.path.isfile(c):
                return os.path.abspath(c)
        raise ImportError_(
            f"cannot resolve import {path!r}; looked in "
            + ", ".join(os.path.dirname(c) or "." for c in cands))

    def _load_import(self, node: A.Import, base_dir) -> None:
        abspath = self._resolve_import(node.path, base_dir)
        if abspath in self._loaded_imports:
            return                      # diamond import: load once, no error
        self._loaded_imports.add(abspath)
        with open(abspath) as f:
            sub = parse(f.read())
        sub_dir = os.path.dirname(abspath)
        for d in sub.decls:
            if isinstance(d, A.Import):
                self._load_import(d, sub_dir)
            elif isinstance(d, A.Component):
                self.components[d.name] = d
            else:
                raise ImportError_(
                    f"{os.path.basename(abspath)}: an imported file may only "
                    f"declare components, found {type(d).__name__.lower()}")
```

Add `from .parser import parse` to `elaborate.py`'s imports if absent.

Note the recursion order: a nested import is loaded before the importing library file's own components are indexed, so a library file's own component wins over one it imports — the same shadowing rule, one level down. The `_loaded_imports` set makes cycles terminate.

- [ ] **Step 6: Ship the library as package data**

In `pyproject.toml`, after the `[tool.setuptools]` block:

```toml
[tool.setuptools.package-data]
soidlc = ["lib/*.soidl"]
```

Without this the library works from a checkout and silently vanishes on `pip install` — `packages` is an explicit list, so non-Python files are not picked up by default.

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m unittest tests.test_library -v`
Expected: PASS, 7 tests.

- [ ] **Step 8: Verify the installed-package path really works**

Run:
```bash
.venv/bin/python -m pip install -e . -q && \
.venv/bin/python -c "
import soidlc, os
p = os.path.join(os.path.dirname(soidlc.__file__), 'lib', 'flexures.soidl')
print('library present:', os.path.isfile(p))
"
```
Expected: `library present: True`.

- [ ] **Step 9: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`.

- [ ] **Step 10: Commit**

```bash
git add soidlc/elaborate.py soidlc/__init__.py soidlc/webbundle.py \
        soidlc/lib/flexures.soidl pyproject.toml tests/test_library.py
git commit -m "feat(lang): resolve imports from a bundled SOIDL standard library"
```

### Task 4: `geometry.wire()` — a polyline of finite width

**Files:**
- Modify: `soidlc/geometry.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `geometry.wire(points: List[Pt], w: float) -> Polygon`, coordinates in micrometres. Used by `serpentine` (Task 9), `chevron` (Task 12) and `route` (Task 16).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_library.py`:

```python
from soidlc import geometry as G                       # noqa: E402


class TestWire(unittest.TestCase):
    def test_straight_wire_is_a_rectangle(self):
        p = G.wire([(0.0, 0.0), (10.0, 0.0)], 4.0)
        self.assertAlmostEqual(p.area(), 40.0, places=6)
        x0, y0, x1, y1 = p.bbox()
        self.assertAlmostEqual(x0, 0.0, places=6)
        self.assertAlmostEqual(x1, 10.0, places=6)
        self.assertAlmostEqual(y0, -2.0, places=6)
        self.assertAlmostEqual(y1, 2.0, places=6)

    def test_right_angle_wire_area_is_mitred(self):
        # two 10-long arms of width 4 sharing a square corner:
        # mitred area = 10*4 + 10*4 - 4*4 = 64 (the corner counted once)
        p = G.wire([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], 4.0)
        self.assertAlmostEqual(p.area(), 64.0, delta=0.01)

    def test_wire_rejects_a_degenerate_path(self):
        with self.assertRaises(ValueError):
            G.wire([(0.0, 0.0)], 4.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_library.TestWire -v`
Expected: FAIL — `module 'soidlc.geometry' has no attribute 'wire'`.

- [ ] **Step 3: Implement `wire`**

Add to `soidlc/geometry.py` (it already imports `math`; add the import if not):

```python
def wire(points: List[Pt], w: float) -> Polygon:
    """A polyline of width `w` as a closed polygon, with mitred joins.

    The miter is clipped to a bevel once it would extend past 4x the
    half-width, so a very sharp corner produces a blunt end rather than a
    spike that self-intersects.
    """
    pts: List[Pt] = []
    for p in points:
        if not pts or (abs(p[0] - pts[-1][0]) > 1e-9
                       or abs(p[1] - pts[-1][1]) > 1e-9):
            pts.append((float(p[0]), float(p[1])))
    if len(pts) < 2:
        raise ValueError("wire() needs at least two distinct points")

    half = w / 2.0

    def unit(a: Pt, b: Pt) -> Pt:
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        return (dx / L, dy / L)

    offsets: List[Pt] = []
    n = len(pts)
    for i in range(n):
        if i == 0:
            d = unit(pts[0], pts[1])
            nrm = (-d[1], d[0])
            offsets.append((nrm[0] * half, nrm[1] * half))
        elif i == n - 1:
            d = unit(pts[-2], pts[-1])
            nrm = (-d[1], d[0])
            offsets.append((nrm[0] * half, nrm[1] * half))
        else:
            d0 = unit(pts[i - 1], pts[i])
            d1 = unit(pts[i], pts[i + 1])
            n0 = (-d0[1], d0[0])
            n1 = (-d1[1], d1[0])
            mx, my = n0[0] + n1[0], n0[1] + n1[1]
            L = math.hypot(mx, my)
            if L < 1e-12:            # 180-degree reversal: use the incoming normal
                offsets.append((n0[0] * half, n0[1] * half))
                continue
            mx, my = mx / L, my / L
            cos_half = mx * n0[0] + my * n0[1]
            scale = half / max(cos_half, 0.25)      # bevel clip at 4x half-width
            offsets.append((mx * scale, my * scale))

    left = [(p[0] + o[0], p[1] + o[1]) for p, o in zip(pts, offsets)]
    right = [(p[0] - o[0], p[1] - o[1]) for p, o in zip(pts, offsets)]
    return Polygon(left + list(reversed(right))).normalized()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_library.TestWire -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add soidlc/geometry.py tests/test_library.py
git commit -m "feat(geom): wire() - a mitred polyline of finite width"
```

### Task 5: `geometry.circle()`, `arc()`, `annulus()`

**Files:**
- Modify: `soidlc/geometry.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `geometry.circle(R, n_seg=64, cx=0.0, cy=0.0) -> Polygon`
  - `geometry.annulus(R, w, n_seg=64, cx=0.0, cy=0.0) -> Polygon` — exterior at `R + w/2`, one hole at `R - w/2`, with `band=True` set (see Task 6).
  - `geometry.arc(R, w, a0_deg, a1_deg, n_seg=64, cx=0.0, cy=0.0) -> Polygon` — a partial band, no hole, so it needs no band handling.
  - `Polygon` gains a `band: bool = False` field.
  Used by `ring`/`disk` (Task 20).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_library.py`:

```python
import math                                            # noqa: E402


class TestCircularGeometry(unittest.TestCase):
    def test_circle_area_converges(self):
        self.assertAlmostEqual(G.circle(10.0, 256).area(), math.pi * 100.0,
                               delta=0.05)

    def test_annulus_area(self):
        R, w = 50.0, 10.0
        p = G.annulus(R, w, 256)
        expected = math.pi * ((R + w / 2) ** 2 - (R - w / 2) ** 2)
        self.assertAlmostEqual(p.area(), expected, delta=expected * 0.001)
        self.assertEqual(len(p.holes), 1)
        self.assertTrue(p.band)

    def test_annulus_rings_pair_one_to_one(self):
        p = G.annulus(50.0, 10.0, 32)
        self.assertEqual(len(p.exterior), len(p.holes[0]))

    def test_arc_is_a_quarter_of_the_band(self):
        R, w = 50.0, 10.0
        full = math.pi * ((R + w / 2) ** 2 - (R - w / 2) ** 2)
        q = G.arc(R, w, 0.0, 90.0, 256).area()
        self.assertAlmostEqual(q, full / 4.0, delta=full * 0.002)

    def test_band_flag_survives_translation(self):
        p = G.annulus(50.0, 10.0, 32).translated(5.0, 5.0)
        self.assertTrue(p.band)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_library.TestCircularGeometry -v`
Expected: FAIL — `module 'soidlc.geometry' has no attribute 'circle'`.

- [ ] **Step 3: Add the `band` field and propagate it**

In `soidlc/geometry.py`, extend the `Polygon` dataclass:

```python
@dataclass
class Polygon:
    exterior: Ring
    holes: List[Ring] = field(default_factory=list)
    band: bool = False
```

Then propagate `band` in every method that constructs a new `Polygon` —
`normalized`, `translated`, `rotated`, `mirrored` — by passing `self.band`
as the third argument. This matters: `elaborate.py` translates and rotates
every shape during placement, and a ring that loses its flag would fall back
to the non-manifold bridging path.

- [ ] **Step 4: Implement the three generators**

```python
def circle(R: float, n_seg: int = 64,
           cx: float = 0.0, cy: float = 0.0) -> Polygon:
    """A regular-polygon approximation of a disk, CCW."""
    if n_seg < 3:
        raise ValueError("circle() needs at least 3 segments")
    ring = [(cx + R * math.cos(2 * math.pi * i / n_seg),
             cy + R * math.sin(2 * math.pi * i / n_seg))
            for i in range(n_seg)]
    return Polygon(ring)


def annulus(R: float, w: float, n_seg: int = 64,
            cx: float = 0.0, cy: float = 0.0) -> Polygon:
    """A ring of centreline radius R and radial width w.

    The hole is emitted with the SAME vertex count and the same angular
    sampling as the exterior, so vertex i of one pairs with vertex i of the
    other.  `band=True` tells the mesher it may use the quad-band cap
    triangulation instead of hole bridging (see mesh.triangulate_with_holes).
    """
    if w <= 0 or w >= 2 * R:
        raise ValueError("annulus() needs 0 < w < 2R")
    outer = circle(R + w / 2.0, n_seg, cx, cy).exterior
    inner = circle(R - w / 2.0, n_seg, cx, cy).exterior
    return Polygon(list(outer), [list(reversed(inner))], band=True)


def arc(R: float, w: float, a0_deg: float, a1_deg: float, n_seg: int = 64,
        cx: float = 0.0, cy: float = 0.0) -> Polygon:
    """A partial band from a0 to a1 degrees; a simple ring, no hole."""
    a0 = math.radians(a0_deg)
    a1 = math.radians(a1_deg)
    k = max(2, int(round(n_seg * abs(a1_deg - a0_deg) / 360.0)) + 1)
    ang = [a0 + (a1 - a0) * i / (k - 1) for i in range(k)]
    ro, ri = R + w / 2.0, R - w / 2.0
    outer = [(cx + ro * math.cos(a), cy + ro * math.sin(a)) for a in ang]
    inner = [(cx + ri * math.cos(a), cy + ri * math.sin(a)) for a in ang]
    return Polygon(outer + list(reversed(inner))).normalized()
```

Note `annulus` reverses the inner ring so the hole is wound opposite to the
exterior, which is what `normalized()` and the connectivity code expect. Task 6
un-reverses it when pairing.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_library.TestCircularGeometry -v`
Expected: PASS, 5 tests.

- [ ] **Step 6: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK` — in particular the existing watertightness tests, which now
run against a `Polygon` that has one more field.

- [ ] **Step 7: Commit**

```bash
git add soidlc/geometry.py tests/test_library.py
git commit -m "feat(geom): circle/annulus/arc generators with a band flag"
```

### Task 6: Fix non-manifold ring extrusion

**Files:**
- Modify: `soidlc/mesh.py:157-165`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `Polygon.band` from Task 5.
- Produces: a watertight extrusion for every `annulus`. Task 20's `ring` primitive depends on it.

**Background — measured, not assumed.** Extruding an annulus today yields exactly **4 non-manifold edges**, constant across `n_seg` ∈ {8, 16, 64, 128}. They are not boundary edges (count 1) but edges shared by **4** triangles: `_bridge_one` (`mesh.py:183`) cuts a zero-width slit from the hole to the exterior and traverses it twice, and `_extrude_general`'s `vert()` cache (`mesh.py:314`) dedupes by rounded coordinate, collapsing the slit's two sides into one edge. The surface is closed but non-manifold, which breaks STL consumers.

The general fix to hole bridging is out of scope. Instead, a ring — the only holed non-rectilinear shape the library generates — gets a cap triangulation that needs no slit: pair vertex `i` of the exterior with vertex `i` of the hole and emit two triangles per segment. **Verified during planning: 0 non-manifold edges, every edge shared by exactly 2 triangles, for n_seg ∈ {8, 16, 64}.** General bridged holes keep their existing behaviour and their existing limitation.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_library.py`:

```python
from collections import Counter                        # noqa: E402

from soidlc import mesh as M                           # noqa: E402


class TestRingMesh(unittest.TestCase):
    def _bad_edges(self, mesh):
        edges = Counter()
        for (a, b, c) in mesh.triangles:
            for e in ((a, b), (b, c), (c, a)):
                edges[frozenset(e)] += 1
        return sum(1 for v in edges.values() if v != 2)

    def test_annulus_extrudes_watertight(self):
        for n_seg in (8, 16, 64):
            with self.subTest(n_seg=n_seg):
                poly = G.annulus(50.0, 10.0, n_seg)
                mesh = M.extrude_polygon(poly, 0.0, 25.0, "DEVICE")
                self.assertEqual(self._bad_edges(mesh), 0)
                self.assertGreater(len(mesh.triangles), 0)

    def test_disk_extrudes_watertight(self):
        mesh = M.extrude_polygon(G.circle(50.0, 64), 0.0, 25.0, "DEVICE")
        self.assertEqual(self._bad_edges(mesh), 0)

    def test_tilted_bar_extrudes_watertight(self):
        # the chevron case: a rectangle rotated off-axis leaves the voxel path
        mesh = M.extrude_polygon(G.rect(100.0, 6.0).rotated(8.0),
                                 0.0, 25.0, "DEVICE")
        self.assertEqual(self._bad_edges(mesh), 0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_library.TestRingMesh -v`
Expected: `test_annulus_extrudes_watertight` FAILS with `4 != 0` for every
`n_seg`. The other two PASS already (verified during planning) — they are
regression guards, not new behaviour.

- [ ] **Step 3: Add the band cap path**

In `soidlc/mesh.py`, replace `triangulate_with_holes`:

```python
def triangulate_with_holes(poly: G.Polygon) -> List[Tuple[G.Pt, G.Pt, G.Pt]]:
    """Return cap triangles (as coordinate triples) for a polygon with holes."""
    rectilinear = (_is_rectilinear(poly.exterior)
                   and all(_is_rectilinear(h) for h in poly.holes))
    if rectilinear:
        return _decompose_rectilinear(poly)
    if (getattr(poly, "band", False) and len(poly.holes) == 1
            and len(poly.holes[0]) == len(poly.exterior)):
        return _band_caps(poly)
    ring = _bridge_holes(poly)
    idx = triangulate_simple(ring)
    return [(ring[a], ring[b], ring[c]) for (a, b, c) in idx]


def _band_caps(poly: G.Polygon) -> List[Tuple[G.Pt, G.Pt, G.Pt]]:
    """Cap triangles for a ring whose hole pairs 1:1 with its exterior.

    Avoids the bridge slit that `_bridge_holes` cuts, which collapses into a
    4-triangle edge once coincident vertices are merged.  Two triangles per
    segment, sharing only real edges -> every edge is used exactly twice.
    """
    outer = list(poly.exterior)
    # the hole is stored wound opposite to the exterior; undo that so index i
    # of each ring is the same angular position
    inner = list(reversed(poly.holes[0]))
    n = len(outer)
    tris: List[Tuple[G.Pt, G.Pt, G.Pt]] = []
    for i in range(n):
        a, b = outer[i], outer[(i + 1) % n]
        c, d = inner[i], inner[(i + 1) % n]
        tris.append((a, b, d))
        tris.append((a, d, c))
    return tris
```

The 1:1 length check is a guard, not an assumption: if a ring is ever built
with mismatched ring lengths the code falls through to bridging rather than
producing garbage.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_library.TestRingMesh -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Check the cap winding is outward**

Run:
```bash
.venv/bin/python -c "
from soidlc import geometry as G, mesh as M, exporters
m = M.extrude_polygon(G.annulus(50.0, 10.0, 64), 0.0, 25.0, 'DEVICE')
import tempfile, os
p = os.path.join(tempfile.mkdtemp(), 'ring.stl')
exporters.write_stl(m, p)
print('stl bytes:', os.path.getsize(p))
zs = [v[2] for v in m.vertices]
print('z range:', min(zs), max(zs))
"
```
Expected: a non-zero STL and `z range: 0.0 25.0`. If the top cap comes out
inward-facing, swap `(a, b, d)`/`(a, d, c)` to `(a, d, b)`/`(a, c, d)` — the
watertightness test passes either way, so this visual check is what catches it.

- [ ] **Step 6: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add soidlc/mesh.py tests/test_library.py
git commit -m "fix(mesh): watertight ring extrusion via band caps, not a bridge slit"
```

---

## Phase 1 — flexures: the standard library earns its keep

### Task 7: `frame`, and migrate the existing examples onto the library

**Files:**
- Modify: `soidlc/lib/flexures.soidl`, `examples/comb_resonator.soidl`, `examples/accelerometer.soidl`, `examples/gyroscope_2mass.soidl`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `import` (Task 3), `guided_beam` (Task 3).
- Produces: `frame(W, H, bar)` in `soidlc/lib/flexures.soidl` — a square ring of four bars, used by `gyroscope_2mass` and by Task 22's ring gyro.

This task is the proof that option C was worth it: three examples get shorter and one stops carrying hand-computed coordinates.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_library.py`:

```python
class TestFrame(unittest.TestCase):
    def test_frame_is_a_hollow_square(self):
        src = TestImportResolution.PROC + """
          import "flexures.soidl";
          device d {
            inst F = frame(W = 680 um, H = 680 um, bar = 40 um) at (0, 0);
            inst A = anchor(30 um, 30 um) at (0, 320 um);
            net GND = F | A;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        x0, y0, x1, y1 = _bbox(art)
        self.assertAlmostEqual(x1 - x0, 680.0, delta=1.0)
        # hollow: total silicon area is the ring, not the full square
        area = sum(s.polygon.area() for s in art.result.shapes
                   if s.layer == "DEVICE")
        self.assertLess(area, 680.0 * 680.0 * 0.5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_library.TestFrame -v`
Expected: FAIL — `frame` is neither a component nor a primitive.

- [ ] **Step 3: Add `frame` to the library**

Append to `soidlc/lib/flexures.soidl`:

```soidl
// A square ring of four bars: the outer drive frame of a decoupled gyro,
// or any structure that must surround an inner mass.  W/H are the OUTER
// dimensions; `bar` is the wall width.  Bars overlap at the corners, which
// is what makes the ring one electrically continuous island.
component frame(W = 680 um, H = 680 um, bar = 40 um) {
  mech port body;

  geometry {
    beam(W, bar, dir = x) at (0, H/2 - bar/2);
    beam(W, bar, dir = x) at (0, bar/2 - H/2);
    beam(H - 2*bar, bar, dir = y) at (bar/2 - W/2, 0);
    beam(H - 2*bar, bar, dir = y) at (W/2 - bar/2, 0);
  }

  check bar >= 2 um;
  check W > 2*bar;
  check H > 2*bar;
}
```

The vertical bars are shortened by `2*bar` and the horizontal ones span the
full width, so the corners are covered exactly once — no double-counted area,
but still abutting, which is what `connectivity.touches` needs to fuse them
into one island.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_library.TestFrame -v`
Expected: PASS.

- [ ] **Step 5: Migrate `comb_resonator.soidl`**

Delete the inline `component beam_flexure(...) { ... }` block and add
`import "flexures.soidl";` above the `process` block. Change the instance to:

```soidl
  inst S  = array(guided_beam(L = 220 um, w = 4 um, n_beams = 2), count = 4,
                  place = corners(M));
```

`guided_beam`'s default `pitch = 12 um` equals the old `2*w + 8 um` at
`w = 4 um`, so the geometry is unchanged.

- [ ] **Step 6: Migrate `accelerometer.soidl`**

Delete the inline `component suspension(...)`. Add `import "flexures.soidl";`.
The old `suspension` was a single beam along **x**; `guided_beam` runs along
**y**. Rather than rotating, give the library a matching one-beam call and let
`attach`/`corners` placement handle orientation:

```soidl
  inst SUS  = array(guided_beam(L = 300 um, w = 5 um, n_beams = 1), count = 4,
                    place = corners(MASS));
```

Then run the example and compare against the committed reference before/after:

```bash
.venv/bin/python -m soidlc.cli examples/accelerometer.soidl -o /tmp/accel_new
```

Expected: `soidlc: ERROR:` lines absent, and the reported `f0` within a few
percent of the pre-migration value. If the connectivity check reports floating
islands, the anchor overlap is wrong for `n_beams = 1` — fix `guided_beam`, not
the example.

- [ ] **Step 7: Migrate `gyroscope_2mass.soidl`**

Replace the four hand-placed frame bars

```soidl
    beam(680 um, 40 um, dir = x) at (0,  320 um);
    beam(680 um, 40 um, dir = x) at (0, 0 - 320 um);
    beam(680 um, 40 um, dir = y) at (0 - 320 um, 0);
    beam(680 um, 40 um, dir = y) at (320 um, 0);
```

with a single call inside `decoupled_core`'s `geometry` block:

```soidl
    frame(680 um, 680 um, 40 um) at (0, 0);
```

Note the original wrote `beam(680 um, 40 um, dir = y)` for the side bars —
680 long, overlapping the top and bottom bars at the corners. `frame` shortens
them to `H - 2*bar`, so the corner silicon is covered once instead of twice.
The released area therefore drops slightly and `f0` rises slightly. Record the
before/after `f0` in the commit message; if it moves more than 5%, the frame
geometry does not match and needs fixing.

- [ ] **Step 8: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`. `test_examples_have_no_errors` (Task 1) now guards all three
migrated files, and `test_resonator_meets_spec` guards that closure still
hits 20 kHz within 1.5% after the comb_resonator migration.

- [ ] **Step 9: Commit**

```bash
git add soidlc/lib/flexures.soidl examples/comb_resonator.soidl \
        examples/accelerometer.soidl examples/gyroscope_2mass.soidl \
        tests/test_library.py
git commit -m "refactor(examples): use the library flexure and frame instead of inline copies"
```

### Task 8: `folded_flexure` and the Tang resonator

**Files:**
- Modify: `soidlc/lib/flexures.soidl`
- Create: `examples/folded_flexure_resonator.soidl`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `import`, `guided_beam`.
- Produces: `folded_flexure(L, w, n_folds, truss_w)` with `derive k.x = n_folds * E * t * w^3 / (2 * L^3)`.

**Physics the test arbitrates.** A folded flexure is two guided beams of length
`L` in **series** per fold, so one fold is half as stiff as a single guided
beam: `k_fold = (1/2) * 12EI/L^3 = E t w^3 / (2 L^3)`. `n_folds` such folds sit
in **parallel**, multiplying it. If the FEM in Step 6 disagrees by more than
15%, the formula is wrong — fix the formula, not the tolerance.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_library.py`:

```python
class TestFoldedFlexure(unittest.TestCase):
    def test_folded_is_softer_than_guided_at_equal_length(self):
        """A fold puts two beams in series, so it must be the more compliant."""
        def k_of(component):
            src = TestImportResolution.PROC + f"""
              import "flexures.soidl";
              device d {{
                inst M = plate(200 um, 200 um) at (0, 0);
                inst S = array({component}, count = 4, place = corners(M));
                net GND = M | S.fixed;
                constraint anchored(S.fixed);
              }}
            """
            art = compile_source(src)
            self.assertEqual(art.errors, [], art.errors)
            return art.model["k"].value

        k_guided = k_of("guided_beam(L = 200 um, w = 4 um, n_beams = 2)")
        k_folded = k_of("folded_flexure(L = 200 um, w = 4 um, n_folds = 2)")
        self.assertLess(k_folded, k_guided)
        self.assertAlmostEqual(k_folded / k_guided, 0.5, delta=0.05)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_library.TestFoldedFlexure -v`
Expected: FAIL — `folded_flexure` is not defined.

- [ ] **Step 3: Add `folded_flexure` to the library**

Append to `soidlc/lib/flexures.soidl`:

```soidl
// Folded flexure: each fold is an outboard beam and an inboard beam joined by
// a truss at the far end, so the pair acts as two guided beams in series.
// Folding is what relieves axial stress -- a straight guided beam stretches as
// it deflects and stiffens; a folded one lets the truss take up the slack, so
// f0 stays put over the stroke.
//
//   k_fold = (1/2) * 12*E*I/L^3 = E*t*w^3 / (2*L^3),  n_folds in parallel
component folded_flexure(L = 200 um, w = 4 um, n_folds = 2,
                         truss_w = 8 um, pitch = 20 um) {
  mech port fixed, shuttle;

  derive k.x = n_folds * (process.DEVICE.E * process.DEVICE.thickness * w^3)
               / (2 * L^3);

  geometry {
    repeat i in 0..n_folds-1 {
      // inboard beam: from the shuttle down to the fold truss
      beam(L, w, dir = y) at (i * pitch, 2 um - L/2);
      // outboard beam: from the fold truss back up to the anchor
      beam(L, w, dir = y) at (i * pitch + pitch/2, 2 um - L/2);
      // fold truss joining the two beam tips at the far end
      beam(pitch/2 + w, truss_w, dir = x)
          at (i * pitch + pitch/4, 2 um - L - truss_w/2 + w/2);
    }
    // shuttle-side tie bar across all inboard beams
    beam((n_folds - 1) * pitch + w, truss_w, dir = x)
        at ((n_folds - 1) * pitch/2, 0);
    // anchor spanning the outboard beams
    anchor((n_folds - 1) * pitch + pitch/2 + w + 12 um, 20 um)
        at ((n_folds - 1) * pitch/2 + pitch/4, 12 um);
  }

  check w >= 2 um;
  check pitch > 2*w;
}
```

The anchor sits **inboard** (at `+12 um`, above the shuttle tie) because a
folded flexure anchors at the same end it attaches from — that is the fold.
Verify with the connectivity check in Step 4: if the anchor does not abut the
outboard beams, the elaborator reports a floating island.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_library.TestFoldedFlexure -v`
Expected: PASS. If it reports `soidlc: ERROR: ... island ... not anchored`,
the anchor rectangle does not overlap the outboard beams — adjust its size and
position until it does, then re-run.

- [ ] **Step 5: Write the example**

Create `examples/folded_flexure_resonator.soidl`:

```soidl
// Tang-style folded-beam comb resonator.
//
// The classic MEMS resonator: a shuttle on four folded flexures, driven and
// sensed by combs on opposite edges.  The fold is the point -- a straight
// guided beam stretches as it deflects, so its stiffness (and f0) climbs with
// amplitude; the fold lets the truss absorb that extension, keeping f0 flat.
// Compile:  python -m soidlc.cli examples/folded_flexure_resonator.soidl

import "flexures.soidl";

process soi25 {
  stack {
    layer METAL  { thickness = 0.52 um; material = Au }
    layer DEVICE { thickness = 25 um;   material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;    material = SiO2; eps_r = 3.9 }
    layer HANDLE { thickness = 400 um;  material = Si }
  }

  masks {
    mask SOID     -> etch(DEVICE, through)
    mask PADMETAL -> deposit(METAL)
  }

  rules {
    release { undercut = 1.5 um; max_solid_span = 30 um;
              hole_size = 5 um; hole_pitch = 25 um }
    drie    { aspect_ratio_max = 20; footing_margin = 1 um }
  }
}

device folded_resonator {
  param f0_target = 18 kHz;
  param V_drive   = 30 V;

  inst M  = plate(300 um, 300 um) at (0, 0);
  inst S  = array(folded_flexure(L = 200 um, w = 4 um, n_folds = 2),
                  count = 4, place = corners(M));
  inst D1 = combdrive(N = 24) attach (rotor -> M.left);
  inst D2 = combdrive(N = 24) attach (rotor -> M.right);

  solve S.L such that f_res(M, S) == f0_target within 2%;

  require f_res(M, S) >= 12 kHz;
  require stroke_max(V_drive) >= 3 um;

  net DRIVE = D1.stator;
  net SENSE = D2.stator;
  net GND   = M | S.fixed;

  isolate DRIVE from GND by trench;
  constraint released(M, S);
  constraint anchored(D1.stator, D2.stator, S.fixed);
}
```

- [ ] **Step 6: Compile it and validate against FEM**

Run:
```bash
.venv/bin/python -m soidlc.cli examples/folded_flexure_resonator.soidl \
    -o /tmp/folded --fem --fem-h 15
```
Expected: no `soidlc: ERROR:` lines; a `closure: solved S.L` line; and an
`fem ... vs lumped ... (N%)` line with `|N| < 15`. If the FEM disagrees by
more than 15%, the `k.x` formula in `folded_flexure` is wrong — that is exactly
what this step exists to catch. Re-derive it before touching the tolerance.

- [ ] **Step 7: Add the FEM regression test**

Add to `tests/test_library.py`:

```python
class TestFoldedFlexureFEM(unittest.TestCase):
    """Validates the folded-flexure stiffness formula against plane-stress FEM."""

    def test_lumped_f0_agrees_with_fem(self):
        with open(os.path.join(EX, "folded_flexure_resonator.soidl")) as f:
            art = compile_source(f.read(), fem=True, fem_h=15.0,
                                 base_dir=EX)
        self.assertEqual(art.errors, [], art.errors)
        cmp = [l for l in art.report
               if l.startswith("fem") and "vs lumped" in l]
        self.assertTrue(cmp, art.report)
        pct = float(cmp[0].rsplit("(", 1)[1].rstrip("%)"))
        self.assertLess(abs(pct), 15.0, cmp[0])
```

- [ ] **Step 8: Run it**

Run: `.venv/bin/python -m unittest tests.test_library.TestFoldedFlexureFEM -v`
Expected: PASS. This test takes ~10-30 s (gmsh + ARPACK).

- [ ] **Step 9: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`.

- [ ] **Step 10: Commit**

```bash
git add soidlc/lib/flexures.soidl examples/folded_flexure_resonator.soidl \
        tests/test_library.py
git commit -m "feat(lib): folded flexure + Tang resonator example, FEM-validated"
```

### Task 9: `serpentine`, `crab_leg`, and a low-g accelerometer

**Files:**
- Modify: `soidlc/lib/flexures.soidl`
- Create: `examples/low_g_accel.soidl`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `geometry.wire` (Task 4) is **not** used here — `serpentine` is built from `beam` calls, so it stays rectilinear and keeps the fast voxel mesher. `wire` is for the inclined shapes in Task 12.
- Produces: `serpentine(L, w, n_turns, pitch)` and `crab_leg(Lx, Ly, w)` in `soidlc/lib/flexures.soidl`.

**Physics.** A serpentine of `n_turns` meanders puts `n_turns` guided spans of
length `L` in series: `k = E t w^3 / (n_turns * L^3)`. A crab-leg has a thigh
`Lx` (bending) and a shin `Ly` (which twists the thigh's end):
`k_x = 12 E I / (Lx^3 * (1 + 3*Ly/Lx))` with `I = t w^3 / 12`, i.e.
`k_x = E t w^3 / (Lx^3 * (1 + 3*Ly/Lx))`. Both are deliberately compliant,
which is the point of a low-g device.

- [ ] **Step 1: Write the failing test**

```python
class TestCompliantFlexures(unittest.TestCase):
    def _k(self, component):
        src = TestImportResolution.PROC + f"""
          import "flexures.soidl";
          device d {{
            inst M = plate(300 um, 300 um) at (0, 0);
            inst S = array({component}, count = 4, place = corners(M));
            net GND = M | S.fixed;
            constraint anchored(S.fixed);
          }}
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        return art.model["k"].value

    def test_more_turns_is_softer(self):
        k2 = self._k("serpentine(L = 120 um, w = 3 um, n_turns = 2)")
        k6 = self._k("serpentine(L = 120 um, w = 3 um, n_turns = 6)")
        self.assertLess(k6, k2)
        self.assertAlmostEqual(k2 / k6, 3.0, delta=0.2)

    def test_crab_leg_shin_softens_it(self):
        short = self._k("crab_leg(Lx = 150 um, Ly = 20 um, w = 4 um)")
        long = self._k("crab_leg(Lx = 150 um, Ly = 120 um, w = 4 um)")
        self.assertLess(long, short)
```

Run: `.venv/bin/python -m unittest tests.test_library.TestCompliantFlexures -v`
Expected: FAIL — neither component exists.

- [ ] **Step 2: Add both components**

Append to `soidlc/lib/flexures.soidl`:

```soidl
// Serpentine (meander) spring: n_turns guided spans in series, so it is
// n_turns times softer than one span.  Used where a low-g device needs a very
// compliant suspension without a metre of straight beam.
//   k = E*t*w^3 / (n_turns * L^3)
component serpentine(L = 120 um, w = 3 um, n_turns = 4, pitch = 12 um) {
  mech port fixed, shuttle;

  derive k.x = (process.DEVICE.E * process.DEVICE.thickness * w^3)
               / (n_turns * L^3);

  geometry {
    // n_turns long spans along y, joined alternately at top and bottom by
    // short connectors along x -- the meander
    repeat i in 0..n_turns-1 {
      beam(L, w, dir = y) at (i * pitch, 2 um - L/2);
    }
    repeat j in 0..n_turns-2 {
      beam(pitch + w, w, dir = x)
          at (j * pitch + pitch/2, 2 um - L + w/2);
    }
    anchor(20 um, 20 um) at ((n_turns - 1) * pitch, 12 um);
  }

  check w >= 2 um;
  check pitch > 2*w;
  check n_turns >= 2;
}

// Crab-leg flexure: a thigh of length Lx that bends, and a shin of length Ly
// whose torsion softens the thigh's end restraint.  Lengthening the shin makes
// it softer without lengthening the thigh.
//   k_x = E*t*w^3 / (Lx^3 * (1 + 3*Ly/Lx))
component crab_leg(Lx = 150 um, Ly = 60 um, w = 4 um) {
  mech port fixed, shuttle;

  derive k.x = (process.DEVICE.E * process.DEVICE.thickness * w^3)
               / (Lx^3 * (1 + 3*Ly/Lx));

  geometry {
    beam(Lx, w, dir = x) at (2 um - Lx/2, 0);              // thigh
    beam(Ly, w, dir = y) at (2 um - Lx + w/2, 0 - Ly/2);   // shin
    anchor(20 um, 20 um) at (2 um - Lx + w/2, 0 - Ly - 8 um);
  }

  check w >= 2 um;
  check Lx > 4*w;
}
```

Run the test again. Expected: PASS. If the connectivity check reports a
floating island, the anchor does not abut the last beam — adjust and re-run.

- [ ] **Step 3: Write the low-g example**

Create `examples/low_g_accel.soidl`:

```soidl
// Low-g in-plane accelerometer on serpentine suspensions.
//
// Sensitivity goes as m/k, so a low-g device wants a heavy mass on a very
// soft spring.  A straight beam soft enough would not fit on the die; a
// serpentine folds the same compliance into a corner.
// Compile:  python -m soidlc.cli examples/low_g_accel.soidl

import "flexures.soidl";

process soi25 {
  stack {
    layer METAL  { thickness = 1 um;   material = Au }
    layer DEVICE { thickness = 25 um;  material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;   material = SiO2 }
    layer HANDLE { thickness = 400 um; material = Si }
  }
  rules {
    release { hole_size = 6 um; hole_pitch = 30 um; max_solid_span = 40 um }
  }
}

device low_g_accel {
  inst MASS  = plate(600 um, 400 um) at (0, 0);
  inst SUS   = array(serpentine(L = 140 um, w = 3 um, n_turns = 5),
                     count = 4, place = corners(MASS));
  inst SENSE = combdrive(N = 40, Lf = 50 um) attach (rotor -> MASS.top);
  inst PAD   = via_metal(120 um, 120 um) at (420 um, 260 um);

  // a low-g device is defined by its low resonance
  require f_res(MASS, SUS) <= 3 kHz;

  net SIG = SENSE.stator;
  net GND = MASS | SUS.fixed;
  isolate SIG from GND by trench;
  constraint released(MASS, SUS);
  constraint anchored(SUS.fixed);
}
```

- [ ] **Step 4: Compile it**

Run: `.venv/bin/python -m soidlc.cli examples/low_g_accel.soidl -o /tmp/lowg`
Expected: no `soidlc: ERROR:` lines, and a reported `f0` below 3 kHz. If
`require` is violated the compiler exits 2 and says so — soften the
suspension (more turns) or widen the mass rather than deleting the `require`.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`. `test_examples_have_no_errors` now covers the new file.

- [ ] **Step 6: Commit**

```bash
git add soidlc/lib/flexures.soidl examples/low_g_accel.soidl tests/test_library.py
git commit -m "feat(lib): serpentine and crab-leg flexures + low-g accelerometer"
```

### Task 10: Double-ended tuning fork resonator

**Files:**
- Create: `examples/tuning_fork_detf.soidl`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `import`, `guided_beam`. Adds no new library component — a DETF is
  two tines anchored at both ends, expressible with `beam` and `anchor`.
- Produces: the second FEM-validated example. Nothing depends on it.

**Why it earns a slot.** Every other resonator here is a mass on a spring. A
DETF is a *distributed* resonator: its frequency comes from the tine geometry,
not from a lumped mass, so it is the case where the lumped model is expected to
be least accurate. It is the honest stress test of `_extract_device_model`.

- [ ] **Step 1: Write the example**

Create `examples/tuning_fork_detf.soidl`:

```soidl
// Double-ended tuning fork (DETF).
//
// Two parallel tines clamped at BOTH ends, driven anti-phase so their
// reaction moments cancel at the anchors -- that cancellation is what gives a
// DETF its high Q.  The frequency is set by the tine, not by a proof mass,
// which makes it the sharpest test of the lumped model in this repo.
// Compile:  python -m soidlc.cli examples/tuning_fork_detf.soidl --fem

import "flexures.soidl";

process soi25 {
  stack {
    layer METAL  { thickness = 0.52 um; material = Au }
    layer DEVICE { thickness = 25 um;   material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;    material = SiO2 }
    layer HANDLE { thickness = 400 um;  material = Si }
  }
  rules {
    release { hole_size = 5 um; hole_pitch = 25 um; max_solid_span = 30 um }
  }
}

// Two tines of length L and width w, separated by `gap`, joined to a shared
// anchor pad at each end.  A clamped-clamped beam's first mode is
//   f0 = (4.730^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A))
// with I = t*w^3/12 and A = t*w, so f0 scales as w/L^2.
component detf_tines(L = 400 um, w = 6 um, gap = 20 um) {
  mech port fixed, tine;

  // effective lumped stiffness of two clamped-clamped tines in parallel:
  // k = 2 * 192*E*I/L^3 = 32*E*t*w^3/L^3
  derive k.x = 32 * (process.DEVICE.E * process.DEVICE.thickness * w^3) / L^3;

  geometry {
    beam(L, w, dir = y) at (0 - gap/2 - w/2, 0);
    beam(L, w, dir = y) at (gap/2 + w/2, 0);
    // shared anchor pads clamping both tines at each end (4 um overlap)
    anchor(gap + 2*w + 40 um, 40 um) at (0, L/2 + 20 um - 4 um);
    anchor(gap + 2*w + 40 um, 40 um) at (0, 4 um - L/2 - 20 um);
  }

  check w >= 2 um;
  check L/w >= 20 warn "short tine: the beam formula assumes slenderness";
}

device tuning_fork_detf {
  inst T  = detf_tines(L = 400 um, w = 6 um, gap = 20 um) at (0, 0);
  inst D  = combdrive(N = 20, Lf = 30 um, ov = 12 um)
            attach (rotor -> T.left);

  require f_res(T) >= 50 kHz;

  net DRIVE = D.stator;
  net GND   = T | T.fixed;

  isolate DRIVE from GND by trench;
  constraint anchored(T.fixed, D.stator);
}
```

- [ ] **Step 2: Compile it**

Run: `.venv/bin/python -m soidlc.cli examples/tuning_fork_detf.soidl -o /tmp/detf`
Expected: no `soidlc: ERROR:` lines.

A DETF has no released proof mass in the usual sense — the tines themselves are
the moving mass. Confirm the report shows a non-zero `m`; if `m` is missing,
`_extract_device_model` found no released DEVICE polygons and the tines were
classified as anchored. In that case the anchor pads overlap the tines too far;
reduce the overlap to 4 um.

- [ ] **Step 3: FEM validation**

Run:
```bash
.venv/bin/python -m soidlc.cli examples/tuning_fork_detf.soidl \
    -o /tmp/detf --fem --fem-h 10
```
Expected: an `fem ... vs lumped ... (N%)` line.

**This is the one place where a wide disagreement is informative rather than a
bug.** A DETF is distributed, so the lumped `sqrt(k/m)` estimate is expected to
be less accurate than for a mass-on-spring. Record the measured percentage.
If `|N| < 15`, add the regression test in Step 4 at 15%. If it lands between
15% and 30%, **do not loosen the shared tolerance** — instead write the test
against the measured value with a documented comment explaining that a
distributed resonator is the known-worst case for the lumped model, and note it
in the spec's Risks section. If it exceeds 30%, the `k.x` formula is wrong.

- [ ] **Step 4: Add the regression test**

```python
class TestDETF(unittest.TestCase):
    """A DETF is distributed, not lumped: the widest lumped-vs-FEM gap here."""

    def test_lumped_f0_agrees_with_fem(self):
        with open(os.path.join(EX, "tuning_fork_detf.soidl")) as f:
            art = compile_source(f.read(), fem=True, fem_h=10.0, base_dir=EX)
        self.assertEqual(art.errors, [], art.errors)
        cmp = [l for l in art.report
               if l.startswith("fem") and "vs lumped" in l]
        self.assertTrue(cmp, art.report)
        pct = float(cmp[0].rsplit("(", 1)[1].rstrip("%)"))
        self.assertLess(abs(pct), 15.0, cmp[0])
```

Adjust the 15.0 only per Step 3's rule, with the comment it requires.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add examples/tuning_fork_detf.soidl tests/test_library.py
git commit -m "feat(examples): double-ended tuning fork, FEM-validated"
```

---

## Phase 2 — actuators

### Task 11: `parallel_plate` primitive

**Files:**
- Modify: `soidlc/primitives.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: nothing.
- Produces: primitive `parallel_plate(W, H, g, n)` registered in `PRIMITIVES`, emitting shapes labelled `rotor_plate` (released) and `stator_plate` (anchored). Task 13 reads those labels.

**Why it is not a comb.** A comb's capacitance is linear in displacement, so
its force is constant with position. A parallel plate's capacitance goes as
`1/(g - x)`, so force rises as it closes and the structure snaps shut past
`x = g/3` — pull-in. That nonlinearity is the whole reason RF switches and
gap-closing sensors are built this way, and Task 13 gives SOIDL the metric.

- [ ] **Step 1: Write the failing test**

```python
class TestParallelPlate(unittest.TestCase):
    def test_emits_a_released_and_an_anchored_plate(self):
        from soidlc import primitives as P
        from soidlc.units import Quantity

        def um(v):
            return Quantity(v * 1e-6, (1, 0, 0, 0))

        ctx = P.PrimitiveCtx()
        shapes = P.PRIMITIVES["parallel_plate"](
            [], {"W": um(100), "H": um(40), "g": um(3), "n": 2}, ctx)
        labels = sorted({s.label for s in shapes})
        self.assertEqual(labels, ["rotor_plate", "stator_plate"])
        mechs = {s.label: s.mech for s in shapes}
        self.assertEqual(mechs["rotor_plate"], "released")
        self.assertEqual(mechs["stator_plate"], "anchored")

    def test_gap_is_respected(self):
        from soidlc import primitives as P
        from soidlc.units import Quantity

        def um(v):
            return Quantity(v * 1e-6, (1, 0, 0, 0))

        ctx = P.PrimitiveCtx()
        shapes = P.PRIMITIVES["parallel_plate"](
            [], {"W": um(100), "H": um(40), "g": um(3), "n": 1}, ctx)
        rot = [s for s in shapes if s.label == "rotor_plate"][0]
        sta = [s for s in shapes if s.label == "stator_plate"][0]
        # facing edges are exactly g apart along y
        self.assertAlmostEqual(sta.polygon.bbox()[1] - rot.polygon.bbox()[3],
                               3.0, places=3)
```

Run: `.venv/bin/python -m unittest tests.test_library.TestParallelPlate -v`
Expected: FAIL — `KeyError: 'parallel_plate'`.

- [ ] **Step 2: Implement the primitive**

Add to `soidlc/primitives.py`, before the `PRIMITIVES` dict:

```python
def prim_parallel_plate(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """Gap-closing electrode pairs: released plates facing anchored ones.

    Unlike a comb, capacitance goes as 1/(g - x), so the force rises as the
    gap closes and the pair snaps shut past x = g/3 (pull-in).  metrics.pull_in
    computes that collapse voltage.
    """
    W = _um(_arg(args, kwargs, 0, "W", Quantity(100e-6, (1, 0, 0, 0))))
    H = _um(_arg(args, kwargs, 1, "H", Quantity(40e-6, (1, 0, 0, 0))))
    g = _um(_arg(args, kwargs, 2, "g", Quantity(3e-6, (1, 0, 0, 0))))
    n = int(round(_num(_arg(args, kwargs, 3, "n", 1))))

    shapes: List[G.Shape] = []
    pitch = 2 * H + 2 * g
    y = -(n - 1) * pitch / 2.0
    for _ in range(n):
        shapes.append(G.Shape(ctx.device_layer, G.rect(W, H, 0.0, y),
                              "rotor_plate", mech="released"))
        shapes.append(G.Shape(ctx.device_layer,
                              G.rect(W, H, 0.0, y + H + g),
                              "stator_plate", mech="anchored"))
        y += pitch
    return shapes
```

and register it:

```python
    "parallel_plate": prim_parallel_plate,
```

Run the test. Expected: PASS, 2 tests.

- [ ] **Step 3: Full suite and commit**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`.

```bash
git add soidlc/primitives.py tests/test_library.py
git commit -m "feat(prim): parallel_plate gap-closing electrode pairs"
```

### Task 12: `chevron` and `hot_arm` thermal actuators

**Files:**
- Modify: `soidlc/primitives.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `geometry.wire` (Task 4).
- Produces: primitives `chevron(n, L, w, angle)` and `hot_arm(L, w_hot, w_cold, g)`, emitting shapes labelled `chevron_beam` / `chevron_shuttle` and `hot_arm` / `cold_arm`. Task 13 reads them.

**Geometry warning, verified during planning.** A chevron's beams are inclined
by a few degrees, so their coordinates are not axis-aligned and
`mesh._is_rectilinear` rejects them — they extrude through `_extrude_general`.
That path **is** watertight for a simple ring (measured: a rectangle rotated 8°
gives 0 non-manifold edges); only holed polygons had the defect Task 6 fixed.
So chevrons are safe, but Task 6's `test_tilted_bar_extrudes_watertight` is the
guard that keeps them so.

- [ ] **Step 1: Write the failing test**

```python
class TestThermalActuators(unittest.TestCase):
    def _um(self, v):
        from soidlc.units import Quantity
        return Quantity(v * 1e-6, (1, 0, 0, 0))

    def test_chevron_beams_are_inclined_and_paired(self):
        from soidlc import primitives as P
        ctx = P.PrimitiveCtx()
        shapes = P.PRIMITIVES["chevron"](
            [], {"n": 3, "L": self._um(200), "w": self._um(6), "angle": 6.0},
            ctx)
        beams = [s for s in shapes if s.label == "chevron_beam"]
        self.assertEqual(len(beams), 6)          # n pairs
        shuttle = [s for s in shapes if s.label == "chevron_shuttle"]
        self.assertEqual(len(shuttle), 1)
        # an inclined beam is not axis-aligned
        from soidlc import mesh as M
        self.assertFalse(M._is_rectilinear(beams[0].polygon.exterior))

    def test_chevron_extrudes_watertight(self):
        from soidlc import primitives as P
        from soidlc import mesh as M
        ctx = P.PrimitiveCtx()
        shapes = P.PRIMITIVES["chevron"](
            [], {"n": 2, "L": self._um(200), "w": self._um(6), "angle": 6.0},
            ctx)
        for s in shapes:
            mesh = M.extrude_polygon(s.polygon, 0.0, 25.0, "DEVICE")
            edges = Counter()
            for (a, b, c) in mesh.triangles:
                for e in ((a, b), (b, c), (c, a)):
                    edges[frozenset(e)] += 1
            self.assertEqual(sum(1 for v in edges.values() if v != 2), 0,
                             f"{s.label} is not watertight")

    def test_hot_arm_is_asymmetric(self):
        from soidlc import primitives as P
        ctx = P.PrimitiveCtx()
        shapes = P.PRIMITIVES["hot_arm"](
            [], {"L": self._um(200), "w_hot": self._um(3),
                 "w_cold": self._um(12), "g": self._um(4)}, ctx)
        hot = [s for s in shapes if s.label == "hot_arm"][0]
        cold = [s for s in shapes if s.label == "cold_arm"][0]
        # the thin arm carries the higher current density, so it runs hotter
        self.assertLess(hot.polygon.area(), cold.polygon.area())
```

Run: `.venv/bin/python -m unittest tests.test_library.TestThermalActuators -v`
Expected: FAIL — `KeyError: 'chevron'`.

- [ ] **Step 2: Implement both primitives**

Add to `soidlc/primitives.py` (add `import math` at the top if absent):

```python
def prim_chevron(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """V-beam thermal actuator: n pairs of beams inclined by `angle` degrees
    from two anchors to a central shuttle.

    Passing current heats the beams; they expand, and because they are
    pre-inclined the expansion resolves into shuttle motion along +y rather
    than buckling in a random direction.  Force is large, stroke is small --
    the opposite trade to a comb drive.
    """
    n = int(round(_num(_arg(args, kwargs, 0, "n", 4))))
    L = _um(_arg(args, kwargs, 1, "L", Quantity(200e-6, (1, 0, 0, 0))))
    w = _um(_arg(args, kwargs, 2, "w", Quantity(6e-6, (1, 0, 0, 0))))
    angle = _num(_arg(args, kwargs, 3, "angle", 6.0))     # degrees

    a = math.radians(angle)
    dx = L * math.cos(a)
    dy = L * math.sin(a)
    shuttle_w = 20.0
    pitch = 3 * w + 10.0

    shapes: List[G.Shape] = []
    y = -(n - 1) * pitch / 2.0
    for _ in range(n):
        # left beam: anchor at (-dx, y) rising to the shuttle at (0, y + dy)
        shapes.append(G.Shape(
            ctx.device_layer,
            G.wire([(-dx, y), (0.0, y + dy)], w),
            "chevron_beam", mech="released"))
        # right beam: mirror image
        shapes.append(G.Shape(
            ctx.device_layer,
            G.wire([(dx, y), (0.0, y + dy)], w),
            "chevron_beam", mech="released"))
        y += pitch

    span = (n - 1) * pitch + 4 * w
    shapes.append(G.Shape(
        ctx.device_layer,
        G.rect(shuttle_w, span, 0.0, dy),
        "chevron_shuttle", mech="released"))
    # anchors at both ends of every beam row
    anc = (n - 1) * pitch + 4 * w
    shapes.append(G.Shape(ctx.device_layer,
                          G.rect(30.0, anc, -dx - 15.0 + w, 0.0),
                          "chevron_anchor", mech="anchored"))
    shapes.append(G.Shape(ctx.device_layer,
                          G.rect(30.0, anc, dx + 15.0 - w, 0.0),
                          "chevron_anchor", mech="anchored"))
    return shapes


def prim_hot_arm(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """U-shaped hot-arm / cold-arm thermal actuator.

    Both arms carry the same current, but the thin one has the higher
    resistance per unit length, so it gets hotter and expands more.  The pair
    is joined at the tip, so the difference bends the actuator toward the cold
    arm -- an arc, not a translation.
    """
    L = _um(_arg(args, kwargs, 0, "L", Quantity(200e-6, (1, 0, 0, 0))))
    w_hot = _um(_arg(args, kwargs, 1, "w_hot", Quantity(3e-6, (1, 0, 0, 0))))
    w_cold = _um(_arg(args, kwargs, 2, "w_cold", Quantity(12e-6, (1, 0, 0, 0))))
    g = _um(_arg(args, kwargs, 3, "g", Quantity(4e-6, (1, 0, 0, 0))))

    y_hot = (w_hot + g) / 2.0
    y_cold = -(w_cold + g) / 2.0
    shapes = [
        G.Shape(ctx.device_layer, G.rect(L, w_hot, 0.0, y_hot),
                "hot_arm", mech="released"),
        G.Shape(ctx.device_layer, G.rect(L, w_cold, 0.0, y_cold),
                "cold_arm", mech="released"),
        # tip yoke joining the two arms
        G.Shape(ctx.device_layer,
                G.rect(w_cold, abs(y_hot - y_cold) + w_cold,
                       L / 2.0 - w_cold / 2.0, (y_hot + y_cold) / 2.0),
                "hot_arm_yoke", mech="released"),
        # anchors at the driven end
        G.Shape(ctx.device_layer, G.rect(24.0, 24.0, -L / 2.0 - 8.0, y_hot),
                "hot_arm_anchor", mech="anchored"),
        G.Shape(ctx.device_layer, G.rect(24.0, 24.0, -L / 2.0 - 8.0, y_cold),
                "hot_arm_anchor", mech="anchored"),
    ]
    return shapes
```

Register both:

```python
    "chevron": prim_chevron,
    "hot_arm": prim_hot_arm,
```

Run the test. Expected: PASS, 3 tests. If `test_chevron_extrudes_watertight`
fails, `wire()`'s miter is self-intersecting at that angle — reduce the bevel
clip threshold in Task 4's `wire` from `0.25` upward and re-run both tasks'
tests.

- [ ] **Step 3: Full suite and commit**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`.

```bash
git add soidlc/primitives.py tests/test_library.py
git commit -m "feat(prim): chevron and hot-arm thermal actuators"
```

### Task 13: `pull_in` and `stroke_thermal` metrics

**Files:**
- Modify: `soidlc/reduce.py:59-116`, `soidlc/metrics.py:30+`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: shape labels from Tasks 11-12.
- Produces, callable from `require`/`solve` in any `.soidl` source:
  - `pull_in()` → collapse voltage [V] of the device's parallel-plate pair.
  - `stroke_thermal(dT)` → chevron shuttle displacement [m] at temperature rise `dT` [K].
  Task 14's examples use both.

**Deliberate deviation from the spec.** The spec wrote `stroke_thermal(P)` —
displacement at drive *power*. Converting power to temperature needs a thermal
resistance model (conduction down the beams into the anchors, plus convection
and conduction through the air gap to the substrate), none of which this
compiler has. Inventing one would produce a number that looks authoritative
and is not. The metric therefore takes the **temperature rise** directly,
which is the quantity the geometry actually determines the stroke from. A
power-to-temperature model is a separate change, noted in the spec's Out of
scope.

**Physics.**
- Pull-in: a parallel plate is stable only while the mechanical restoring force
  outruns the electrostatic one. It loses that race at `x = g/3`, giving
  `V_pi = sqrt(8 k g^3 / (27 eps0 A))`.
- Chevron: a beam from anchor `(-L cos a, 0)` to shuttle `(0, L sin a)` keeps
  its endpoints' separation equal to its length. Heating it to length `L(1+e)`
  with `e = alpha * dT` moves the shuttle to
  `y = L sqrt((1+e)^2 - cos^2 a)`, so the stroke is
  `d = L (sqrt((1+e)^2 - cos^2 a) - sin a)`. Silicon's `alpha = 2.6e-6 /K`.
  The exact form is used rather than the small-angle `d ≈ L e / sin a`, because
  shallow angles are exactly where the approximation degrades.

- [ ] **Step 1: Write the failing test**

```python
class TestActuatorMetrics(unittest.TestCase):
    PROC = TestImportResolution.PROC

    def test_pull_in_voltage_is_reported(self):
        src = self.PROC + """
          import "flexures.soidl";
          device d {
            inst M = plate(200 um, 100 um) at (0, 0);
            inst S = array(guided_beam(L = 250 um, w = 4 um), count = 4,
                           place = corners(M));
            inst PP = parallel_plate(W = 150 um, H = 30 um, g = 3 um, n = 2)
                      attach (rotor -> M.top);
            net DRIVE = PP.stator;
            net GND   = M | S.fixed;
            isolate DRIVE from GND by trench;
            require pull_in() >= 5 V;
            constraint anchored(S.fixed, PP.stator);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        ok = [l for l in art.report if l.startswith("require") and ": ok" in l]
        self.assertTrue(ok, art.report)

    def test_pull_in_scales_with_gap_to_the_three_halves(self):
        """V_pi ~ g^(3/2): doubling the gap must raise it by 2^1.5 = 2.83x.

        Exercises the compiler, not a formula retyped in the test -- the two
        devices differ only in `g`, so the ratio isolates the gap dependence.
        """
        from soidlc import metrics

        def vpi(gap_um):
            src = self.PROC + f"""
              import "flexures.soidl";
              device d {{
                inst M = plate(200 um, 100 um) at (0, 0);
                inst S = array(guided_beam(L = 250 um, w = 4 um), count = 4,
                               place = corners(M));
                inst PP = parallel_plate(W = 150 um, H = 30 um,
                                         g = {gap_um} um, n = 2)
                          attach (rotor -> M.top);
                net DRIVE = PP.stator;
                net GND   = M | S.fixed;
                isolate DRIVE from GND by trench;
                constraint anchored(S.fixed, PP.stator);
              }}
            """
            art = compile_source(src)
            self.assertEqual(art.errors, [], art.errors)
            env = metrics.build_env(art.elab, art.result)
            return env["pull_in"]().value

        self.assertAlmostEqual(vpi(6.0) / vpi(3.0), 2 ** 1.5, delta=0.05)

    def test_thermal_stroke_grows_with_temperature(self):
        from soidlc import metrics
        d1 = metrics.chevron_stroke(200e-6, 6.0, 50.0)
        d2 = metrics.chevron_stroke(200e-6, 6.0, 200.0)
        self.assertGreater(d2, d1)
        self.assertGreater(d1, 0.0)

    def test_shallower_chevron_gives_more_stroke(self):
        """A shallower V converts the same expansion into more motion."""
        from soidlc import metrics
        shallow = metrics.chevron_stroke(200e-6, 3.0, 100.0)
        steep = metrics.chevron_stroke(200e-6, 12.0, 100.0)
        self.assertGreater(shallow, steep)
```

Run: `.venv/bin/python -m unittest tests.test_library.TestActuatorMetrics -v`
Expected: FAIL — `module 'soidlc.metrics' has no attribute 'chevron_stroke'`,
and the `require pull_in()` device errors with an unknown name.

- [ ] **Step 2: Recognise the new transducers in `reduce.py`**

In `soidlc/reduce.py`, inside `find_transducers`'s `for owner, ss in ...` loop,
after the existing comb branch's `if not rf or not sf: continue`, add a
parallel-plate branch **before** that guard so plate-only owners are not
skipped:

```python
        rp = [s for s in ss if s.label == "rotor_plate"]
        sp = [s for s in ss if s.label == "stator_plate"]
        if rp and sp:
            t_pp = t_m
            rb = rp[0].polygon.bbox()
            sb = min(sp, key=lambda s: abs(_center(s.polygon.bbox())[1]
                                           - _center(rb)[1])).polygon.bbox()
            g_um = abs(_center(sb)[1] - _center(rb)[1]) \
                - (rb[3] - rb[1]) / 2.0 - (sb[3] - sb[1]) / 2.0
            if g_um <= 0:
                continue
            overlap_um = max(0.0, min(rb[2], sb[2]) - max(rb[0], sb[0]))
            n_pairs = len(rp)
            area_m2 = n_pairs * (overlap_um * 1e-6) * t_pp
            out.append(Transducer(
                name=owner, net=net_of.get(owner, owner),
                dcdx=EPS0 * area_m2 / ((g_um * 1e-6) ** 2),
                C0=EPS0 * area_m2 / (g_um * 1e-6),
                axis=1, sign=1.0, bbox=G.bbox_of(rp),
                N=n_pairs, g_um=g_um, ov_um=overlap_um))
            continue
```

Note `dcdx` for a plate is `eps0*A/g^2`, not the comb's `2*eps0*t/g` — the
plate's derivative depends on the gap squared, which is the nonlinearity.

- [ ] **Step 3: Add the metrics**

In `soidlc/metrics.py`, add module-level constants and a free function:

```python
EPS0 = 8.8541878128e-12
ALPHA_SI = 2.6e-6           # 1/K, linear thermal expansion of silicon


def chevron_stroke(L_m: float, angle_deg: float, dT: float) -> float:
    """Shuttle displacement [m] of a chevron beam of length L_m inclined by
    angle_deg, heated by dT kelvin.

    The beam's endpoints stay one beam-length apart, so heating it to
    L(1+e) with e = alpha*dT lifts the shuttle to y = L*sqrt((1+e)^2 - cos^2 a).
    Exact rather than the small-angle form, because shallow angles -- where a
    chevron is most useful -- are where the approximation is worst.
    """
    a = math.radians(angle_deg)
    e = ALPHA_SI * dT
    inner = (1.0 + e) ** 2 - math.cos(a) ** 2
    if inner <= 0.0:
        return 0.0
    return L_m * (math.sqrt(inner) - math.sin(a))
```

Then inside `build_env`, alongside the existing `stroke_max` / `arw`
definitions, add:

```python
    def pull_in(*_a, **_k) -> Quantity:
        """Collapse voltage of the device's parallel-plate transducer."""
        pp = next((t for t in trans if t.ov_um > 0 and t.axis == 1
                   and t.N and t.C0 > 0), None)
        if pp is None:
            raise MetricError("pull_in() needs a parallel_plate transducer")
        k = _model_q("k", "stiffness").value
        g = pp.g_um * 1e-6
        area = pp.C0 * g / EPS0            # recover A from C0 = eps0*A/g
        if area <= 0:
            raise MetricError("pull_in(): degenerate plate overlap")
        v = math.sqrt(8.0 * k * g ** 3 / (27.0 * EPS0 * area))
        return _q(v, (2, 1, -3, -1))       # volts

    def stroke_thermal(dT, *_a, **_k) -> Quantity:
        """Chevron shuttle stroke at a temperature rise dT [K]."""
        beams = [s for s in res.shapes if s.label == "chevron_beam"]
        if not beams:
            raise MetricError("stroke_thermal() needs a chevron actuator")
        bb = beams[0].polygon.bbox()
        L_m = math.hypot(bb[2] - bb[0], bb[3] - bb[1]) * 1e-6
        ang = math.degrees(math.atan2(abs(bb[3] - bb[1]),
                                      abs(bb[2] - bb[0]))) or 6.0
        dt = dT.as_float() if isinstance(dT, Quantity) else float(dT)
        return _q(chevron_stroke(L_m, ang, dt), (1, 0, 0, 0))
```

and register both in the returned `env` dict next to the existing entries:

```python
    env["pull_in"] = pull_in
    env["stroke_thermal"] = stroke_thermal
```

Check the exact registration idiom already used for `f_res` / `stroke_max` at
the end of `build_env` and follow it — if they are added via a dict literal,
add these to that literal instead.

The volts dimension `(2, 1, -3, -1)` is `kg·m²·s⁻³·A⁻¹`. Confirm against how
`soidlc/units.py` parses `V` (`grep -n '"V"' soidlc/units.py`) and use the
same tuple, so `require pull_in() >= 5 V` type-checks.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m unittest tests.test_library.TestActuatorMetrics -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Full suite and commit**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK` — in particular the existing comb-based transducer tests, which
the `reduce.py` change must not disturb.

```bash
git add soidlc/reduce.py soidlc/metrics.py tests/test_library.py
git commit -m "feat(metrics): pull-in voltage and chevron thermal stroke"
```

### Task 14: Thermal actuator, microgripper, RF switch

**Files:**
- Create: `examples/thermal_actuator.soidl`, `examples/microgripper.soidl`, `examples/rf_switch.soidl`

**Interfaces:**
- Consumes: Tasks 11-13.
- Produces: three examples covered by `test_examples_have_no_errors`.

- [ ] **Step 1: Write `examples/thermal_actuator.soidl`**

```soidl
// Chevron (V-beam) thermal actuator with a travel limit.
//
// Current through the inclined beams heats them; they expand, and the
// pre-inclination turns that expansion into shuttle motion instead of random
// buckling.  Large force, small stroke -- the opposite trade to a comb drive,
// which is why grippers and latches use chevrons and resonators do not.
// Compile:  python -m soidlc.cli examples/thermal_actuator.soidl

process soi25 {
  stack {
    layer METAL  { thickness = 1 um;   material = Au }
    layer DEVICE { thickness = 25 um;  material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;   material = SiO2 }
    layer HANDLE { thickness = 400 um; material = Si }
  }
  rules {
    release { hole_size = 5 um; hole_pitch = 25 um; max_solid_span = 30 um }
  }
}

device thermal_actuator {
  param dT_drive = 300 K;

  inst A  = chevron(n = 6, L = 250 um, w = 6 um, angle = 5) at (0, 0);
  // hard stop 4 um out: a chevron will happily crush whatever it reaches
  inst ST = gap_stop(4 um) at (0, 60 um);
  inst P1 = via_metal(100 um, 100 um) at (0 - 300 um, 0);
  inst P2 = via_metal(100 um, 100 um) at (300 um, 0);

  require stroke_thermal(dT_drive) >= 2 um;

  net HOT  = A.chevron_anchor;
  net GND  = ST;

  constraint anchored(ST);
}
```

- [ ] **Step 2: Compile it**

Run: `.venv/bin/python -m soidlc.cli examples/thermal_actuator.soidl -o /tmp/therm`
Expected: no `soidlc: ERROR:` lines. If the connectivity check reports the two
chevron anchors as separate islands both on net `HOT`, that is correct and
expected — a net split across disconnected islands is an **error** by design.
Fix it by giving the two anchors one net each (`net HOT_L`, `net HOT_R`), which
is also electrically honest: current flows in one anchor and out the other.

- [ ] **Step 3: Write `examples/microgripper.soidl`**

```soidl
// Electrothermal microgripper: two chevron-driven arms closing on a target.
//
// Each jaw is driven by its own chevron, so the jaws can be actuated together
// (grip) or differentially (rotate the held object).  The gap stop sets the
// minimum jaw opening so the actuator cannot crush what it is holding.
// Compile:  python -m soidlc.cli examples/microgripper.soidl

process soi25 {
  stack {
    layer METAL  { thickness = 1 um;   material = Au }
    layer DEVICE { thickness = 25 um;  material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;   material = SiO2 }
    layer HANDLE { thickness = 400 um; material = Si }
  }
  rules {
    release { hole_size = 5 um; hole_pitch = 25 um; max_solid_span = 30 um }
  }
}

component jaw(L = 400 um, w = 12 um) {
  mech port base, tip;
  geometry {
    beam(L, w, dir = y) at (0, L/2);
  }
}

device microgripper {
  param dT_drive = 250 K;

  inst AL = chevron(n = 4, L = 200 um, w = 6 um, angle = 5)
            at (0 - 160 um, 0);
  inst AR = chevron(n = 4, L = 200 um, w = 6 um, angle = 5)
            at (160 um, 0);
  inst JL = jaw(L = 400 um, w = 12 um) at (0 - 160 um, 30 um);
  inst JR = jaw(L = 400 um, w = 12 um) at (160 um, 30 um);

  require stroke_thermal(dT_drive) >= 1 um;

  net HOT_L = AL.chevron_anchor;
  net HOT_R = AR.chevron_anchor;
}
```

Compile it the same way. If the two `chevron_anchor` shapes inside one
instance land on separate islands, split the nets as in Step 2.

- [ ] **Step 4: Write `examples/rf_switch.soidl`**

```soidl
// RF MEMS shunt switch: a cantilever pulled down onto a gap-closing electrode.
//
// The switch is deliberately operated PAST pull-in -- the snap is the switching
// action, not a failure.  `require pull_in() <= 40 V` is therefore an upper
// bound on the drive supply, the opposite sense to a sensor, where pull-in is
// the thing to stay below.
// Compile:  python -m soidlc.cli examples/rf_switch.soidl

import "flexures.soidl";

process soi25 {
  stack {
    layer METAL  { thickness = 1 um;   material = Au }
    layer DEVICE { thickness = 25 um;  material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;   material = SiO2 }
    layer HANDLE { thickness = 400 um; material = Si }
  }
  rules {
    release { hole_size = 5 um; hole_pitch = 25 um; max_solid_span = 30 um }
  }
}

device rf_switch {
  inst BEAM = plate(300 um, 60 um) at (0, 0);
  inst SUS  = array(guided_beam(L = 180 um, w = 4 um, n_beams = 1),
                    count = 4, place = corners(BEAM));
  inst GAP  = parallel_plate(W = 200 um, H = 30 um, g = 3 um, n = 1)
              attach (rotor -> BEAM.top);
  inst RF1  = via_metal(80 um, 80 um) at (0 - 220 um, 0 - 90 um);
  inst RF2  = via_metal(80 um, 80 um) at (220 um, 0 - 90 um);

  require pull_in() <= 40 V;

  net ACT = GAP.stator;
  net GND = BEAM | SUS.fixed;

  isolate ACT from GND by trench;
  constraint released(BEAM, SUS);
  constraint anchored(SUS.fixed, GAP.stator);
}
```

- [ ] **Step 5: Compile all three and run the suite**

```bash
for e in thermal_actuator microgripper rf_switch; do
  .venv/bin/python -m soidlc.cli examples/$e.soidl -o /tmp/$e || echo "FAILED: $e"
done
.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q
```
Expected: three clean compiles and `OK`. `test_examples_have_no_errors` now
covers 8 examples.

- [ ] **Step 6: Commit**

```bash
git add examples/thermal_actuator.soidl examples/microgripper.soidl \
        examples/rf_switch.soidl
git commit -m "feat(examples): thermal actuator, microgripper, RF switch"
```

---

## Phase 3 — out-of-plane: the half of the process nobody could reach

### Task 15: Backside trench subtracts from HANDLE and BOX

**Files:**
- Modify: `soidlc/primitives.py` (`prim_trench`), `soidlc/build3d.py:20-42`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `trench(W, H)` emitting a `G.Shape(layer="TRENCH", label="trench", mech="anchored")`. `build3d.build_mesh` subtracts the union of TRENCH footprints from the HANDLE slab and skips BOX pillars inside them. Tasks 18-19 depend on it.

**Why this is the interesting change.** Every example process declares
`layer HANDLE` and `mask TRENCH -> etch(HANDLE, through, backside)`, and
nothing has ever used them: `prim_trench` returns `[]` (`primitives.py:159`)
and `build_mesh` lays the handle down as an unbroken slab
(`build3d.py:32`). Half the declared process has been decorative. A membrane
or a torsional mirror is not expressible until the substrate can have a hole
in it.

**Why it is safe.** TRENCH is not in `proc.layers`, so `build_mesh`'s
`if lay is None: continue` already skips extruding it as solid, and
`_check_connectivity` filters on the DEVICE layer, so trenches cannot create
or break electrical islands. Both the slab and the trenches are rectangles, so
the holed slab stays rectilinear and keeps the watertight voxel mesher.

- [ ] **Step 1: Write the failing test**

```python
class TestBacksideTrench(unittest.TestCase):
    PROC = TestImportResolution.PROC

    def _handle_volume(self, art):
        zs = [v[2] for v in art.mesh.vertices]
        return min(zs), max(zs)

    def test_trench_removes_handle_silicon(self):
        base = self.PROC + """
          device d {
            inst M = plate(400 um, 400 um) at (0, 0);
            inst A = anchor(40 um, 40 um) at (0, 0 - 220 um);
            net GND = M | A;
            constraint anchored(A);
          }
        """
        holed = self.PROC + """
          device d {
            inst M = plate(400 um, 400 um) at (0, 0);
            inst A = anchor(40 um, 40 um) at (0, 0 - 220 um);
            inst T = trench(300 um, 300 um) at (0, 0);
            net GND = M | A;
            constraint anchored(A);
          }
        """
        a = compile_source(base)
        b = compile_source(holed)
        self.assertEqual(a.errors, [], a.errors)
        self.assertEqual(b.errors, [], b.errors)
        # a hole in the slab means MORE triangles, not fewer
        self.assertGreater(len(b.mesh.triangles), len(a.mesh.triangles))

    def test_trenched_mesh_is_still_watertight(self):
        src = self.PROC + """
          device d {
            inst M = plate(400 um, 400 um) at (0, 0);
            inst A = anchor(40 um, 40 um) at (0, 0 - 220 um);
            inst T = trench(300 um, 300 um) at (0, 0);
            net GND = M | A;
            constraint anchored(A);
          }
        """
        art = compile_source(src)
        edges = Counter()
        for (a, b, c) in art.mesh.triangles:
            for e in ((a, b), (b, c), (c, a)):
                edges[frozenset(e)] += 1
        self.assertEqual(sum(1 for v in edges.values() if v != 2), 0)

    def test_trench_creates_no_electrical_island(self):
        """TRENCH is not DEVICE silicon, so it must not appear as a net."""
        src = self.PROC + """
          device d {
            inst M = plate(400 um, 400 um) at (0, 0);
            inst A = anchor(40 um, 40 um) at (0, 0 - 220 um);
            inst T = trench(300 um, 300 um) at (0, 0);
            net GND = M | A;
            constraint anchored(A);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
```

Run: `.venv/bin/python -m unittest tests.test_library.TestBacksideTrench -v`
Expected: `test_trench_removes_handle_silicon` FAILS (equal triangle counts —
the trench is a no-op today). The other two pass already; they are the
regression guards for this change.

- [ ] **Step 2: Give `trench` geometry**

In `soidlc/primitives.py`, replace `prim_trench`:

```python
def prim_trench(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A backside etch opening: an *absence* of substrate, not silicon.

    Emitted on the pseudo-layer TRENCH, which is deliberately not part of the
    process stack -- build3d subtracts these footprints from the HANDLE slab
    and from the BOX beneath them instead of extruding them as solid, and the
    connectivity extractor ignores them because they are not DEVICE.
    """
    W = _um(_arg(args, kwargs, 0, "W", Quantity(200e-6, (1, 0, 0, 0))))
    H = _um(_arg(args, kwargs, 1, "H", Quantity(200e-6, (1, 0, 0, 0))))
    return [G.Shape("TRENCH", G.rect(W, H), "trench", mech="anchored")]
```

- [ ] **Step 3: Subtract trenches in `build3d`**

Replace `build_mesh` in `soidlc/build3d.py`:

```python
def build_mesh(result: InstanceResult, proc: ProcessInfo,
               include_handle: bool = True) -> M.Mesh:
    out = M.Mesh()
    box = proc.box()
    handle = proc.handle()

    trenches = [sh.polygon for sh in result.shapes if sh.layer == "TRENCH"]

    # handle slab spanning the footprint (a little margin), minus the
    # backside etch openings
    if include_handle and handle is not None and result.shapes:
        x0, y0, x1, y1 = result.bbox
        pad = 20.0
        slab = G.rect_corner(x0 - pad, y0 - pad,
                             (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad)
        if trenches:
            slab = G.Polygon(slab.exterior,
                             [list(t.exterior) for t in trenches])
        out.extend(M.extrude_polygon(slab, handle.z0, handle.z1, "HANDLE"))

    for sh in result.shapes:
        lay = proc.layers.get(sh.layer)
        if lay is None:
            continue                    # TRENCH and any other pseudo-layer
        out.extend(M.extrude_polygon(sh.polygon, lay.z0, lay.z1, sh.layer))
        # anchored device silicon keeps the oxide beneath it -- unless the
        # backside etch removed the substrate under it, in which case there is
        # nothing for the pillar to stand on
        if (sh.layer == "DEVICE" and sh.mech == "anchored" and box is not None
                and not _inside_any(sh.polygon, trenches)):
            solid = G.Polygon(sh.polygon.exterior)   # drop holes for the pillar
            out.extend(M.extrude_polygon(solid, box.z0, box.z1, "BOX"))

    return out


def _inside_any(poly: G.Polygon, trenches) -> bool:
    """True if `poly`'s bbox lies wholly within some trench footprint."""
    px0, py0, px1, py1 = poly.bbox()
    for t in trenches:
        tx0, ty0, tx1, ty1 = t.bbox()
        if px0 >= tx0 and py0 >= ty0 and px1 <= tx1 and py1 <= ty1:
            return True
    return False
```

`_inside_any` uses a wholly-contained bbox test, not an overlap test,
deliberately: a shape straddling a trench edge still has substrate under part
of it, so it keeps its pillar. Partial-overlap pillar clipping is not modelled
and does not need to be — the rim of a membrane is exactly the straddling case,
and it must stay anchored.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m unittest tests.test_library.TestBacksideTrench -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Regression check on every existing example**

Run: `.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q`
Expected: `OK`. No existing example places a `trench`, so `trenches` is empty
and the slab code path is byte-identical to before — that is what makes this
change safe to land.

- [ ] **Step 6: Commit**

```bash
git add soidlc/primitives.py soidlc/build3d.py tests/test_library.py
git commit -m "feat(build3d): backside trench cuts the handle slab and its oxide"
```

### Task 16: `pad` and `route` on METAL

**Files:**
- Modify: `soidlc/primitives.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `geometry.wire` (Task 4).
- Produces: primitives `pad(w, h)` and `route(points, w)` on the METAL layer. `via_metal` stays as-is so no existing example changes.

- [ ] **Step 1: Write the failing test**

```python
class TestMetalRouting(unittest.TestCase):
    def test_pad_and_route_land_on_metal(self):
        from soidlc import primitives as P
        from soidlc.units import Quantity

        def um(v):
            return Quantity(v * 1e-6, (1, 0, 0, 0))

        ctx = P.PrimitiveCtx()
        pad = P.PRIMITIVES["pad"]([], {"w": um(100), "h": um(100)}, ctx)
        self.assertEqual([s.layer for s in pad], [ctx.metal_layer])
        rt = P.PRIMITIVES["route"](
            [], {"points": [(0.0, 0.0), (100.0, 0.0), (100.0, 80.0)],
                 "w": um(8)}, ctx)
        self.assertEqual([s.layer for s in rt], [ctx.metal_layer])
        self.assertGreater(rt[0].polygon.area(), 0.0)
```

Run it. Expected: FAIL — `KeyError: 'pad'`.

- [ ] **Step 2: Implement**

```python
def prim_pad(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A bond pad on METAL."""
    w = _um(_arg(args, kwargs, 0, "w", Quantity(100e-6, (1, 0, 0, 0))))
    h = _um(_arg(args, kwargs, 1, "h", Quantity(100e-6, (1, 0, 0, 0))))
    return [G.Shape(ctx.metal_layer, G.rect(w, h), "pad", mech="anchored")]


def prim_route(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A metal interconnect track following a polyline."""
    pts = _arg(args, kwargs, 0, "points")
    w = _um(_arg(args, kwargs, 1, "w", Quantity(8e-6, (1, 0, 0, 0))))
    if not pts:
        raise ValueError("route() needs a points list")
    coords = [(_um(p[0]) if isinstance(p[0], Quantity) else float(p[0]),
               _um(p[1]) if isinstance(p[1], Quantity) else float(p[1]))
              for p in pts]
    return [G.Shape(ctx.metal_layer, G.wire(coords, w), "route",
                    mech="anchored")]
```

Register `"pad": prim_pad, "route": prim_route`.

Whether SOIDL's parser accepts a bracketed list literal for `points` needs
checking: `grep -n '"\["' soidlc/parser.py`. If list literals are not
supported, `route` is callable from Python but not from `.soidl` source — in
that case leave `route` registered (Task 22 does not require it from source)
and note the limitation in the commit message rather than extending the
grammar here.

- [ ] **Step 3: Run, full suite, commit**

```bash
.venv/bin/python -m unittest tests.test_library.TestMetalRouting -v
.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q
git add soidlc/primitives.py tests/test_library.py
git commit -m "feat(prim): pad and route metal interconnect"
```

### Task 17: Torsional lumped model

**Files:**
- Modify: `soidlc/lib/flexures.soidl` (add `torsion_bar`), `soidlc/elaborate.py:603-625`, `soidlc/metrics.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `import` (Task 3).
- Produces:
  - `torsion_bar(L, w)` with `derive k_theta = G*J/L`.
  - `_extract_device_model` additionally reports `J_m` (mass moment of inertia, kg·m²) and `f0_theta` when instances contribute `k_theta`.
  - `f_res(..., axis = "theta")` returns the torsional frequency; a device with `k_theta` and no `k_x` raises `MetricError` from a bare `f_res()`.
  Task 18 depends on all of it.

**The dangerous failure this prevents.** `_extract_device_model`
(`elaborate.py:603`) computes `m` from released area × thickness × ρ and
`f0 = sqrt(k/m)/2π`. That is purely translational. Feed a torsional mirror into
it and it does not raise — it returns a plausible, meaningless number. Silent
wrongness is worse than an error, so the torsional path is explicit and the
mismatched case is made to raise.

**Physics.** For a rectangular bar of thickness `t` and width `w` with
`a = max(t,w)/2`, `b = min(t,w)/2`, the torsion constant is
`J = a*b^3*(16/3 - 3.36*(b/a)*(1 - b^4/(12*a^4)))`, and
`k_theta = G*J/L` with `G = E/(2*(1+nu))`. The mass moment of inertia about
the torsion axis is `J_m = Σ ρ*t*dA*r²`, which for a plate of width `W` about
its own centreline is `ρ*t*W³*H/12`.

- [ ] **Step 1: Write the failing test**

```python
class TestTorsion(unittest.TestCase):
    PROC = TestImportResolution.PROC

    SRC = TestImportResolution.PROC + """
      import "flexures.soidl";
      device mirror {
        inst PL = plate(400 um, 200 um, holes = none) at (0, 0);
        inst TL = torsion_bar(L = 120 um, w = 6 um) at (0 - 260 um, 0);
        inst TR = torsion_bar(L = 120 um, w = 6 um) at (260 um, 0);
        net GND = PL | TL.fixed | TR.fixed;
        constraint anchored(TL.fixed, TR.fixed);
      }
    """

    def test_torsional_model_is_reported(self):
        art = compile_source(self.SRC)
        self.assertEqual(art.errors, [], art.errors)
        self.assertIn("k_theta", art.model)
        self.assertIn("J_m", art.model)
        self.assertIn("f0_theta", art.model)
        self.assertGreater(art.model["f0_theta"].value, 1e3)

    def test_bare_f_res_refuses_a_purely_torsional_device(self):
        """The silent-wrong-number case: it must raise, not guess."""
        src = self.SRC.replace(
            "constraint anchored(TL.fixed, TR.fixed);",
            "require f_res() >= 1 kHz;\n        "
            "constraint anchored(TL.fixed, TR.fixed);")
        art = compile_source(src)
        self.assertTrue(
            any("torsional" in str(e).lower() or "f_res" in str(e)
                for e in art.errors) or
            any("metrics unavailable" in w or "torsional" in w.lower()
                for w in art.warnings),
            f"errors={art.errors} warnings={art.warnings}")

    def test_shorter_bar_is_stiffer(self):
        long = compile_source(self.SRC)
        short = compile_source(self.SRC.replace("L = 120 um", "L = 60 um"))
        self.assertGreater(short.model["f0_theta"].value,
                           long.model["f0_theta"].value)
```

Run it. Expected: FAIL — `torsion_bar` is not defined.

- [ ] **Step 2: Add `torsion_bar` to the library**

Append to `soidlc/lib/flexures.soidl`:

```soidl
// Torsion bar: a beam loaded in twist rather than bending, the hinge of a
// scanning mirror or a teeter-totter z-accelerometer.
//
// For a rectangular section with a = max(t,w)/2 and b = min(t,w)/2,
//   J = a*b^3 * (16/3 - 3.36*(b/a)*(1 - b^4/(12*a^4)))
//   k_theta = G*J/L,   G = E / (2*(1+nu))
// Two bars on opposite sides of a mirror act in parallel.
component torsion_bar(L = 120 um, w = 6 um) {
  mech port fixed, hinge;

  derive G_si = process.DEVICE.E / (2 * (1 + process.DEVICE.nu));
  derive a_ts = max(process.DEVICE.thickness, w) / 2;
  derive b_ts = min(process.DEVICE.thickness, w) / 2;
  derive J_ts = a_ts * b_ts^3
                * (16/3 - 3.36 * (b_ts/a_ts) * (1 - b_ts^4 / (12 * a_ts^4)));
  derive k_theta = G_si * J_ts / L;

  geometry {
    beam(L, w, dir = x) at (2 um - L/2, 0);
    anchor(24 um, 24 um) at (2 um - L - 10 um, 0);
  }

  check w >= 2 um;
}
```

If the expression evaluator lacks `max`/`min`, add them to `_eval_call`'s
builtin table in `soidlc/elaborate.py:268` alongside whatever functions are
already there — they are pure scalar helpers and belong in the same place.

- [ ] **Step 3: Extend the lumped model**

In `soidlc/elaborate.py`, replace `_extract_device_model`:

```python
    def _extract_device_model(self, insts) -> Dict[str, object]:
        dev = self.process.device()
        t = (dev.thickness * 1e-6) if dev else 25e-6
        rho = 2330.0
        m = 0.0
        k = 0.0
        k_theta = 0.0
        j_m = 0.0
        for name, ir in insts.items():
            for sh in ir.shapes:
                if sh.layer == "DEVICE" and sh.mech == "released":
                    a_m2 = sh.polygon.area() * 1e-12
                    m += a_m2 * t * rho
                    # inertia about the global y = 0 torsion axis:
                    # a rectangle of width W about its own centre is W^2/12,
                    # shifted to the axis by the parallel-axis theorem
                    x0, y0, x1, y1 = sh.polygon.bbox()
                    w_um = x1 - x0
                    cx = (x0 + x1) / 2.0
                    r2 = ((w_um * 1e-6) ** 2) / 12.0 + (cx * 1e-6) ** 2
                    j_m += a_m2 * t * rho * r2
            kx = ir.model.get("k_x")
            if isinstance(kx, Quantity):
                k += kx.value
            kt = ir.model.get("k_theta")
            if isinstance(kt, Quantity):
                k_theta += kt.value
        model: Dict[str, object] = {}
        if m > 0:
            model["m"] = Quantity(m, (0, 1, 0, 0))
        if k > 0:
            model["k"] = Quantity(k, (0, 1, -2, 0))
        if m > 0 and k > 0:
            model["f0"] = Quantity(math.sqrt(k / m) / (2 * math.pi),
                                   (0, 0, -1, 0))
        if k_theta > 0:
            model["k_theta"] = Quantity(k_theta, (2, 1, -2, 0))
            if j_m > 0:
                model["J_m"] = Quantity(j_m, (2, 1, 0, 0))
                model["f0_theta"] = Quantity(
                    math.sqrt(k_theta / j_m) / (2 * math.pi), (0, 0, -1, 0))
        return model
```

Confirm the elaborator records a `derive k_theta = ...` under the model key
`"k_theta"` the same way it records `derive k.x` as `"k_x"` — check
`_record_derive` (`elaborate.py:744`) and `_extract_component_model`
(`elaborate.py:575`) and follow whatever naming they produce.

- [ ] **Step 4: Make `f_res` refuse to guess**

In `soidlc/metrics.py`'s `build_env`, replace `f_res`:

```python
    def f_res(*_a, **_k) -> Quantity:
        axis = _k.get("axis")
        if axis in ("theta", "torsion"):
            v = model.get("f0_theta")
            if not isinstance(v, Quantity):
                raise MetricError("f_res(axis=theta): no torsional mode "
                                  "(device has no k_theta)")
            return v
        v = model.get("f0")
        if not isinstance(v, Quantity):
            if isinstance(model.get("f0_theta"), Quantity):
                raise MetricError(
                    "this device has only a torsional mode; "
                    "call f_res(axis = theta) -- a translational f_res would "
                    "return a meaningless number")
            raise MetricError("resonant frequency not available "
                              "(no extractable f0 in this device)")
        return v                  # <-- see the note below before writing this
```

**Do not copy that last line blindly.** The current `f_res`
(`soidlc/metrics.py:67`) applies the closure stage's FEM calibration factor
from `cal` (`metrics.py:39`) before returning, and dropping it would silently
break `--fem-closure` and `test_fem_in_the_loop_calibration`. Read the existing
body first and keep its return expression verbatim on the translational path;
the only change is the two new `raise MetricError` branches above it. The
torsional path returns `f0_theta` uncalibrated, because no closure calibration
is computed for torsional modes.

- [ ] **Step 5: Run the tests, full suite, commit**

```bash
.venv/bin/python -m unittest tests.test_library.TestTorsion -v
.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q
git add soidlc/lib/flexures.soidl soidlc/elaborate.py soidlc/metrics.py \
        tests/test_library.py
git commit -m "feat(model): torsional stiffness and inertia; f_res refuses to guess"
```

### Task 18: Micromirror and teeter-totter, validated against the 3D solver

**Files:**
- Create: `examples/micromirror.soidl`, `examples/teeter_totter_accel.soidl`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: Tasks 15 and 17.
- Produces: the first two devices whose primary mode is out-of-plane, and the plan's only 3D-FEM validation.

**Why the 3D solver is the right arbiter.** The 2D plane-stress solver cannot
see a torsional mode at all — it has no out-of-plane degree of freedom. The
repo's own record says so: on the comb island, 3D found modes at 21.4 / 116.1 /
185.1 / 389.4 / 453.5 kHz where 2D saw only 21.5 / 447.8, three out-of-plane
modes invisible to 2D (`docs/superpowers/specs/2026-06-11-solid3d-fem-design.md`).
So `soidlc/fem/solid3d.py` is not a nicety here; it is the only thing that can
check `k_theta`.

- [ ] **Step 1: Write `examples/micromirror.soidl`**

```soidl
// Electrostatic torsional micromirror.
//
// A mirror plate hung on two collinear torsion bars.  Driving one side pulls
// that edge down and the plate rotates about the bar axis -- the mode is
// torsional, so its frequency comes from k_theta/J_m, not from k/m.  A plain
// f_res() would return a meaningless translational number here, which is why
// this device asks for f_res(axis = theta).
//
// The plate carries NO release holes: it is a mirror, and holes would scatter
// the beam.  It stays releasable because the backside trench removes the
// substrate under it instead.
// Compile:  python -m soidlc.cli examples/micromirror.soidl

import "flexures.soidl";

process soi25 {
  stack {
    layer METAL  { thickness = 0.52 um; material = Au }
    layer DEVICE { thickness = 25 um;   material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;    material = SiO2 }
    layer HANDLE { thickness = 400 um;  material = Si }
  }
  masks {
    mask SOID   -> etch(DEVICE, through)
    mask TRENCH -> etch(HANDLE, through, backside)
  }
  rules {
    release { hole_size = 5 um; hole_pitch = 25 um; max_solid_span = 30 um }
  }
}

device micromirror {
  // optical surface: no perforation
  inst MIR = plate(500 um, 400 um, holes = none) at (0, 0);
  inst TL  = torsion_bar(L = 140 um, w = 6 um) at (0 - 320 um, 0);
  inst TR  = torsion_bar(L = 140 um, w = 6 um) at (320 um, 0);
  // backside cavity so the unperforated mirror still releases, and so it has
  // clearance to rotate into
  inst CAV = trench(560 um, 460 um) at (0, 0);

  require f_res(axis = theta) >= 2 kHz;

  net GND = MIR | TL.fixed | TR.fixed;
  constraint anchored(TL.fixed, TR.fixed);
}
```

- [ ] **Step 2: Compile it**

Run: `.venv/bin/python -m soidlc.cli examples/micromirror.soidl -o /tmp/mirror`
Expected: no `soidlc: ERROR:` lines, and a report containing `f0_theta`.

`plate(..., holes = none)` is verified to work: `prim_plate` (`primitives.py:70`)
only generates holes when the argument is `"auto"` or `True`, and a
400x400 plate compiles to 144 holes with `holes = auto` and 0 with
`holes = none`, no errors either way. One thing does need checking:

- If the mirror is reported as an unanchored floating island, the torsion bars
  are not reaching it: `torsion_bar`'s beam runs from `2 um - L/2` centred at
  the instance origin, so the instance must sit within 2 µm of the mirror edge.
  Adjust the placement, not the component.

- [ ] **Step 3: Write `examples/teeter_totter_accel.soidl`**

```soidl
// Teeter-totter (see-saw) z-axis accelerometer.
//
// The proof mass is deliberately ASYMMETRIC about the torsion axis: more
// silicon on one side than the other.  Vertical acceleration therefore applies
// a net torque, and the plate rocks -- which is how an in-plane process senses
// the out-of-plane axis at all.  Two sense electrodes under the two halves
// read the rotation differentially, rejecting common-mode translation.
// Compile:  python -m soidlc.cli examples/teeter_totter_accel.soidl --fem

import "flexures.soidl";

process soi25 {
  stack {
    layer METAL  { thickness = 0.52 um; material = Au }
    layer DEVICE { thickness = 25 um;   material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;    material = SiO2 }
    layer HANDLE { thickness = 400 um;  material = Si }
  }
  masks {
    mask SOID   -> etch(DEVICE, through)
    mask TRENCH -> etch(HANDLE, through, backside)
  }
  rules {
    release { hole_size = 6 um; hole_pitch = 30 um; max_solid_span = 40 um }
  }
}

device teeter_totter_accel {
  // asymmetric mass: the long arm reaches 300 um one way, 150 um the other,
  // so the centre of mass sits off the hinge and gravity makes a torque
  inst LONG  = plate(300 um, 260 um) at (150 um, 0);
  inst SHORT = plate(150 um, 260 um) at (0 - 75 um, 0);
  inst TT    = torsion_bar(L = 100 um, w = 5 um) at (0, 140 um);
  inst TB    = torsion_bar(L = 100 um, w = 5 um) at (0, 0 - 140 um);
  inst CAV   = trench(560 um, 400 um) at (40 um, 0);

  require f_res(axis = theta) >= 1 kHz;

  net GND = LONG | SHORT | TT.fixed | TB.fixed;
  constraint anchored(TT.fixed, TB.fixed);
}
```

Compile it the same way. The two plates abut along `x = 0`, so connectivity
fuses them into one island — that is intended; they are one proof mass.

- [ ] **Step 4: Run the 3D solid solver by hand and read the mode shape**

```bash
.venv/bin/python - <<'PY'
import os
from soidlc import compile_source
from soidlc.fem import solid3d
EX = "examples"
art = compile_source(open(os.path.join(EX, "teeter_totter_accel.soidl")).read(),
                     base_dir=EX)
dev = art.process.device()
shapes = [s for s in art.result.shapes if s.layer == "DEVICE"]
m3 = solid3d.mesh_island_3d(shapes, dev.thickness, h=12.0)
f, vecs, dof_of = solid3d.modal3d(m3, dev.E * 1e9 if dev.E < 1e6 else dev.E,
                                  0.22, 2330.0, n_modes=4)
print("3D modes [Hz]:", [round(x) for x in f])
print("lumped f0_theta:", art.model["f0_theta"].value)
PY
```

Read `dev.E`'s units from `ProcessInfo` before running — if `E` is already
stored in pascals, drop the conditional.

Expected: mode 1 is torsional and within 15% of `f0_theta`. To confirm it is
torsional rather than a translational bounce, check that the mode's `uz` is
**antisymmetric** about the hinge: nodes at `x > 0` move opposite in z to
nodes at `x < 0`.

- [ ] **Step 5: Add the 3D regression test**

```python
class TestTeeterTotter3D(unittest.TestCase):
    """The 2D plane-stress solver cannot see this mode at all -- only 3D can."""

    def test_first_mode_is_torsional_and_matches_the_lumped_model(self):
        from soidlc.fem import solid3d
        with open(os.path.join(EX, "teeter_totter_accel.soidl")) as f:
            art = compile_source(f.read(), base_dir=EX)
        self.assertEqual(art.errors, [], art.errors)
        dev = art.process.device()
        shapes = [s for s in art.result.shapes if s.layer == "DEVICE"]
        m3 = solid3d.mesh_island_3d(shapes, dev.thickness, h=12.0)
        freqs, vecs, dof_of = solid3d.modal3d(
            m3, dev.E, dev.nu, 2330.0, n_modes=3)
        f_lumped = art.model["f0_theta"].value
        rel = abs(freqs[0] - f_lumped) / f_lumped
        self.assertLess(rel, 0.15,
                        f"3D mode1 {freqs[0]:.0f} Hz vs lumped "
                        f"{f_lumped:.0f} Hz ({rel*100:.1f}%)")

        # torsional: uz must flip sign across the hinge at x = 0
        left, right = [], []
        for i, node in enumerate(m3.nodes):
            b = dof_of.get(i)
            if b is None:
                continue
            uz = vecs[0][b + 2]
            (left if node[0] < 0 else right).append(uz)
        self.assertLess(sum(left) * sum(right), 0.0,
                        "mode 1 is not antisymmetric about the hinge")
```

Adapt the attribute names (`m3.nodes`, `dof_of`, the vector layout) to
`solid3d.modal3d`'s documented return contract — read its docstring first; the
2D solver returns `(freqs, vecs, dof_of)` with `vec[b], vec[b+1]` per vertex,
and the 3D one adds `vec[b+2]`.

- [ ] **Step 6: Run everything and commit**

```bash
.venv/bin/python -m unittest tests.test_library.TestTeeterTotter3D -v
.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q
git add examples/micromirror.soidl examples/teeter_totter_accel.soidl \
        tests/test_library.py
git commit -m "feat(examples): torsional micromirror and teeter-totter, 3D-FEM validated"
```

### Task 19: `membrane` and a pressure sensor

**Files:**
- Create: `soidlc/lib/membranes.soidl`, `examples/pressure_sensor.soidl`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: Task 15 (`trench`).
- Produces: `membrane(W, H, margin)` in `soidlc/lib/membranes.soidl`.

- [ ] **Step 1: Write `soidlc/lib/membranes.soidl`**

```soidl
// SOIDL standard library: membranes.
// Imported as:  import "membranes.soidl";

// A pressure-sensing diaphragm: an UNPERFORATED plate whose substrate is
// removed from behind by a backside etch.  Every other released structure in
// this library floats because the sacrificial oxide under it was dissolved
// through release holes; a membrane must stay solid to hold a pressure
// difference, so it is freed from below instead.
//
// `margin` is the width of the clamped rim: the trench is inset by that much
// on every side, and the rim is what the diaphragm bends against.
component membrane(W = 400 um, H = 400 um, margin = 40 um) {
  mech port rim, centre;

  geometry {
    plate(W, H, holes = none) at (0, 0);
    trench(W - 2*margin, H - 2*margin) at (0, 0);
  }

  check margin >= 10 um;
  check W > 4*margin;
  check H > 4*margin;
}
```

- [ ] **Step 2: Write the failing test**

```python
class TestMembrane(unittest.TestCase):
    def test_membrane_is_unperforated_and_backed_by_a_trench(self):
        src = TestImportResolution.PROC + """
          import "membranes.soidl";
          device d {
            inst MB = membrane(W = 400 um, H = 400 um, margin = 40 um)
                      at (0, 0);
            net GND = MB;
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        dev = [s for s in art.result.shapes if s.layer == "DEVICE"]
        self.assertTrue(dev)
        self.assertEqual(sum(len(s.polygon.holes) for s in dev), 0,
                         "a diaphragm must not be perforated")
        tr = [s for s in art.result.shapes if s.layer == "TRENCH"]
        self.assertEqual(len(tr), 1)
        x0, y0, x1, y1 = tr[0].polygon.bbox()
        self.assertAlmostEqual(x1 - x0, 320.0, delta=1.0)
```

Run it; expected FAIL (`membranes.soidl` not found), then PASS after Step 1.

- [ ] **Step 3: Write `examples/pressure_sensor.soidl`**

```soidl
// Piezoresistive membrane pressure sensor.
//
// The first example in this repo that uses the backside etch the process has
// always declared: `mask TRENCH -> etch(HANDLE, through, backside)`.  Pressure
// deflects the unperforated diaphragm; the strain is largest at the rim, so
// that is where sense elements belong.
// Compile:  python -m soidlc.cli examples/pressure_sensor.soidl

import "membranes.soidl";

process soi25 {
  stack {
    layer METAL  { thickness = 1 um;   material = Au }
    layer DEVICE { thickness = 25 um;  material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;   material = SiO2 }
    layer HANDLE { thickness = 400 um; material = Si }
  }
  masks {
    mask SOID     -> etch(DEVICE, through)
    mask TRENCH   -> etch(HANDLE, through, backside)
    mask PADMETAL -> deposit(METAL)
  }
  rules {
    release { hole_size = 6 um; hole_pitch = 30 um; max_solid_span = 40 um }
  }
}

device pressure_sensor {
  inst DIA = membrane(W = 600 um, H = 600 um, margin = 60 um) at (0, 0);
  inst P1  = via_metal(100 um, 100 um) at (0 - 380 um, 0 - 380 um);
  inst P2  = via_metal(100 um, 100 um) at (380 um, 0 - 380 um);

  net GND = DIA;
}
```

- [ ] **Step 4: Compile and inspect the 3D result**

```bash
.venv/bin/python -m soidlc.cli examples/pressure_sensor.soidl -o /tmp/press
```
Expected: no errors. Open `/tmp/press.png` and confirm the handle slab has a
square hole under the diaphragm. This is the visual proof that Task 15 landed:
before it, the substrate was solid everywhere.

- [ ] **Step 5: Full suite and commit**

```bash
.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q
git add soidlc/lib/membranes.soidl examples/pressure_sensor.soidl \
        tests/test_library.py
git commit -m "feat(lib): membrane element and a backside-etched pressure sensor"
```

---

## Phase 4 — circular geometry

### Task 20: `ring` and `disk` primitives

**Files:**
- Modify: `soidlc/primitives.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `geometry.circle`/`annulus` (Task 5) and the band mesher (Task 6).
- Produces: primitives `ring(R, w, n_seg)` and `disk(R, n_seg)`, both on DEVICE, released.

- [ ] **Step 1: Write the failing test**

```python
class TestRingDisk(unittest.TestCase):
    def _um(self, v):
        from soidlc.units import Quantity
        return Quantity(v * 1e-6, (1, 0, 0, 0))

    def test_ring_area_matches_the_annulus(self):
        from soidlc import primitives as P
        ctx = P.PrimitiveCtx()
        shapes = P.PRIMITIVES["ring"](
            [], {"R": self._um(200), "w": self._um(20), "n_seg": 128}, ctx)
        self.assertEqual(len(shapes), 1)
        expected = math.pi * (210.0 ** 2 - 190.0 ** 2)
        self.assertAlmostEqual(shapes[0].polygon.area(), expected,
                               delta=expected * 0.002)
        self.assertTrue(shapes[0].polygon.band)

    def test_ring_extrudes_watertight_through_the_compiler(self):
        src = TestImportResolution.PROC + """
          device d {
            inst R = ring(R = 200 um, w = 20 um) at (0, 0);
            inst A = anchor(30 um, 30 um) at (0, 195 um);
            net GND = R | A;
            constraint anchored(A);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
        edges = Counter()
        for (a, b, c) in art.mesh.triangles:
            for e in ((a, b), (b, c), (c, a)):
                edges[frozenset(e)] += 1
        self.assertEqual(sum(1 for v in edges.values() if v != 2), 0)

    def test_disk_gets_release_holes_when_it_is_wide(self):
        from soidlc import primitives as P
        ctx = P.PrimitiveCtx(max_solid_span=30.0, hole_pitch=25.0,
                             hole_size=5.0)
        big = P.PRIMITIVES["disk"]([], {"R": self._um(150)}, ctx)[0]
        self.assertGreater(len(big.polygon.holes), 0)
        small = P.PRIMITIVES["disk"]([], {"R": self._um(10)}, ctx)[0]
        self.assertEqual(len(small.polygon.holes), 0)
```

Run it. Expected: FAIL — `KeyError: 'ring'`.

- [ ] **Step 2: Implement both**

```python
def prim_ring(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """An annular resonator: the body of a ring gyroscope.

    A ring's two degenerate wine-glass modes sit 45 degrees apart, and Coriolis
    coupling transfers energy between them -- the rate signal.  That degeneracy
    is why the ring is circular rather than square.
    """
    R = _um(_arg(args, kwargs, 0, "R", Quantity(200e-6, (1, 0, 0, 0))))
    w = _um(_arg(args, kwargs, 1, "w", Quantity(20e-6, (1, 0, 0, 0))))
    n_seg = int(round(_num(_arg(args, kwargs, 2, "n_seg", 64))))
    return [G.Shape(ctx.device_layer, G.annulus(R, w, n_seg), "ring",
                    mech="released")]


def prim_disk(args, kwargs, ctx: PrimitiveCtx) -> List[G.Shape]:
    """A solid disk, perforated from the process release rules like a plate."""
    R = _um(_arg(args, kwargs, 0, "R", Quantity(150e-6, (1, 0, 0, 0))))
    n_seg = int(round(_num(_arg(args, kwargs, 1, "n_seg", 64))))
    holes_arg = _arg(args, kwargs, 2, "holes", "auto")
    poly = G.circle(R, n_seg)
    want_holes = (holes_arg == "auto" or holes_arg is True)
    if want_holes and 2 * R > ctx.max_solid_span:
        keep = []
        for h in _hole_grid(2 * R, 2 * R, ctx):
            # only holes fully inside the disk, with a solid rim left
            if all(math.hypot(x, y) < R - ctx.hole_size for (x, y) in h):
                keep.append(h)
        if keep:
            poly = G.Polygon(poly.exterior, keep)
    return [G.Shape(ctx.device_layer, poly, "disk", mech="released")]
```

Register `"ring": prim_ring, "disk": prim_disk`.

Note `disk` reuses `_hole_grid` and then filters by radius, so a perforated
disk is a non-rectilinear exterior with rectilinear holes — that combination
goes through `_bridge_holes`, **not** the band path, and therefore inherits the
non-manifold seam Task 6 documented. The test above only asserts that holes are
generated. If a watertight perforated disk is needed later, that is the case
that forces a real fix to `_bridge_holes`; it is out of scope here and no
example in this plan uses a perforated disk.

- [ ] **Step 3: Run, full suite, commit**

```bash
.venv/bin/python -m unittest tests.test_library.TestRingDisk -v
.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q
git add soidlc/primitives.py tests/test_library.py
git commit -m "feat(prim): ring and disk primitives on the band mesher"
```

### Task 21: Lorentz-force metric

**Files:**
- Modify: `soidlc/metrics.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: the lumped model (`k`, `m`) and `q_estimate`.
- Produces: `lorentz_force(I, B)` [N] and `lorentz_stroke(I, B)` [m], callable from `require`/`solve`.

**Physics.** A conductor of length `L` carrying current `I` across a field `B`
feels `F = B I L`. Driven at resonance, the displacement is amplified by the
quality factor: `x = F Q / k`. `L` is measured from the released span
perpendicular to the drive axis, which is the part of the structure actually
carrying current across the field.

- [ ] **Step 1: Write the failing test**

```python
class TestLorentz(unittest.TestCase):
    def test_force_is_linear_in_current_and_field(self):
        from soidlc import metrics
        f1 = metrics.lorentz_F(1e-3, 50e-6, 400e-6)
        f2 = metrics.lorentz_F(2e-3, 50e-6, 400e-6)
        f3 = metrics.lorentz_F(1e-3, 100e-6, 400e-6)
        self.assertAlmostEqual(f2 / f1, 2.0, places=9)
        self.assertAlmostEqual(f3 / f1, 2.0, places=9)

    def test_metric_is_callable_from_source(self):
        src = TestImportResolution.PROC + """
          import "flexures.soidl";
          device d {
            inst M = plate(400 um, 100 um) at (0, 0);
            inst S = array(guided_beam(L = 250 um, w = 4 um), count = 4,
                           place = corners(M));
            net GND = M | S.fixed;
            require lorentz_stroke(1 mA, 50 uT) >= 0 um;
            constraint anchored(S.fixed);
          }
        """
        art = compile_source(src)
        self.assertEqual(art.errors, [], art.errors)
```

Run it. Expected: FAIL — `lorentz_F` does not exist.

- [ ] **Step 2: Implement**

Add to `soidlc/metrics.py` at module level:

```python
def lorentz_F(current_a: float, field_t: float, length_m: float) -> float:
    """Lorentz force [N] on a conductor of length `length_m` carrying
    `current_a` across a field of `field_t` tesla:  F = B*I*L."""
    return field_t * current_a * length_m
```

and inside `build_env`:

```python
    def lorentz_force(I, B, *_a, **_k) -> Quantity:
        L_m = _lorentz_span()
        i = I.as_float() if isinstance(I, Quantity) else float(I)
        b = B.as_float() if isinstance(B, Quantity) else float(B)
        return _q(lorentz_F(i, b, L_m), (1, 1, -2, 0))

    def lorentz_stroke(I, B, *_a, **_k) -> Quantity:
        k = _model_q("k", "stiffness").value
        f = lorentz_force(I, B).value
        q = q_estimate().as_float()
        return _q(f * q / k, (1, 0, 0, 0))

    def _lorentz_span() -> float:
        """Released span perpendicular to the drive axis, in metres."""
        rel = [s for s in res.shapes
               if s.layer == "DEVICE" and s.mech == "released"]
        if not rel:
            raise MetricError("lorentz_force() needs released geometry")
        bb = _G.bbox_of([s for s in rel])
        return max(bb[2] - bb[0], bb[3] - bb[1]) * 1e-6
```

Register `env["lorentz_force"] = lorentz_force` and
`env["lorentz_stroke"] = lorentz_stroke` next to the others, and add
`from . import geometry as _G` to `metrics.py`'s imports (it is stdlib-only, so
the dependency rule in Global Constraints still holds).

Check that `soidlc/units.py` knows `mA` and `uT`
(`grep -n "'T'\|\"T\"\|tesla" soidlc/units.py`). If tesla is not a known unit,
add it as `kg·s⁻²·A⁻¹` — dimension tuple `(0, 1, -2, -1)` — following however
the file registers its other derived units. Without it, `require
lorentz_stroke(1 mA, 50 uT) >= 0 um` fails to parse.

- [ ] **Step 3: Run, full suite, commit**

```bash
.venv/bin/python -m unittest tests.test_library.TestLorentz -v
.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q
git add soidlc/metrics.py soidlc/units.py tests/test_library.py
git commit -m "feat(metrics): Lorentz force and resonant stroke"
```

### Task 22: Ring gyroscope and Lorentz magnetometer

**Files:**
- Create: `examples/ring_gyro.soidl`, `examples/lorentz_magnetometer.soidl`

**Interfaces:**
- Consumes: Tasks 20-21.
- Produces: the final two examples. `test_examples_have_no_errors` reaches 15 files.

- [ ] **Step 1: Write `examples/ring_gyro.soidl`**

```soidl
// Vibrating-ring gyroscope.
//
// A ring has two degenerate wine-glass (cos 2-theta) modes 45 degrees apart.
// Drive one; rotation about the normal axis Coriolis-couples energy into the
// other, and its amplitude is the rate signal.  The degeneracy is the whole
// design: a square would split the two modes and destroy the transfer, which
// is why this is the one device here that must be round.
//
// The ring hangs on compliant radial spokes -- stiff enough to hold it, soft
// enough not to perturb the mode shapes.
// Compile:  python -m soidlc.cli examples/ring_gyro.soidl

process soi25 {
  stack {
    layer METAL  { thickness = 0.52 um; material = Au }
    layer DEVICE { thickness = 25 um;   material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;    material = SiO2 }
    layer HANDLE { thickness = 400 um;  material = Si }
  }
  rules {
    release { hole_size = 5 um; hole_pitch = 25 um; max_solid_span = 30 um }
  }
}

// eight radial spokes from a central hub out to the ring
component ring_suspension(R = 200 um, w = 4 um, hub = 40 um) {
  mech port fixed, ring_port;

  geometry {
    anchor(hub, hub) at (0, 0);
    // four orthogonal spokes; the ring's 4-fold symmetry keeps the two
    // wine-glass modes degenerate
    beam(R - hub/2, w, dir = x) at (R/2 + hub/4, 0);
    beam(R - hub/2, w, dir = x) at (0 - R/2 - hub/4, 0);
    beam(R - hub/2, w, dir = y) at (0, R/2 + hub/4);
    beam(R - hub/2, w, dir = y) at (0, 0 - R/2 - hub/4);
  }

  check w >= 2 um;
}

device ring_gyro {
  inst RG  = ring(R = 200 um, w = 20 um, n_seg = 96) at (0, 0);
  inst SUS = ring_suspension(R = 200 um, w = 4 um, hub = 40 um) at (0, 0);

  require f_res(RG) >= 5 kHz;

  net GND = RG | SUS.fixed;
  constraint anchored(SUS.fixed);
}
```

- [ ] **Step 2: Compile it**

Run: `.venv/bin/python -m soidlc.cli examples/ring_gyro.soidl -o /tmp/ringgyro`
Expected: no errors, and a watertight mesh (Task 1's example test plus Task 6's
band mesher cover this). If connectivity reports the ring and the spokes as
separate islands, the spokes stop short of the ring's inner radius (`R - w/2 =
190 um`) — lengthen them until they overlap it.

- [ ] **Step 3: Write `examples/lorentz_magnetometer.soidl`**

```soidl
// Resonant Lorentz-force magnetometer.
//
// An AC current at the beam's resonance, crossed with the field being
// measured, drives the beam at resonance; the amplitude is proportional to B
// and amplified by Q.  No magnetic material anywhere -- the sensor is plain
// silicon, which is exactly why it can be made in this SOI process.
// Compile:  python -m soidlc.cli examples/lorentz_magnetometer.soidl

import "flexures.soidl";

process soi25 {
  stack {
    layer METAL  { thickness = 1 um;   material = Au }
    layer DEVICE { thickness = 25 um;  material = Si;
                   E = 169 GPa; rho = 2330 kg/m^3; nu = 0.22 }
    layer BOX    { thickness = 2 um;   material = SiO2 }
    layer HANDLE { thickness = 400 um; material = Si }
  }
  rules {
    release { hole_size = 5 um; hole_pitch = 25 um; max_solid_span = 30 um }
  }
}

device lorentz_magnetometer {
  param I_drive = 1 mA;
  param B_earth = 50 uT;

  inst BAR   = plate(500 um, 80 um) at (0, 0);
  inst SUS   = array(guided_beam(L = 220 um, w = 4 um, n_beams = 2),
                     count = 4, place = corners(BAR));
  inst SENSE = combdrive(N = 30, Lf = 40 um) attach (rotor -> BAR.top);
  inst PIN   = via_metal(100 um, 100 um) at (0 - 320 um, 0 - 120 um);
  inst POUT  = via_metal(100 um, 100 um) at (320 um, 0 - 120 um);

  // at Earth's field the resonant stroke is tiny; the requirement records the
  // detection floor the readout must reach, not a comfortable margin
  require lorentz_stroke(I_drive, B_earth) >= 1 pm;

  net SIG = SENSE.stator;
  net GND = BAR | SUS.fixed;
  isolate SIG from GND by trench;
  constraint released(BAR, SUS);
  constraint anchored(SUS.fixed, SENSE.stator);
}
```

If `pm` (picometre) is not a known unit in `soidlc/units.py`, use
`0.000001 um` rather than adding a unit for one call site.

- [ ] **Step 4: Compile both, run everything**

```bash
for e in ring_gyro lorentz_magnetometer; do
  .venv/bin/python -m soidlc.cli examples/$e.soidl -o /tmp/$e || echo "FAILED: $e"
done
.venv/bin/python -m unittest tests.test_soidlc tests.test_library -q
```
Expected: `OK`, with `test_examples_have_no_errors` now covering 15 files.

- [ ] **Step 5: Verify the web viewer picks up every new example**

The backend lists examples by globbing the directory
(`webapp/backend/app.py:29`), so all 11 new devices should appear with no
frontend change:

```bash
.venv/bin/python -c "
from webapp.backend import app
print(len(app._example_names()), 'examples:', app._example_names())
"
```
Expected: 15 names. This needs `fastapi` installed
(`.venv/bin/python -m pip install -e '.[web]'`); if it is not, skip this step
and note it.

- [ ] **Step 6: Commit**

```bash
git add examples/ring_gyro.soidl examples/lorentz_magnetometer.soidl
git commit -m "feat(examples): vibrating-ring gyroscope and Lorentz magnetometer"
```

---

## Final: update the README

### Task 23: Document the standard library

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add an "Element library" section**

After the "How it works" section, document: the `import` statement and its two
resolution paths; the component table from `soidlc/lib/flexures.soidl` and
`membranes.soidl` with each one's `derive`d stiffness; the full primitive list
(now 15); and the 15 examples grouped by what they demonstrate — in-plane
resonators, thermal actuators, out-of-plane/torsional, circular.

Update the existing pipeline description at step 4, which lists the primitives
as "`beam`, `plate`, `comb`/`combdrive`, `anchor`, `gap_stop`, `via_metal`,
`trench`" — `trench` is no longer a no-op and there are seven more.

- [ ] **Step 2: Record what the FEM validation measured**

State the measured lumped-vs-FEM agreement for `folded_flexure_resonator`,
`tuning_fork_detf` and `teeter_totter_accel` as actual numbers from the test
runs, not as "within 15%". The README already reports real numbers elsewhere
(the connectivity check's two caught bugs, the 3D-vs-2D mode comparison); match
that standard.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document the SOIDL element library and its validation"
```
