# Element library — SOIDL standard library, new primitives, new devices

**Date:** 2026-09-05
**Status:** Approved (user: "да, вариант C, пиши спеку")
**Roadmap:** grows the language's vocabulary; no change to the compiler pipeline stages

## Goal

The library today is 8 Python primitives (`beam`, `anchor`, `plate`,
`comb`/`combdrive`, `gap_stop`, `trench`, `via_metal`) and 4 example devices.
Everything expressible is a rectangle, in-plane, electrostatically driven by a
comb. Three consequences:

- **No standard library.** `beam_flexure` (comb_resonator) and `suspension`
  (accelerometer) are the same clamped-guided flexure written twice, inline,
  because there is no way to share a component between files.
- **One flexure type, one actuator type.** No folded flexure, serpentine or
  crab-leg; no gap-closing plates, no thermal actuators.
- **Half the declared process is unreachable.** `layer HANDLE` and
  `mask TRENCH -> etch(HANDLE, through, backside)` are declared by every
  example process and used by none, because `prim_trench` returns `[]` and
  `build3d` lays HANDLE down as an unbroken slab. Membranes and torsional
  out-of-plane devices cannot be described.

This spec adds an `import` mechanism plus a standard library written **in
SOIDL**, seven new Python primitives for geometry that composition cannot
express, backside-trench support in the 3D builder, and 11 new example
devices.

## Approach: hybrid library (option C)

Considered and rejected:

- **(A) Everything as Python primitives.** Fastest, but the library of a
  language for describing MEMS would then not itself be written in that
  language, and a user could not add an element without editing the compiler.
- **(B) Everything in SOIDL.** Dogfooding, but `comb`, `ring` and `chevron`
  are not compositions of beams — they are parametric point-set generators.
  Forcing them into SOIDL means adding loops-with-trigonometry to the
  language, which is a much larger change than adding a primitive.

**Chosen: (C).** Irreducible geometry stays in `soidlc/primitives.py`.
Everything expressible as a composition of primitives moves to `soidlc/lib/*.soidl`,
shipped with the compiler and pulled in with a new `import` statement. This
also deduplicates the two copies of the clamped-guided flexure that exist
today, and it is the design that keeps the library extensible by users who do
not touch Python.

## Phase 0 — core: `import`, path geometry

Nothing in phases 2 and 4 can land without this.

### `import` (lexer, parser, elaborator, cli)

- `soidlc/lexer.py`: add `import` to `KEYWORDS`.
- `soidlc/sast.py`: `@dataclass class Import: path: str`.
- `soidlc/parser.py`: `parse_file` accepts `import "…";` at top level, before
  the existing `process`/`component`/`device`/`chip` dispatch. Imports are
  only legal at top level.
- `soidlc/elaborate.py`: `Elaborator._index` resolves each `Import`, parses
  the imported file and merges its declarations into `self.components`
  **before** indexing the importing file's own declarations, so a local
  `component` of the same name shadows a library one (matching the existing
  rule where a user component shadows a builtin primitive, elaborate.py:418).
  Resolution order for `import "std/flexures.soidl"`:
  1. relative to the importing file's directory;
  2. relative to the bundled library directory `soidlc/lib/`.
  Cyclic imports are detected by a set of already-loaded absolute paths and
  are a no-op on revisit (not an error — diamond imports are legitimate).
  A missing file raises the existing `ParseError`-style diagnostic with the
  importing file and line.
- `soidlc/cli.py`: the parse entry point already reads one file; the
  elaborator does the recursive loading, so the CLI needs no flag.
- `pyproject.toml`: `[tool.setuptools]` lists packages explicitly, so the
  library must live inside the `soidlc` package (not a repo-root `lib/`) and
  needs `[tool.setuptools.package-data] soidlc = ["lib/*.soidl"]` to ship on
  `pip install`. Without this the standard library works from a checkout and
  silently vanishes when installed.

A library file declares `component`s only — no `process`, no `device`. The
elaborator rejects a `process` or `device` inside an imported file, because
which process a component is built against must stay the importing file's
decision (components read process values through `process.DEVICE.E`, resolved
at elaboration time against the importing file's process).

### Path geometry (`soidlc/geometry.py`)

- `wire(points, w)` → `Polygon`: a polyline of the given width. Segments are
  joined with a miter; the miter is clipped to a bevel past a 4× ratio to
  avoid spikes at sharp corners. Needed by `serpentine`, `chevron`, `route`.
- `circle(R, n_seg)` / `arc(R, w, a0, a1, n_seg)` → `Polygon`: regular-polygon
  approximation. `n_seg` defaults to 64, overridable per call. Needed by
  `ring`, `disk`.

### Watertightness of the general extrusion path

`mesh.extrude_polygon` already has two paths: a shared-grid voxel mesher for
rectilinear polygons and `_extrude_general` (ear clipping + hole bridging) for
everything else. Only the rectilinear path is covered by a watertightness
test today. Since every non-rectangular element in phases 2 and 4 lands on
`_extrude_general`, this phase adds the missing coverage: a test that an
`arc`-derived ring with holes extrudes to a closed 2-manifold (every edge
shared by exactly two triangles), reusing the existing watertightness helper
in `tests/test_soidlc.py`.

If `_extrude_general` proves not watertight for bridged holes, the fallback is
to emit ring geometry as a hole-free outline (an annulus becomes two nested
solids rather than one holed solid). This is a rendering-fidelity compromise
only; it does not affect connectivity or the lumped model.

## Phase 1 — flexures (SOIDL library, zero geometric risk)

New file `soidlc/lib/flexures.soidl`, all rectangles, all compositions of `beam` and
`anchor`:

| component | parameters | `derive` |
| --- | --- | --- |
| `guided_beam` | `L`, `w`, `n_beams` | `k.x = n*E*t*w^3/L^3` |
| `folded_flexure` | `L`, `w`, `n_folds`, `truss_w` | `k.x = n*E*t*w^3/(2L^3)` |
| `serpentine` | `L`, `w`, `n_turns`, `pitch` | `k.x` per meander count |
| `crab_leg` | `Lx`, `Ly`, `w` | `k.x` from the two-segment L-beam |
| `frame` | `W`, `H`, `bar` | — (a square ring of four bars) |

`guided_beam` replaces the duplicated `beam_flexure` / `suspension`
components; `comb_resonator.soidl` and `accelerometer.soidl` are rewritten to
import it. `frame` replaces the four hand-placed `beam(...)` calls with
hand-computed coordinates in `gyroscope_2mass.soidl`.

The stiffness formulas are the deliverable here, not the polygons — they feed
`solve`/`require` through `_extract_device_model`, which sums `k_x` across
instances (elaborate.py:603). Phase 1's FEM validation (below) is what proves
them.

New examples:

- `examples/folded_flexure_resonator.soidl` — the Tang-style folded-beam comb
  resonator; the folded flexure's point is that it relieves axial stress, so
  its `f0` should stay put where `guided_beam`'s drifts.
- `examples/tuning_fork_detf.soidl` — double-ended tuning fork, two tines
  driven anti-phase.
- `examples/low_g_accel.soidl` — serpentine suspension, deliberately compliant.

## Phase 2 — actuators

New Python primitives in `soidlc/primitives.py`, registered in `PRIMITIVES`:

- `parallel_plate(W, H, g, n)` — gap-closing electrode pairs: a released plate
  facing an anchored one across `g`. Unlike `comb`, capacitance is nonlinear
  in displacement, which is the point.
- `chevron(n, L, w, angle)` — V-beam thermal actuator: `n` pairs of beams
  inclined by `angle` from a central shuttle to two anchors. Built on
  `wire()`.
- `hot_arm(L, w_hot, w_cold, g)` — U-shaped hot-arm/cold-arm thermal actuator.

New metrics in `soidlc/metrics.py` (`build_env`), so they are callable from
`require`/`solve`:

- `pull_in(g)` → `V_pi = sqrt(8 k g^3 / (27 eps0 A))`, the collapse voltage of
  the gap-closing pair. Raises `MetricError` when the device has no
  parallel-plate transducer.
- `stroke_thermal(P)` → displacement at drive power `P`, from the
  chevron geometry and silicon's thermal expansion.

Both need `find_transducers` (`soidlc/reduce.py:59`) extended to recognise
`parallel_plate` and `chevron` labels alongside the comb labels it matches
today.

New examples: `thermal_actuator.soidl` (chevron plus `gap_stop` travel limit),
`microgripper.soidl` (two opposing chevron-driven arms), `rf_switch.soidl`
(cantilever pulled down onto a `parallel_plate` electrode).

**Known wrinkle:** `Polygon.rotated` is exact only for multiples of 90°
(geometry.py:51). Chevron beams are inclined by ~5–10°, so their coordinates
carry float error and `_is_rectilinear` will reject them — they fall through
to `_extrude_general`, which is correct and expected. Phase 0's watertightness
test is what makes this safe.

## Phase 3 — out-of-plane

The largest real change, because it touches the 3D builder rather than only
adding elements.

### Backside trench in the stack (`soidlc/build3d.py`, `soidlc/primitives.py`)

- `prim_trench(W, H)` stops returning `[]` and returns a `Shape` on a new
  `TRENCH` pseudo-layer, carrying the backside opening's footprint.
- `build_mesh` subtracts the union of TRENCH footprints from the HANDLE slab
  and from the BOX pillars beneath it, instead of extruding an unbroken slab
  (build3d.py:32). Rectangular subtraction only: the HANDLE slab becomes a
  rectilinear polygon with holes, which the existing voxel mesher already
  handles, so no new meshing risk.
- A DEVICE plate over a TRENCH footprint is a **membrane**: still continuous
  with its anchored rim (so connectivity is unchanged), but with nothing
  beneath it. `mech` stays `anchored`; the trench is what makes it move.

### New elements

- `torsion_bar(L, w)` — Python primitive is unnecessary (it is a `beam`), so
  it lives in `soidlc/lib/flexures.soidl` with `derive k_theta = G*J/L`, where
  `J` is the rectangular-section torsion constant and `G = E/(2(1+nu))`.
- `membrane(W, H)` — `soidlc/lib/membranes.soidl`: a `plate` with holes suppressed
  plus the `trench` beneath it.
- `pad(w, h)` / `route(points, w)` — Python primitives on METAL, replacing the
  single-rectangle `via_metal` for real interconnect. `route` is `wire()` on
  the metal layer.

### Torsional lumped model (`soidlc/elaborate.py`, `soidlc/reduce.py`)

`_extract_device_model` computes `m` from released area×thickness×ρ and
`f0 = sqrt(k/m)/2π` — purely translational. A torsional mode needs the mass
moment of inertia, not the mass; feeding a mirror's mass into that formula
produces a meaningless number rather than an error, which is the dangerous
failure.

`_extract_device_model` gains a parallel torsional path: when instances
contribute `k_theta` (rather than `k_x`), it accumulates `J_m = Σ ∫r² dm` over
released DEVICE polygons about the torsion axis and reports
`f0_theta = sqrt(k_theta/J_m)/2π`. A device that yields both gets both, under
distinct model keys; `f_res()` prefers the translational one and gains an
`axis` argument to ask for the torsional one. A device that yields `k_theta`
and no `k_x` must not silently report a translational `f0` — that case raises
`MetricError`.

New examples: `micromirror.soidl` (torsion bars plus buried drive electrodes),
`teeter_totter_accel.soidl` (asymmetric z-axis proof mass on torsion bars),
`pressure_sensor.soidl` (membrane over a backside cavity — the first example
that exercises `TRENCH`).

## Phase 4 — circular geometry

- `ring(R, w, n_seg)` and `disk(R, n_seg)` primitives on Phase 0's `arc`.
  `disk` gets release holes from the process rules, like `plate`.
- Lorentz-force metric in `metrics.py`: `lorentz_force(I, B)` = `B·I·L` over
  the current-carrying released span, plus the resulting displacement at
  resonance (`F·Q/k`).

New examples: `ring_gyro.soidl` (ring resonator on radial spokes),
`lorentz_magnetometer.soidl` (resonant beam carrying a current in a field).

## Testing

The existing `tests/test_soidlc.py::test_examples_have_no_errors` iterates
`examples/*.soidl`, so **every new example is covered on the day it lands**:
parse, elaborate, connectivity/LVS (split nets, shorts, floating islands,
trench isolation), and watertight mesh. This is a real check — the same test
caught the comb tip-gap short and the floating flexure anchors recorded in the
README.

Added deliberately:

1. **Phase 0:** non-rectilinear extrusion is watertight (an `arc`-derived
   annulus), covering the previously untested `_extrude_general` path.
2. **Phase 0:** `import` resolves from `soidlc/lib/`, a local component shadows a
   library one, a diamond import loads once, a cyclic import terminates, a
   missing import is a clean diagnostic.
3. **Phase 1 (FEM validation, per user request):** `folded_flexure_resonator`
   and `tuning_fork_detf` run the 2D plane-stress modal solver (`--fem`); the
   lumped `f0` from the new `derive k` must agree with the FEM mode-1
   frequency within 15%. This is what proves the new stiffness formulas
   rather than just their polygons. The tolerance matches the existing
   FEM-vs-lumped agreement band; the closure stage's calibration factors
   (`metric_calibration`) exist precisely because the bias is systematic and
   nonzero.
4. **Phase 3 (FEM validation):** `teeter_totter_accel` runs the 3D solid
   solver (`soidlc/fem/solid3d.py`) and its mode 1 must be torsional — mode
   shape dominated by antisymmetric z displacement about the torsion axis —
   and must agree with `f0_theta` within 15%. A 2D plane-stress solver cannot
   see this mode at all, which is exactly why the 3D solver was built.
5. **Phase 3:** a device with `k_theta` and no `k_x` raises `MetricError` from
   `f_res()` rather than returning a translational number.

## Risks

- **Torsional model correctness.** Mitigated by test 4: the 3D solid solver is
  an independent check on `k_theta = G*J/L` and on the inertia accumulation.
  If they disagree beyond 15%, the formula is wrong, not the tolerance.
- **`_extrude_general` watertightness with bridged holes.** Mitigated by
  test 1 landing in Phase 0, before anything depends on it, with the
  nested-solids fallback described above.
- **Backside trench touches the shared 3D build path.** Every existing example
  declares a TRENCH mask but places no trench geometry, so the subtraction is
  a no-op for them and the existing mesh tests guard against regression.
- **Scope.** 7 new Python primitives, 7 SOIDL library components, 11 new
  examples, 2 compiler changes (`import`, trench subtraction) and 2 model
  changes (torsional lumped model, new metrics). Each phase commits
  separately and leaves the tree green; phases 1–4 are independently
  droppable after Phase 0.

## Out of scope

- Non-Manhattan `attach`: edge attachment resolves against bounding boxes
  (elaborate.py:402), so attaching to a `ring` attaches to its bbox side.
  Acceptable; radial attachment is a separate change.
- Electrothermal simulation. `stroke_thermal(P)` is a closed-form estimate;
  no coupled thermal FEM.
- Squeeze-film damping for parallel plates. The existing `_damping()` model
  (Couette over the BOX gap) is kept unchanged.
