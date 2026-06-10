# SOIDL → 3D model compiler (`soidlc`)

`soidlc` is a compiler for **SOIDL** — a declarative, parametric language for
describing MEMS devices fabricated in an SOI process (DRIE of the device
layer, sacrificial BOX release, optional backside trench), in the spirit of
SOIMUMPs.

This repository implements the *"compile a SOIDL description into a 3D model"*
path from the language spec: it parses a `.soidl` source, elaborates the
component/device hierarchy with **unit-checked** expressions, synthesises
per-layer 2D geometry, and **extrudes the layer stack into a watertight 2.5D
3D mesh**, which it exports as STL and OBJ/MTL plus an SVG top-view and a
rendered PNG preview.

It is **pure standard-library Python — zero dependencies** (the parser,
geometry kernel, mesh extruder, software renderer and all exporters are
hand-written).

| comb resonator | in-plane accelerometer |
| --- | --- |
| ![comb resonator](docs/comb_resonator.png) | ![accelerometer](docs/accelerometer.png) |

*Gray = HANDLE substrate, blue = DEVICE silicon (suspended structures float
above the etched BOX), gold = METAL bond pad. Release holes are generated
automatically from the process rules.*

## Quick start

```bash
python -m soidlc.cli examples/comb_resonator.soidl
# or, after `pip install -e .`
soidlc examples/comb_resonator.soidl -o out/comb
```

Output (written next to `-o/--out` prefix, default `out/<stem>`):

```
process : soi25 (4 layers)
mesh    : 2520 vertices, 4836 triangles; bbox = [396.0 x 588.0 x 427.0] um
model   : m=6.07e-09 [mass], k=406.3 [stiffness], f0=41192.7 [freq]
files   :
          stl  out/comb_resonator.stl     # 3D mesh (binary STL)
          obj  out/comb_resonator.obj     # 3D mesh + per-layer materials
          svg  out/comb_resonator.svg     # top-view layout preview
          png  out/comb_resonator.png     # shaded isometric render
report  : derive/check/solve results with evaluated values
```

CLI flags: `-o/--out <prefix>`, `-d/--device <name>` (which device to build),
`--no-handle` (omit the substrate slab), `-q/--quiet`.

## How it works

The pipeline mirrors the `soidlc` stages from the spec:

1. **Lex + parse** (`lexer.py`, `parser.py`) — recursive-descent parser for
   `process` / `component` / `device`. Numeric literals carry units
   (`200 um`, `20 kHz`, `2330 kg/m^3`); the lexer attaches a unit to a number
   only if the trailing word is a real unit.
2. **Units** (`units.py`) — every quantity is stored in SI base units with a
   dimension vector `(length, mass, time, current)`. Arithmetic propagates
   dimensions and a mismatch (`length + freq`) is an error. `derive`/`check`
   expressions are evaluated and reported with their dimensions.
3. **Elaboration** (`elaborate.py`) — resolves parameters, expands `repeat`
   loops and component hierarchy, instantiates `array(...)`, and applies
   placement: `at (x, y)`, `attach (rotor -> M.left)`, and
   `place = corners(M)` (resolved from instance bounding boxes). Produces a
   flat list of layer-tagged 2D polygons in micrometres.
4. **Geometry synthesis** (`primitives.py`) — `beam`, `plate` (with
   auto-generated release holes from the process `release` rules), `comb`/
   `combdrive`, `anchor`, `gap_stop`, `via_metal` (METAL), `trench`.
5. **Mechanical status from the stack** — each DEVICE polygon is `anchored`
   (BOX kept beneath, tied to the handle) or `released` (floating above the
   BOX gap). This drives the 3D build, not an annotation.
6. **2.5D extrusion** (`mesh.py`, `build3d.py`) — the process `stack`
   assigns each layer a z-range; polygons are extruded into prisms. Anchored
   silicon gets a BOX pillar; a HANDLE slab spans the footprint. Rectilinear
   shapes use a shared-grid voxel-surface mesher → **watertight,
   T-junction-free** geometry (verified in tests).
7. **Model extraction** — a lumped model is assembled from the geometry:
   suspended mass `m` from released DEVICE area × thickness × ρ, stiffness
   `k` from folded-flexure `derive`s, and the resonant frequency
   `f0 = √(k/m)/2π`.
8. **Export** (`exporters.py`, `svg.py`, `render.py`) — binary STL, OBJ+MTL
   (per-layer colours), SVG top view, and a z-buffered isometric PNG from a
   built-in software rasteriser.

## Supported SOIDL subset (v0.1)

Implemented: `process { stack / masks / rules }`, `component(params) { port,
derive, geometry, check }`, `device { inst, net, isolate, constraint, check,
solve }`; `repeat` loops, `array`, placement/attachment, unit-checked
expressions, primitives listed above.

Best-effort / partial: `solve` (a numeric fallback length is used unless a
closed-form is known); `net`/`isolate`/`constraint` are parsed and reported
but full connectivity extraction and DRC are not yet enforced. Behavioural
exports (Verilog-A / SPICE) and FEM hand-off are out of scope for this v0.1,
which targets the 3D-model deliverable. Unknown functions inside
`derive`/`check` degrade gracefully to a symbolic report entry rather than
failing the build.

## Layout

```
soidlc/
  units.py       dimensional quantities + unit parsing
  lexer.py       tokenizer (unit-aware numbers)
  parser.py      recursive-descent parser
  sast.py        AST node definitions
  primitives.py  primitive -> 2D polygon generators
  elaborate.py   hierarchy expansion, placement, model extraction
  geometry.py    2D polygon kernel
  mesh.py        triangulation + 2.5D extrusion (watertight grid mesher)
  build3d.py     stack-aware mesh assembly (anchors/box/handle)
  exporters.py   STL + OBJ/MTL writers
  svg.py         top-view SVG preview
  render.py      pure-Python PNG software renderer
  cli.py         `soidlc` command-line entry point
examples/        comb_resonator.soidl, accelerometer.soidl
tests/           unittest suite (units, parsing, watertight meshes, e2e)
```

## Tests

```bash
python -m unittest discover -s tests -v
```
