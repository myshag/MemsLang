"""SOIDL — a compiler from SOI MEMS device descriptions to 3D models.

Public entry points::

    from soidlc import compile_file
    artifacts = compile_file("device.soidl", out_prefix="out/device")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import build3d, exporters, render, svg
from .elaborate import Elaborator, InstanceResult, ProcessInfo
from .parser import parse

__version__ = "0.1.0"


@dataclass
class Artifacts:
    process: ProcessInfo
    result: InstanceResult
    mesh: object
    report: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)
    model: Dict[str, object] = field(default_factory=dict)


def compile_source(src: str, device: Optional[str] = None,
                   out_prefix: Optional[str] = None,
                   include_handle: bool = True,
                   fem: bool = False, fem_h: float = 12.0,
                   fem_closure: bool = False) -> Artifacts:
    ast = parse(src)
    elab = Elaborator(ast)

    # design closure: solve free parameters against the spec, then do the
    # final (loud) elaboration with the solved values
    from . import closure
    overrides = closure.run(elab, device, fem_calibrate=fem_closure,
                            fem_h=fem_h if fem else 20.0)
    result = elab.elaborate_device(device, overrides=overrides)
    mesh = build3d.build_mesh(result, elab.process, include_handle=include_handle)

    art = Artifacts(elab.process, result, mesh, elab.report, elab.warnings,
                    elab.errors, model=result.model)
    closure.enforce_requires(elab, art, device)

    if out_prefix:
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        stl = out_prefix + ".stl"
        obj = out_prefix + ".obj"
        svgf = out_prefix + ".svg"
        png = out_prefix + ".png"
        exporters.write_stl(mesh, stl)
        exporters.write_obj(mesh, obj, build3d.layer_colors(elab.process))
        svg.write_svg(result, elab.process, svgf)
        render.render_mesh(mesh, build3d.layer_colors(elab.process), png)
        art.files = {"stl": stl, "obj": obj, "svg": svgf, "png": png,
                     "mtl": out_prefix + ".mtl"}

    if fem:
        from . import fem2d
        fem2d.analyze(elab, art, h=fem_h, plot_prefix=out_prefix)
    return art


def compile_file(path: str, **kw) -> Artifacts:
    with open(path) as f:
        return compile_source(f.read(), **kw)
