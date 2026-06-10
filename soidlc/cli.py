"""Command-line driver: ``soidlc`` compiles a .soidl file into a 3D model."""

from __future__ import annotations

import argparse
import sys

from . import compile_file, exporters


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="soidlc",
        description="Compile a SOIDL SOI-MEMS description into a 3D model "
                    "(STL + OBJ/MTL) with an SVG top-view preview.")
    ap.add_argument("input", help="path to a .soidl source file")
    ap.add_argument("-o", "--out", default=None,
                    help="output prefix (default: out/<input stem>)")
    ap.add_argument("-d", "--device", default=None,
                    help="name of the device to elaborate (default: last)")
    ap.add_argument("--no-handle", action="store_true",
                    help="omit the HANDLE substrate slab")
    ap.add_argument("--fem", action="store_true",
                    help="run the built-in 2D plane-stress FEM "
                         "(modal analysis of each suspended island)")
    ap.add_argument("--fem-h", type=float, default=12.0, metavar="UM",
                    help="FEM target element size in um (default 12)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    out = args.out
    if out is None:
        import os
        stem = os.path.splitext(os.path.basename(args.input))[0]
        out = os.path.join("out", stem)

    try:
        art = compile_file(args.input, device=args.device, out_prefix=out,
                           include_handle=not args.no_handle,
                           fem=args.fem, fem_h=args.fem_h)
    except Exception as e:  # noqa: BLE001 - surface a clean message
        print(f"soidlc: error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"process : {art.process.name} "
              f"({len(art.process.layers)} layers)")
        print(f"mesh    : {exporters.mesh_stats(art.mesh)}")
        if art.model:
            ms = ", ".join(f"{k}={v!r}" for k, v in art.model.items())
            print(f"model   : {ms}")
        for w in art.warnings:
            print(f"warn    : {w}")
        print("files   :")
        for kind, path in art.files.items():
            print(f"          {kind:4s} {path}")
        if art.report:
            print("report  :")
            for line in art.report:
                print(f"          {line}")
    # connectivity / netlist violations are compile errors (printed even in
    # quiet mode); artifacts are still written to aid debugging
    for e in art.errors:
        print(f"soidlc: ERROR: {e}", file=sys.stderr)
    return 2 if art.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
