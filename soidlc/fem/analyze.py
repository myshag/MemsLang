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
