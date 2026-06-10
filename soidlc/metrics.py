"""Device-level metric functions for design closure and `require` checks.

These are the quantities a specification talks about — resonant frequency,
stroke at a drive voltage, quality factor, die area — computed from a freshly
elaborated geometry (lumped model + transducers measured from the polygons).
They are injected into the expression evaluator as callables, so SOIDL source
can write ``solve S.L such that f_res(M, S) == f0_target`` or
``require stroke_max(V_drive) >= 5 um``.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .units import Quantity

MU_AIR = 1.85e-5


class MetricError(Exception):
    """A metric cannot be computed for this device (e.g. no flexures)."""


def _q(value: float, dim) -> Quantity:
    return Quantity(value, dim)


def build_env(elab, res) -> Dict[str, object]:
    """Environment of metric callables bound to one elaborated result."""
    from . import reduce as _reduce      # lazy: avoids an import cycle

    env: Dict[str, object] = dict(getattr(elab, "device_env", {}) or {})
    model = res.model or {}
    proc = elab.process
    # FEM-in-the-loop calibration factors set by the closure stage: they
    # absorb the systematic lumped-vs-FEM bias so that solving against the
    # metric drives the *FEM-predicted* value onto the spec
    cal = getattr(elab, "metric_calibration", {}) or {}

    trans = _reduce.find_transducers(
        res.shapes, getattr(elab, "device_ast", None), proc)
    drive = next((t for t in trans if t.net.upper().startswith("DRIVE")),
                 trans[0] if trans else None)

    def _model_q(key: str, what: str) -> Quantity:
        v = model.get(key)
        if not isinstance(v, Quantity):
            raise MetricError(f"{what} not available "
                              f"(no extractable {key} in this device)")
        return v

    def _damping() -> float:
        box = proc.box()
        gap = (box.thickness if box else 2.0) * 1e-6
        a_rel = sum(s.polygon.area() for s in res.shapes
                    if s.layer == "DEVICE" and s.mech == "released") * 1e-12
        b = MU_AIR * a_rel / gap
        dev = proc.device()
        t_m = (dev.thickness if dev else 25.0) * 1e-6
        for tr in trans:
            b += MU_AIR * (tr.N * 2 * tr.ov_um * 1e-6 * t_m) \
                / (tr.g_um * 1e-6)
        return b

    def f_res(*_a, **_k) -> Quantity:
        q = _model_q("f0", "resonant frequency")
        return _q(q.value * cal.get("f_res", 1.0), q.dim)

    def q_estimate(*_a, **_k) -> Quantity:
        m = _model_q("m", "Q").value
        f0 = _model_q("f0", "Q").value
        return _q(2 * math.pi * f0 * m / _damping(), (0, 0, 0, 0))

    def stroke_max(V, *_a, **_k) -> Quantity:
        """Resonant drive amplitude: Q * F_comb(V) / k."""
        if drive is None:
            raise MetricError("stroke_max: no comb transducer found")
        if not isinstance(V, Quantity):
            raise MetricError("stroke_max expects a voltage argument")
        k = _model_q("k", "stroke").value
        F = 0.5 * drive.dcdx * V.value ** 2
        Q = q_estimate().value
        return _q(Q * F / k, (1, 0, 0, 0))

    def stroke_static(V, *_a, **_k) -> Quantity:
        if drive is None:
            raise MetricError("stroke_static: no comb transducer found")
        k = _model_q("k", "stroke").value
        return _q(0.5 * drive.dcdx * V.value ** 2 / k, (1, 0, 0, 0))

    def area(*_a, **_k) -> Quantity:
        x0, y0, x1, y1 = res.bbox
        return _q((x1 - x0) * (y1 - y0) * 1e-12, (2, 0, 0, 0))

    env.update({
        "f_res": f_res,
        "Q_estimate": q_estimate,
        "stroke_max": stroke_max,
        "stroke_static": stroke_static,
        "area": area,
    })
    return env
