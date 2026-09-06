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

from . import geometry as _G
from .units import Quantity

MU_AIR = 1.85e-5
KB = 1.380649e-23
EPS0 = 8.8541878128e-12
ALPHA_SI = 2.6e-6           # 1/K, linear thermal expansion of silicon


def lorentz_F(current_a: float, field_t: float, length_m: float) -> float:
    """Lorentz force [N] on a conductor of length `length_m` carrying
    `current_a` across a field of `field_t` tesla:  F = B*I*L.

    No magnetic material anywhere -- which is exactly why a resonant
    magnetometer can be built in a plain SOI process.
    """
    return field_t * current_a * length_m
VOLTAGE = (2, 1, -3, -1)


def chevron_stroke(L_m: float, angle_deg: float, dT: float) -> float:
    """Shuttle displacement [m] of a chevron beam of length `L_m` inclined by
    `angle_deg`, heated by `dT` kelvin.

    A beam's two endpoints stay one beam-length apart, so heating it to
    L*(1+e) with e = alpha*dT lifts the shuttle to
    y = L*sqrt((1+e)^2 - cos^2 a).  Used in exact form rather than the
    small-angle d ~ L*e/sin(a), because shallow angles -- where a chevron is
    most useful, since they trade force for stroke -- are exactly where that
    approximation is worst.
    """
    a = math.radians(angle_deg)
    e = ALPHA_SI * dT
    inner = (1.0 + e) ** 2 - math.cos(a) ** 2
    if inner <= 0.0:
        return 0.0
    return L_m * (math.sqrt(inner) - math.sin(a))


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
        """Resonant frequency.  `axis = theta` asks for the torsional mode.

        A torsional mode is governed by the mass moment of inertia, not the
        mass, so there is no honest translational answer for a device that has
        only a torsional one -- sqrt(k/m) would return a plausible, meaningless
        number.  This raises instead.
        """
        axis = _k.get("axis")
        axis_name = getattr(axis, "id", axis)      # bare `theta` arrives as a name
        if axis_name in ("theta", "torsion", "torsional"):
            v = model.get("f0_theta")
            if not isinstance(v, Quantity):
                raise MetricError(
                    "f_res(axis = theta): this device has no torsional mode "
                    "(no k_theta was derived)")
            return v
        if not isinstance(model.get("f0"), Quantity) \
                and isinstance(model.get("f0_theta"), Quantity):
            raise MetricError(
                "this device has only a torsional mode; call "
                "f_res(axis = theta). A translational f_res would divide a "
                "torsional stiffness by a mass and return a meaningless "
                "number")
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

    def _drive_voltage() -> float:
        q = env.get("V_drive")
        if isinstance(q, Quantity) and q.dim == (2, 1, -3, -1):
            return q.value
        for v in env.values():
            if isinstance(v, Quantity) and v.dim == (2, 1, -3, -1):
                return v.value
        return 1.0

    def arw(T=None, *_a, **_k) -> Quantity:
        """Brownian-limited angle random walk, returned as a dimensionless
        number in deg/sqrt(hour) (IEEE-952 convention: sigma_Allan(tau) =
        ARW/sqrt(tau)).

        Fluctuation-dissipation: the gas that damps the structure (b) also
        shakes it with force PSD 4*kB*T*b, indistinguishable from the
        Coriolis force 2*m*Omega*x_drive'.
        """
        if drive is None:
            raise MetricError("arw: no comb transducer found")
        Tv = (T.value if isinstance(T, Quantity)
              else float(T) if T is not None else 300.0)
        m = _model_q("m", "ARW").value
        k = _model_q("k", "ARW").value
        f0 = _model_q("f0", "ARW").value
        w = 2 * math.pi * f0
        b = _damping()
        V = _drive_voltage()
        F = 0.5 * drive.dcdx * V * V
        Q = w * m / b
        x_d = Q * F / k                         # resonant drive amplitude
        s_rate = 4 * KB * Tv * b / (2 * m * w * x_d) ** 2   # (rad/s)^2/Hz
        arw_si = math.sqrt(s_rate / 2.0)        # rad/sqrt(s), IEEE-952
        return _q(arw_si * (180.0 / math.pi) * 60.0, (0, 0, 0, 0))

    def pull_in(*_a, **_k) -> Quantity:
        """Collapse voltage of the device's gap-closing pair.

        A parallel plate is stable only while the mechanical restoring force
        outruns the electrostatic one; it loses that race at x = g/3, giving
        V_pi = sqrt(8*k*g^3 / (27*eps0*A)).
        """
        pp = next((t for t in trans if t.axis == 1 and t.C0 > 0
                   and t.ov_um > 0), None)
        if pp is None:
            raise MetricError("pull_in() needs a parallel_plate transducer")
        k = _model_q("k", "pull-in voltage").value
        g = pp.g_um * 1e-6
        area = pp.C0 * g / EPS0           # recover A from C0 = eps0*A/g
        if area <= 0:
            raise MetricError("pull_in(): degenerate plate overlap")
        return _q(math.sqrt(8.0 * k * g ** 3 / (27.0 * EPS0 * area)), VOLTAGE)

    def stroke_thermal(dT, *_a, **_k) -> Quantity:
        """Chevron shuttle stroke at a temperature rise dT [K].

        Takes the temperature rise, not the drive power.  Converting power to
        temperature needs a thermal-resistance model (conduction down the
        beams into the anchors, plus the air gap to the substrate) that this
        compiler does not have; inventing one would return a number that looks
        authoritative and is not.  Temperature is the quantity the geometry
        actually determines the stroke from.
        """
        beams = [s for s in res.shapes if s.label == "chevron_beam"]
        if not beams:
            raise MetricError("stroke_thermal() needs a chevron actuator")
        bb = beams[0].polygon.bbox()
        run = abs(bb[2] - bb[0])
        rise = abs(bb[3] - bb[1])
        L_m = math.hypot(run, rise) * 1e-6
        ang = math.degrees(math.atan2(rise, run)) if run > 0 else 90.0
        dt = dT.value if isinstance(dT, Quantity) else float(dT)
        return _q(chevron_stroke(L_m, ang, dt), (1, 0, 0, 0))

    def _lorentz_span() -> float:
        """Released span carrying current across the field, in metres."""
        rel = [sh for sh in res.shapes
               if sh.layer == "DEVICE" and sh.mech == "released"]
        if not rel:
            raise MetricError("lorentz_force() needs released geometry")
        bb = _G.bbox_of(rel)
        return max(bb[2] - bb[0], bb[3] - bb[1]) * 1e-6

    def lorentz_force(I, B, *_a, **_k) -> Quantity:
        i = I.value if isinstance(I, Quantity) else float(I)
        b = B.value if isinstance(B, Quantity) else float(B)
        return _q(lorentz_F(i, b, _lorentz_span()), (1, 1, -2, 0))

    def lorentz_stroke(I, B, *_a, **_k) -> Quantity:
        """Displacement at resonance: F*Q/k.

        Driven at the beam's resonance, the Lorentz force is amplified by the
        quality factor, which is the only reason an Earth-field signal is
        detectable at all.
        """
        k = _model_q("k", "Lorentz stroke").value
        f = lorentz_force(I, B).value
        q = q_estimate().value
        return _q(f * q / k, (1, 0, 0, 0))

    env.update({
        "lorentz_force": lorentz_force,
        "lorentz_stroke": lorentz_stroke,
        "pull_in": pull_in,
        "stroke_thermal": stroke_thermal,
        "f_res": f_res,
        "Q_estimate": q_estimate,
        "stroke_max": stroke_max,
        "stroke_static": stroke_static,
        "area": area,
        "arw": arw,
    })
    return env
