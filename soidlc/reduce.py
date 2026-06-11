"""Model order reduction: FEM modes -> behavioural (mathematical) models.

The FEM gives M-normalised mode shapes phi_i and frequencies f_i.  Projecting
the dynamics onto the first few modes turns thousands of dofs into N
decoupled oscillators

    q_i'' + 2 zeta_i w_i q_i' + w_i^2 q_i = phi_i^T f(t)

The MEMS-specific part is the coupling: each comb transducer contributes a
modal force ``u_i * F(V)`` and reads out ``x = sum u_i q_i``, where
``u_i = phi_i (at the rotor backbone, along the drive axis)`` is the modal
participation.  Effective drive-point parameters follow as
``m_eff = 1/u_1^2`` and ``k_eff = w_1^2 m_eff``.

Transducer data (N, gap, overlap, drive axis/sign, rotor attachment) is
re-derived **from the elaborated geometry**, not from annotations, so the
model can never disagree with the layout.  Gas damping (slide-film over the
BOX gap plus comb-finger films) supplies b and Q.

Exports: a standalone Python ODE module, a SPICE Butterworth–Van Dyke
subcircuit, and a Verilog-A module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import geometry as G
from . import sast as A
from .elaborate import _net_refs
from .fem import FemMesh as Mesh2D
from .units import Quantity

EPS0 = 8.8541878128e-12
MU_AIR = 1.85e-5            # Pa*s


@dataclass
class Transducer:
    name: str               # instance name
    net: str                # electrical net of the stator
    dcdx: float             # F/m
    C0: float               # F (static overlap capacitance)
    axis: int               # 0 = x, 1 = y (drive direction)
    sign: float             # +1 if attraction is toward +axis
    bbox: Tuple[float, float, float, float]   # rotor backbone, um
    N: int
    g_um: float
    ov_um: float
    u: List[float] = field(default_factory=list)   # modal participation


def _center(bb):
    return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)


def find_transducers(all_shapes: List[G.Shape], dev_ast, proc
                     ) -> List[Transducer]:
    """Locate comb drives in the elaborated geometry and measure them."""
    dev = proc.device()
    if dev is None:
        return []
    t_m = dev.thickness * 1e-6

    net_of: Dict[str, str] = {}
    if dev_ast is not None:
        for it in dev_ast.items:
            if isinstance(it, A.Net):
                for iname, port in _net_refs(it.expr):
                    if port and "stator" in port:
                        net_of[iname] = it.name

    by_owner: Dict[str, List[G.Shape]] = {}
    for s in all_shapes:
        if s.layer == "DEVICE" and s.owner:
            by_owner.setdefault(s.owner, []).append(s)

    out: List[Transducer] = []
    for owner, ss in sorted(by_owner.items()):
        rf = [s for s in ss if s.label == "rotor_finger"]
        sf = [s for s in ss if s.label == "stator_finger"]
        if not rf or not sf:
            continue
        bb0 = rf[0].polygon.bbox()
        w_, h_ = bb0[2] - bb0[0], bb0[3] - bb0[1]
        axis = 0 if w_ > h_ else 1
        per = 1 - axis
        wf = min(w_, h_)
        c0 = _center(bb0)
        near = min(sf, key=lambda s: abs(_center(s.polygon.bbox())[per]
                                         - c0[per]))
        nbb = near.polygon.bbox()
        g = abs(_center(nbb)[per] - c0[per]) - wf
        ov = max(0.0, min(bb0[axis + 2], nbb[axis + 2])
                 - max(bb0[axis], nbb[axis]))
        if g <= 0:
            continue
        bars = [s for s in ss if "rotor_bar" in s.label] or rf
        abox = G.bbox_of(bars)
        sbars = [s for s in ss if "stator_bar" in s.label]
        sign = 1.0
        if sbars:
            sc = _center(G.bbox_of(sbars))
            rc = _center(abox)
            sign = 1.0 if (sc[axis] - rc[axis]) > 0 else -1.0
        N = len(rf)
        out.append(Transducer(
            name=owner, net=net_of.get(owner, owner),
            dcdx=N * 2 * EPS0 * t_m / (g * 1e-6),
            C0=N * 2 * EPS0 * t_m * (ov * 1e-6) / (g * 1e-6),
            axis=axis, sign=sign, bbox=abox, N=N, g_um=g, ov_um=ov))
    return out


def _mean_phi(mesh: Mesh2D, dof_of, vec, bbox, axis, pad=2.0
              ) -> Optional[float]:
    x0, y0, x1, y1 = bbox
    vals = []
    for n, (px, py) in enumerate(mesh.nodes):
        if x0 - pad <= px <= x1 + pad and y0 - pad <= py <= y1 + pad:
            b = dof_of.get(n, -1)
            if b >= 0:
                vals.append(vec[b + axis])
    if not vals:
        return None
    return sum(vals) / len(vals)


def _find_vdc(elab) -> Tuple[float, str]:
    env = getattr(elab, "device_env", {}) or {}
    q = env.get("V_drive")
    if isinstance(q, Quantity) and q.dim == (2, 1, -3, -1):
        return q.value, "V_drive"
    for k, v in env.items():
        if isinstance(v, Quantity) and v.dim == (2, 1, -3, -1):
            return v.value, k
    return 1.0, "default"


def build(elab, art, island_shapes: List[G.Shape], mesh: Mesh2D,
          freqs: List[float], vecs, dof_of,
          out_prefix: Optional[str] = None) -> None:
    """Assemble the reduced model for one suspended island and export it."""
    if not freqs:
        return
    proc = elab.process
    dev = proc.device()
    trans = find_transducers(art.result.shapes,
                             getattr(elab, "device_ast", None), proc)
    tlist: List[Transducer] = []
    for tr in trans:
        us = [_mean_phi(mesh, dof_of, v, tr.bbox, tr.axis) for v in vecs]
        if us[0] is None:
            continue                      # rotor not on this island
        tr.u = [tr.sign * (u or 0.0) for u in us]
        tlist.append(tr)
    if not tlist:
        return

    drive = next((t for t in tlist if t.net.upper().startswith("DRIVE")),
                 tlist[0])
    sense = next((t for t in tlist if t is not drive), drive)
    u1 = drive.u[0]
    if abs(u1) < 1e-9:
        art.warnings.append(
            "rom: drive transducer has ~zero participation in mode 1")
        return
    w = [2 * math.pi * f for f in freqs]
    m_eff = 1.0 / (u1 * u1)
    k_eff = w[0] ** 2 * m_eff

    # gas damping: slide film under released silicon + comb finger films
    box = proc.box()
    gap_box = (box.thickness if box else 2.0) * 1e-6
    a_rel = sum(s.polygon.area() for s in island_shapes
                if s.mech == "released") * 1e-12
    b = MU_AIR * a_rel / gap_box
    t_m = dev.thickness * 1e-6
    for tr in tlist:
        b += MU_AIR * (tr.N * 2 * tr.ov_um * 1e-6 * t_m) / (tr.g_um * 1e-6)
    Q1 = w[0] * m_eff / b
    # modal damping ratios via the drive-point participation
    zetas = [max(1e-4, min(0.5, b * (tr_u ** 2) / (2 * wi)))
             for tr_u, wi in zip(drive.u, w)]

    v_dc, v_src = _find_vdc(elab)
    eta_d = v_dc * drive.dcdx
    eta_s = v_dc * sense.dcdx
    Rm = b / (eta_d * eta_s)
    Lm = m_eff / (eta_d * eta_s)
    Cm = (eta_d * eta_s) / k_eff
    f_series = 1.0 / (2 * math.pi * math.sqrt(Lm * Cm))

    # Brownian-limited angle random walk (IEEE-952): the damping b also
    # shakes the structure with force PSD 4*kB*T*b (fluctuation-dissipation)
    KB = 1.380649e-23
    T_REF = 300.0
    x_drive = Q1 * (0.5 * drive.dcdx * v_dc ** 2) / k_eff
    s_rate = 4 * KB * T_REF * b / (2 * m_eff * w[0] * x_drive) ** 2
    arw_si = math.sqrt(s_rate / 2.0)            # rad/sqrt(s)
    arw_deg = arw_si * (180.0 / math.pi) * 60.0  # deg/sqrt(h)

    for tr in tlist:
        art.report.append(
            f"rom    transducer {tr.name} (net {tr.net}): N={tr.N}, "
            f"g={tr.g_um:.2f} um, dC/dx={tr.dcdx:.3e} F/m, "
            f"C0={tr.C0 * 1e15:.1f} fF, u1={tr.u[0]:.3e}")
    art.report.append(
        f"rom    mode 1 @ drive point: m_eff={m_eff:.3e} kg, "
        f"k_eff={k_eff:.3e} N/m, b={b:.3e} N*s/m, Q≈{Q1:.0f} (air)")
    art.report.append(
        f"rom    BVD @ V_dc={v_dc:g} V ({v_src}): Rm={Rm:.3e} ohm, "
        f"Lm={Lm:.3e} H, Cm={Cm:.3e} F, f_series={f_series / 1e3:.2f} kHz")
    art.report.append(
        f"rom    ARW (Brownian, {T_REF:g} K, x_d={x_drive * 1e6:.2f} um): "
        f"{arw_deg:.4f} deg/sqrt(h)  ({arw_si:.3e} rad/sqrt(s))")

    if out_prefix:
        name = getattr(getattr(elab, "device_ast", None), "name", "device")
        py = out_prefix + "_model.py"
        cir = out_prefix + "_model.cir"
        va = out_prefix + "_model.va"
        _write_python(py, name, freqs, zetas, tlist, drive, sense, v_dc,
                      m_eff, k_eff, Q1, b, x_drive)
        _write_spice(cir, name, Rm, Lm, Cm, drive, sense, v_dc, f_series)
        _write_veriloga(va, name, freqs, zetas, tlist, drive, sense, v_dc)
        art.files["rom_py"] = py
        art.files["rom_cir"] = cir
        art.files["rom_va"] = va
        art.report.append(f"rom    files: {py}, {cir}, {va}")
        _plot_allan(art, py, out_prefix)


def _plot_allan(art, model_path: str, out_prefix: str) -> None:
    """Run the generated model's virtual rate-table experiment and plot the
    Allan deviation curve (simulated points vs the analytic ARW slope)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_rom_tmp", model_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pts, est = mod.arw_experiment(n=120000, dt=1e-3, seed=2)
        from . import femplot
        path = out_prefix + "_allan.png"
        femplot.plot_allan(pts, mod.ARW_RADS, est, path)
        art.files["rom_allan"] = path
        art.report.append(
            f"rom    Allan experiment: ARW_est={est * 3437.75:.4f} "
            f"deg/sqrt(h) vs analytic "
            f"{mod.ARW_RADS * 3437.75:.4f} -> {path}")
    except Exception as e:  # noqa: BLE001 - plotting is best-effort
        art.warnings.append(f"rom: allan plot failed: {e}")


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------
_PY_BODY = '''

def force(port, V):
    """Electrostatic comb force, N (always attractive along +u)."""
    return 0.5 * PORTS[port]["dcdx"] * V * V


def _deriv(state, Vd):
    F = force(DRIVE_PORT, Vd)
    ud = PORTS[DRIVE_PORT]["u"]
    out = []
    for i, f in enumerate(F_MODES):
        w = 2.0 * math.pi * f
        q, v = state[2 * i], state[2 * i + 1]
        out.append(v)
        out.append(-w * w * q - 2.0 * ZETA[i] * w * v + ud[i] * F)
    return out


def displacement(port, state):
    """Physical displacement at a port's rotor, metres."""
    u = PORTS[port]["u"]
    return sum(u[i] * state[2 * i] for i in range(len(F_MODES)))


def velocity(port, state):
    u = PORTS[port]["u"]
    return sum(u[i] * state[2 * i + 1] for i in range(len(F_MODES)))


def sense_current(state, v_bias=V_DC):
    """Motional current at the sense port, A."""
    return v_bias * PORTS[SENSE_PORT]["dcdx"] * velocity(SENSE_PORT, state)


def simulate(v_drive, t_end, dt=None, state=None):
    """RK4 transient.  ``v_drive(t)`` -> volts.  Returns (t, x_sense, i_sense)."""
    if dt is None:
        dt = 1.0 / (25.0 * max(F_MODES))
    if state is None:
        state = [0.0] * (2 * len(F_MODES))
    ts, xs, cur = [], [], []
    t = 0.0
    while t < t_end:
        v = v_drive(t)
        k1 = _deriv(state, v)
        k2 = _deriv([s + 0.5 * dt * k for s, k in zip(state, k1)],
                    v_drive(t + 0.5 * dt))
        k3 = _deriv([s + 0.5 * dt * k for s, k in zip(state, k2)],
                    v_drive(t + 0.5 * dt))
        k4 = _deriv([s + dt * k for s, k in zip(state, k3)],
                    v_drive(t + dt))
        state = [s + dt / 6.0 * (a + 2 * b + 2 * c + d)
                 for s, a, b, c, d in zip(state, k1, k2, k3, k4)]
        t += dt
        ts.append(t)
        xs.append(displacement(SENSE_PORT, state))
        cur.append(sense_current(state))
    return ts, xs, cur


def freq_response(f, drive=None, sense=None):
    """Complex mechanical transfer x_sense/F_drive (m/N) at frequency f."""
    drive = drive or DRIVE_PORT
    sense = sense or SENSE_PORT
    ud, us = PORTS[drive]["u"], PORTS[sense]["u"]
    w = 2.0 * math.pi * f
    H = 0j
    for i, fi in enumerate(F_MODES):
        wi = 2.0 * math.pi * fi
        H += ud[i] * us[i] / (wi * wi - w * w + 2j * ZETA[i] * wi * w)
    return H


def estimate_resonance():
    """Ring-down test: static comb deflection released at t=0; the dominant
    zero-crossing rate of the drive-point displacement gives f1."""
    F0 = force(DRIVE_PORT, V_DC)
    ud = PORTS[DRIVE_PORT]["u"]
    state = []
    for i, f in enumerate(F_MODES):
        w = 2.0 * math.pi * f
        state += [ud[i] * F0 / (w * w), 0.0]
    t_end = 30.0 / F_MODES[0]
    dt = 1.0 / (25.0 * max(F_MODES))
    t, crossings, prev, t_first, t_last = 0.0, 0, None, None, None
    while t < t_end:
        k1 = _deriv(state, 0.0)
        k2 = _deriv([s + 0.5 * dt * k for s, k in zip(state, k1)], 0.0)
        k3 = _deriv([s + 0.5 * dt * k for s, k in zip(state, k2)], 0.0)
        k4 = _deriv([s + dt * k for s, k in zip(state, k3)], 0.0)
        state = [s + dt / 6.0 * (a + 2 * b + 2 * c + d)
                 for s, a, b, c, d in zip(state, k1, k2, k3, k4)]
        t += dt
        x = displacement(DRIVE_PORT, state)
        if prev is not None and prev < 0.0 <= x:
            crossings += 1
            t_last = t
            if t_first is None:
                t_first = t
        prev = x
    if crossings < 2:
        return 0.0
    return (crossings - 1) / (t_last - t_first)


KB = 1.380649e-23
T_REF = 300.0


def equivalent_rate_psd(T=T_REF):
    """One-sided PSD of the thermal-equivalent input rate, (rad/s)^2/Hz.

    Fluctuation-dissipation: force PSD 4*kB*T*B_TOTAL referred through the
    Coriolis scale factor 2*M_EFF*w_drive*X_DRIVE.
    """
    sf = 2.0 * M_EFF * (2.0 * math.pi * F_MODES[0]) * X_DRIVE
    return 4.0 * KB * T * B_TOTAL / (sf * sf)


def arw(T=T_REF):
    """Angle random walk, rad/sqrt(s) (IEEE-952: sigma(tau)=ARW/sqrt(tau))."""
    return math.sqrt(equivalent_rate_psd(T) / 2.0)


def allan_deviation(y, dt, ms=None):
    """Overlapping Allan deviation of a rate record. -> [(tau, sigma)]."""
    n = len(y)
    if ms is None:
        ms, m = [], 1
        while m <= n // 10:
            ms.append(m)
            m *= 2
    th = [0.0] * (n + 1)            # cumulative angle
    for i, v in enumerate(y):
        th[i + 1] = th[i] + v * dt
    out = []
    for m in ms:
        tau = m * dt
        cnt = n - 2 * m + 1
        s = 0.0
        for k in range(cnt):
            d = th[k + 2 * m] - 2.0 * th[k + m] + th[k]
            s += d * d
        out.append((tau, math.sqrt(s / (2.0 * tau * tau * cnt))))
    return out


def arw_experiment(T=T_REF, n=200000, dt=1e-3, seed=2):
    """Virtual rate-table run at Omega = 0: the gyro output is pure
    thermal-equivalent rate noise; its Allan deviation follows
    sigma(tau) = ARW/sqrt(tau).  Returns ([(tau, sigma)], arw_estimate)."""
    import random
    rng = random.Random(seed)
    sig = math.sqrt(equivalent_rate_psd(T) / (2.0 * dt))
    y = [rng.gauss(0.0, sig) for _ in range(n)]
    pts = allan_deviation(y, dt)
    est = sum(s * math.sqrt(t) for t, s in pts[:4]) / 4.0
    return pts, est


def thermal_x_rms(T=T_REF):
    """Equipartition: expected RMS thermal displacement, sqrt(kB*T/k)."""
    return math.sqrt(KB * T / K_EFF)


def simulate_thermal(t_end, T=T_REF, dt=None, seed=1, n_modes=None):
    """Langevin integration (semi-implicit Euler) under thermal force only.
    The modal damping and the thermal forcing are tied by the
    fluctuation-dissipation theorem, so equipartition holds:
    <x^2> -> kB*T/K_EFF.  Returns (dt, x_samples) at the drive point."""
    import random
    nm = n_modes or len(F_MODES)
    if dt is None:
        dt = 1.0 / (40.0 * max(F_MODES[:nm]))
    rng = random.Random(seed)
    sig_f = math.sqrt(2.0 * KB * T * B_TOTAL / dt)
    ud = PORTS[DRIVE_PORT]["u"]
    state = [0.0] * (2 * nm)
    xs = []
    for _ in range(int(t_end / dt)):
        fth = rng.gauss(0.0, sig_f)
        for i in range(nm):
            w = 2.0 * math.pi * F_MODES[i]
            v = state[2 * i + 1] + (
                -w * w * state[2 * i]
                - 2.0 * ZETA[i] * w * state[2 * i + 1]
                + ud[i] * fth) * dt
            state[2 * i + 1] = v
            state[2 * i] += v * dt
        xs.append(sum(ud[i] * state[2 * i] for i in range(nm)))
    return dt, xs


if __name__ == "__main__":
    f_est = estimate_resonance()
    print("ring-down resonance : %.1f Hz  (modal f1 = %.1f Hz)"
          % (f_est, F_MODES[0]))
    H = abs(freq_response(F_MODES[0]))
    print("peak |x_sense/F|    : %.3e m/N  (Q*static)" % H)
    print("static sense defl.  : %.3e m at V_dc" % (
        abs(freq_response(1.0)) * force(DRIVE_PORT, V_DC)))
    print("ARW analytic        : %.4f deg/sqrt(h)"
          % (arw() * (180.0 / math.pi) * 60.0))
    _pts, _est = arw_experiment(n=100000)
    print("ARW from Allan exp. : %.4f deg/sqrt(h)"
          % (_est * (180.0 / math.pi) * 60.0))
    _dt, _xs = simulate_thermal(0.25, n_modes=1, seed=3)
    _rms = math.sqrt(sum(x * x for x in _xs) / len(_xs))
    print("thermal x_rms       : %.2e m (equipartition: %.2e m)"
          % (_rms, thermal_x_rms()))
'''


def _write_python(path, name, freqs, zetas, tlist, drive, sense, v_dc,
                  m_eff, k_eff, Q1, b_total, x_drive):
    ports = {tr.net: {"dcdx": tr.dcdx, "C0": tr.C0, "u": list(tr.u)}
             for tr in tlist}
    arw_si = math.sqrt(4 * 1.380649e-23 * 300.0 * b_total
                       / (2 * m_eff * 2 * math.pi * freqs[0] * x_drive) ** 2
                       / 2.0)
    head = (
        f'"""Reduced-order model of `{name}` — generated by soidlc.\n\n'
        f'Modal superposition of the lowest {len(freqs)} FEM modes; '
        f'comb transducers\nenter via modal participation factors. '
        f'm_eff={m_eff:.3e} kg, k_eff={k_eff:.3e} N/m, Q≈{Q1:.0f}.\n'
        f'"""\n\nimport math\n\n'
        f"F_MODES = {[round(f, 3) for f in freqs]!r}\n"
        f"ZETA = {[round(z, 8) for z in zetas]!r}\n"
        f"PORTS = {ports!r}\n"
        f"DRIVE_PORT = {drive.net!r}\n"
        f"SENSE_PORT = {sense.net!r}\n"
        f"V_DC = {v_dc!r}\n"
        f"M_EFF = {m_eff!r}\nK_EFF = {k_eff!r}\nQ1 = {Q1!r}\n"
        f"B_TOTAL = {b_total!r}     # total gas damping, N*s/m\n"
        f"X_DRIVE = {x_drive!r}     # resonant drive amplitude at V_DC, m\n"
        f"ARW_RADS = {arw_si!r}     # analytic ARW @300K, rad/sqrt(s)\n")
    with open(path, "w") as f:
        f.write(head + _PY_BODY)


def _write_spice(path, name, Rm, Lm, Cm, drive, sense, v_dc, f_series):
    two_port = sense is not drive
    lines = [
        f"* Butterworth-Van Dyke model of `{name}` - generated by soidlc",
        f"* bias V_dc = {v_dc:g} V; series resonance = {f_series:.1f} Hz",
        f"* motional branch: Rm={Rm:.4e} Lm={Lm:.4e} Cm={Cm:.4e}",
        f".subckt {name}_rom drive sense gnd",
        f"Lm drive n1 {Lm:.4e}",
        f"Cm n1   n2 {Cm:.4e}",
        f"Rm n2   {'sense' if two_port else 'gnd'} {Rm:.4e}",
        f"C0d drive gnd {drive.C0:.4e}",
    ]
    if two_port:
        lines.append(f"C0s sense gnd {sense.C0:.4e}")
        # capacitive feedthrough between the ports (rough: 1% of C0)
        lines.append(f"Cft drive sense {0.01 * drive.C0:.4e}")
    lines += [".ends", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _write_veriloga(path, name, freqs, zetas, tlist, drive, sense, v_dc):
    n = len(freqs)
    L = [
        f"// Reduced-order model of `{name}` - generated by soidlc",
        '`include "disciplines.vams"',
        "",
        f"module {name}_rom(d, s, g);",
        "  inout d, s, g;",
        "  electrical d, s, g;",
        "  electrical " + ", ".join(f"q{i}, v{i}"
                                    for i in range(1, n + 1)) + ";",
        f"  parameter real VDC   = {v_dc:g};",
        f"  parameter real DCDXD = {drive.dcdx:.6e};",
        f"  parameter real DCDXS = {sense.dcdx:.6e};",
        f"  parameter real CFT   = {0.01 * drive.C0:.6e};",
    ]
    for i, (f_, z) in enumerate(zip(freqs, zetas), 1):
        L.append(f"  parameter real W{i} = {2 * math.pi * f_:.6e};")
        L.append(f"  parameter real Z{i} = {z:.6e};")
        L.append(f"  parameter real UD{i} = {drive.u[i - 1]:.6e};")
        L.append(f"  parameter real US{i} = {sense.u[i - 1]:.6e};")
    L += [
        "  real F;",
        "  analog begin",
        "    F = 0.5 * DCDXD * V(d, g) * V(d, g);",
    ]
    for i in range(1, n + 1):
        L.append(f"    I(q{i}) <+ ddt(V(q{i})) - V(v{i});")
        L.append(f"    I(v{i}) <+ ddt(V(v{i})) + 2.0*Z{i}*W{i}*V(v{i})"
                 f" + W{i}*W{i}*V(q{i}) - UD{i}*F;")
    vel = " + ".join(f"US{i}*V(v{i})" for i in range(1, n + 1))
    L += [
        f"    I(s, g) <+ VDC * DCDXS * ({vel});",
        "    I(d, s) <+ ddt(CFT * (V(d) - V(s)));",
        "  end",
        "endmodule",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(L))
