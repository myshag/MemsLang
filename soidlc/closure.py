"""Design closure: solve the device's free parameters against its spec.

Every ``solve <target> such that <equation> [within tol]`` declares a free
design variable; every ``require <inequality>`` adds a hard constraint.  The
declared parameter values are only *initial guesses*: this stage re-elaborates
the device (quietly — no reports, no DRC) with candidate values, evaluates
the equations/requirements through the metric layer, and minimises the total
violation with Nelder–Mead.  The solved values then drive the final, loud
elaboration — so the geometry that gets meshed, checked and exported is the
one that meets the spec.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import metrics
from . import sast as A
from .units import Quantity

MAX_EVALS = 160
MULT_LO, MULT_HI = 0.2, 5.0


def _expr_str(e) -> str:
    if isinstance(e, A.Binary):
        return f"{_expr_str(e.left)} {e.op} {_expr_str(e.right)}"
    if isinstance(e, A.Unary):
        return f"{e.op}{_expr_str(e.operand)}"
    if isinstance(e, A.Call):
        return f"{_expr_str(e.func)}({', '.join(_expr_str(a) for a in e.args)})"
    if isinstance(e, A.Member):
        return f"{_expr_str(e.obj)}.{e.attr}"
    if isinstance(e, A.Name):
        return e.id
    if isinstance(e, A.Num):
        return f"{e.value:g}{' ' + e.unit if e.unit else ''}"
    return "?"


def _as_value(v) -> float:
    if isinstance(v, Quantity):
        return v.value
    if isinstance(v, (int, float)):
        return float(v)
    raise metrics.MetricError(f"non-numeric value {v!r}")


def _residual(elab, expr, env) -> float:
    """Relative residual of an equation / signed violation of an inequality."""
    if not isinstance(expr, A.Binary) or expr.op not in (
            "==", ">=", "<=", ">", "<"):
        v = elab._eval(expr, env)
        return 0.0 if v is True else 1.0
    l = _as_value(elab._eval(expr.left, env))
    r = _as_value(elab._eval(expr.right, env))
    scale = max(abs(r), abs(l), 1e-30)
    if expr.op == "==":
        return (l - r) / scale
    if expr.op in (">=", ">"):
        return max(0.0, (r - l) / scale)
    return max(0.0, (l - r) / scale)        # <= / <


def _nelder_mead(f, x0: List[float], step: float = 0.18,
                 max_evals: int = MAX_EVALS, ftol: float = 1e-9):
    n = len(x0)
    pts = [x0[:]]
    for i in range(n):
        p = x0[:]
        p[i] += step
        pts.append(p)
    evals = [0]

    def fe(p):
        evals[0] += 1
        return f(p)

    vals = [fe(p) for p in pts]
    while evals[0] < max_evals:
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[-1] - vals[0] < ftol:
            break
        cen = [sum(p[i] for p in pts[:-1]) / n for i in range(n)]
        refl = [cen[i] + (cen[i] - pts[-1][i]) for i in range(n)]
        fr = fe(refl)
        if fr < vals[0]:
            exp = [cen[i] + 2 * (cen[i] - pts[-1][i]) for i in range(n)]
            fx = fe(exp)
            pts[-1], vals[-1] = (exp, fx) if fx < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            con = [cen[i] + 0.5 * (pts[-1][i] - cen[i]) for i in range(n)]
            fc = fe(con)
            if fc < vals[-1]:
                pts[-1], vals[-1] = con, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [(pts[i][j] + pts[0][j]) / 2 for j in range(n)]
                    vals[i] = fe(pts[i])
    best = min(range(n + 1), key=lambda i: vals[i])
    return pts[best], vals[best], evals[0]


def run(elab, device: Optional[str] = None,
        fem_calibrate: bool = False, fem_h: float = 20.0
        ) -> Dict[str, Quantity]:
    """Solve the device's free parameters.  Returns the overrides dict.

    With ``fem_calibrate`` an outer loop measures the systematic
    lumped-vs-FEM frequency bias on each closed design, folds it into the
    ``f_res`` metric as a calibration factor, and re-solves — so the spec is
    met by the FEM-predicted frequency, not the lumped estimate.
    """
    name = elab.resolve_device_name(device)
    dev = elab.devices[name]
    solves = [it for it in dev.items if isinstance(it, A.Solve)]
    requires = [it for it in dev.items if isinstance(it, A.Require)]
    if not solves:
        return {}
    elab.metric_calibration = {}

    # baseline run records the declared values of the solve targets
    mark = (len(elab.report), len(elab.warnings), len(elab.errors))

    def _truncate():
        del elab.report[mark[0]:]
        del elab.warnings[mark[1]:]
        del elab.errors[mark[2]:]

    elab.elaborate_device(name, quiet=True)
    _truncate()
    targets, x0 = [], []
    for s in solves:
        q = elab.observed_params.get(s.target)
        if not isinstance(q, Quantity) or q.value == 0:
            elab.warnings.append(
                f"closure: no observable initial value for {s.target}; "
                f"skipped")
            continue
        targets.append(s)
        x0.append(q)
    if not targets:
        return {}

    def overrides_for(mults: List[float]) -> Dict[str, Quantity]:
        out = {}
        for s, q, m in zip(targets, x0, mults):
            m = min(MULT_HI, max(MULT_LO, m))
            out[s.target] = Quantity(q.value * m, q.dim)
        return out

    # tolerance per equation: a 1%-tolerance equation weighs more in the
    # penalty than a 5% one, so NM spends its budget where the spec is tight
    def _tol_of(s) -> float:
        if s.within is None:
            return 0.02
        try:
            t = elab._eval(s.within, {})
            return t.value if isinstance(t, Quantity) else float(t)
        except Exception:
            return 0.02
    tols = [_tol_of(s) for s in targets]

    def penalty(mults: List[float]) -> float:
        ov = overrides_for(mults)
        try:
            res = elab.elaborate_device(name, overrides=ov, quiet=True)
            env = metrics.build_env(elab, res)
            p = 0.0
            for s, tol in zip(targets, tols):
                p += (_residual(elab, s.expr, env) / tol) ** 2
            for rq in requires:
                p += (_residual(elab, rq.expr, env) / 0.02) ** 2
            return p
        except Exception:
            return 1e6
        finally:
            _truncate()

    def solve_once(start: Optional[List[float]] = None):
        start = start or [1.0] * len(targets)
        if penalty(start) >= 1e6:
            return None, None, 0.0, 0
        best, v, ev = _nelder_mead(penalty, start)
        return overrides_for(best), best, v, ev

    def fem_measure(ov) -> Optional[Tuple[float, float]]:
        """(f_fem_mode1, f_lumped) for a candidate design, or None.

        Meshes only the suspended island (same domain as the verification
        stage) so the calibration factor transfers exactly.
        """
        from . import connectivity, fem
        try:
            res = elab.elaborate_device(name, overrides=ov, quiet=True)
            f0 = res.model.get("f0")
            if not isinstance(f0, Quantity):
                return None
            shapes = [s for s in res.shapes if s.layer == "DEVICE"]
            comp = connectivity.components([s.polygon for s in shapes])
            groups: Dict[int, list] = {}
            for s, c in zip(shapes, comp):
                groups.setdefault(c, []).append(s)
            suspended = [ss for ss in groups.values()
                         if any(s.mech == "anchored" for s in ss)
                         and any(s.mech == "released" for s in ss)]
            if not suspended:
                return None
            island = max(suspended,
                         key=lambda ss: sum(s.polygon.area() for s in ss
                                            if s.mech == "released"))
            mesh = fem.build_mesh(island, fem_h)
            if not mesh.cells or mesh.n_cells > fem.MAX_ELEMENTS:
                return None
            d = elab.process.device()
            freqs, _vecs, _dof = fem.modal(mesh, d.E, d.nu, d.rho,
                                           d.thickness * 1e-6, n_modes=1)
            return (freqs[0], f0.value) if freqs else None
        except Exception:
            return None
        finally:
            _truncate()

    overrides, best_mults, val, n_evals = solve_once()
    if overrides is None:
        elab.warnings.append(
            "closure: spec metrics could not be evaluated; "
            "solve targets left at their declared values")
        return {}

    cal_lines: List[str] = []
    if fem_calibrate:
        factor_prev = 1.0
        for outer in range(1, 4):
            meas = fem_measure(overrides)
            if meas is None:
                cal_lines.append(
                    "closure: FEM calibration unavailable for this device")
                break
            f_fem, f_lump = meas
            factor = f_fem / f_lump
            cal_lines.append(
                f"closure: FEM calibration pass {outer}: "
                f"f_fem={f_fem / 1e3:.2f} kHz vs lumped "
                f"{f_lump / 1e3:.2f} kHz -> factor {factor:.4f} "
                f"(h = {fem_h:g} um)")
            if abs(factor - factor_prev) < 0.003:
                break
            elab.metric_calibration = {"f_res": factor}
            factor_prev = factor
            o2, m2, val, ev2 = solve_once(start=best_mults)
            if o2 is None:
                break
            overrides, best_mults = o2, m2
            n_evals += ev2

    # report the solution with per-equation residuals
    res = elab.elaborate_device(name, overrides=overrides, quiet=True)
    env = metrics.build_env(elab, res)
    residuals = [_residual(elab, s.expr, env) for s in targets]
    _truncate()
    elab.report.extend(cal_lines)
    for s, r in zip(targets, residuals):
        tol = None
        if s.within is not None:
            try:
                t = elab._eval(s.within, {})
                tol = t.value if isinstance(t, Quantity) else float(t)
            except Exception:
                tol = None
        q = overrides[s.target]
        status = "ok" if (tol is None or abs(r) <= tol) \
            else f"OUT OF TOLERANCE (allowed {tol:.1%})"
        elab.report.append(
            f"closure: solved {s.target} = {q!r}  "
            f"[{_expr_str(s.expr)}; residual {r:+.2%}, {status}]")
    elab.report.append(
        f"closure: {n_evals} design evaluations, total penalty {val:.3e}")
    return overrides


def enforce_requires(elab, art, device: Optional[str] = None) -> None:
    """Evaluate `require` statements against the final geometry: a violated
    requirement is a compile error, like a connectivity mismatch."""
    name = elab.resolve_device_name(device)
    dev = elab.devices[name]
    requires = [it for it in dev.items if isinstance(it, A.Require)]
    if not requires:
        return
    try:
        env = metrics.build_env(elab, art.result)
    except Exception as e:  # noqa: BLE001
        art.warnings.append(f"require: metrics unavailable ({e})")
        return
    for rq in requires:
        txt = _expr_str(rq.expr)
        try:
            lhs = None
            if isinstance(rq.expr, A.Binary):
                lhs = elab._eval(rq.expr.left, env)
            v = elab._eval(rq.expr, env)
        except metrics.MetricError as e:
            art.warnings.append(f"require {txt}: not evaluable ({e})")
            continue
        detail = f" (actual: {lhs!r})" if isinstance(lhs, Quantity) else ""
        if v is True:
            art.report.append(f"require {txt}: ok{detail}")
        elif v is False:
            art.errors.append(f"require violated: {txt}{detail}")
        else:
            art.warnings.append(f"require {txt}: not evaluable")
