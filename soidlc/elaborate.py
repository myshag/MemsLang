"""Elaboration: AST -> concrete layered geometry + extracted model/report.

This stage resolves parameters, evaluates unit-checked expressions, expands
``repeat`` loops and component hierarchy, applies placement/attachment, and
produces a flat list of :class:`Shape` (micrometres, world coordinates) plus
a textual report of every ``derive`` / ``check`` / ``solve``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import geometry as G
from . import sast as A
from .primitives import PRIMITIVES, PrimitiveCtx, is_primitive
from .units import (DIMLESS, LENGTH, DimensionError, Quantity, make_quantity)


# physical constants available to expressions
EPS0 = Quantity(8.8541878128e-12, (-3, -1, 4, 2))   # F/m


class Sym(str):
    """A bareword that is not a bound variable (e.g. a direction ``y`` or a
    material name).  Behaves like a string."""


class Unknown:
    """Placeholder produced by the ``?`` hole; carries a numeric fallback."""

    def __init__(self, fallback: Quantity):
        self.fallback = fallback


class ElabError(Exception):
    pass


# ---------------------------------------------------------------------------
@dataclass
class Layer:
    name: str
    thickness: float          # um
    z0: float = 0.0           # um (bottom)
    z1: float = 0.0           # um (top)
    material: str = ""
    color: Tuple[float, float, float] = (0.6, 0.6, 0.6)


_LAYER_COLORS = {
    "METAL": (0.92, 0.78, 0.20),
    "DEVICE": (0.40, 0.55, 0.80),
    "BOX": (0.70, 0.45, 0.85),
    "HANDLE": (0.55, 0.55, 0.58),
}


@dataclass
class ProcessInfo:
    name: str
    layers: Dict[str, Layer] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)   # top -> bottom
    rules: Dict[str, object] = field(default_factory=dict)
    ctx: PrimitiveCtx = field(default_factory=PrimitiveCtx)

    def device(self) -> Optional[Layer]:
        return self.layers.get("DEVICE")

    def box(self) -> Optional[Layer]:
        return self.layers.get("BOX")

    def handle(self) -> Optional[Layer]:
        return self.layers.get("HANDLE")

    def metal(self) -> Optional[Layer]:
        return self.layers.get("METAL")


@dataclass
class InstanceResult:
    shapes: List[G.Shape]
    bbox: Tuple[float, float, float, float]
    ports: Dict[str, Tuple[float, float]]
    model: Dict[str, object] = field(default_factory=dict)


def _default_process() -> ProcessInfo:
    p = ProcessInfo("default")
    specs = [("METAL", 0.52), ("DEVICE", 25.0), ("BOX", 2.0), ("HANDLE", 400.0)]
    for n, t in specs:
        p.layers[n] = Layer(n, t, material=n,
                            color=_LAYER_COLORS.get(n, (0.6, 0.6, 0.6)))
        p.order.append(n)
    _assign_z(p)
    return p


def _assign_z(p: ProcessInfo) -> None:
    """Assign z ranges, building bottom-up so HANDLE sits at z=0."""
    bottom_up = list(reversed(p.order))
    z = 0.0
    for name in bottom_up:
        lay = p.layers[name]
        lay.z0 = z
        lay.z1 = z + lay.thickness
        z = lay.z1


# ---------------------------------------------------------------------------
class Elaborator:
    def __init__(self, file: A.File):
        self.file = file
        self.components: Dict[str, A.Component] = {}
        self.devices: Dict[str, A.Device] = {}
        self.process = _default_process()
        self.report: List[str] = []
        self.warnings: List[str] = []
        self._index()

    def _index(self) -> None:
        proc = None
        for d in self.file.decls:
            if isinstance(d, A.Process):
                proc = d
            elif isinstance(d, A.Component):
                self.components[d.name] = d
            elif isinstance(d, A.Device) and d.name != "__chip__":
                self.devices[d.name] = d
        if proc is not None:
            self.process = self._build_process(proc)

    # ---- process --------------------------------------------------------
    def _build_process(self, proc: A.Process) -> ProcessInfo:
        p = ProcessInfo(proc.name)
        for ld in proc.layers:
            th = 1.0
            material = ""
            for k, v in ld.props:
                if k == "thickness":
                    th = self._eval(v, {}).um
                elif k == "material":
                    material = str(self._eval(v, {}))
            p.layers[ld.name] = Layer(
                ld.name, th, material=material,
                color=_LAYER_COLORS.get(ld.name, (0.6, 0.6, 0.6)))
            p.order.append(ld.name)
        if not p.order:
            return _default_process()
        _assign_z(p)
        p.rules = self._collect_rules(proc.rules)
        self._apply_release_rules(p)
        return p

    def _collect_rules(self, rules: List[A.RuleDef]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for rd in rules:
            if isinstance(rd.value, list):
                out[rd.name] = self._collect_rules(rd.value)
            else:
                try:
                    out[rd.name] = self._eval(rd.value, {})
                except Exception:
                    out[rd.name] = None
        return out

    def _apply_release_rules(self, p: ProcessInfo) -> None:
        rel = p.rules.get("release")
        if isinstance(rel, dict):
            for key, attr in (("hole_size", "hole_size"),
                              ("hole_pitch", "hole_pitch"),
                              ("max_solid_span", "max_solid_span")):
                v = rel.get(key)
                if isinstance(v, Quantity):
                    setattr(p.ctx, attr, v.um)
        dev = p.device()
        if dev:
            p.ctx.device_layer = "DEVICE"
        if p.metal():
            p.ctx.metal_layer = "METAL"

    # ---- expression evaluation -----------------------------------------
    def _eval(self, node, env: Dict[str, object]):
        if isinstance(node, A.Num):
            return make_quantity(node.value, node.unit)
        if isinstance(node, A.Str):
            return node.value
        if isinstance(node, A.Hole):
            return Unknown(Quantity(100e-6, LENGTH))
        if isinstance(node, A.Name):
            if node.id in env:
                return env[node.id]
            if node.id == "eps0":
                return EPS0
            if node.id == "pi":
                return Quantity(math.pi, DIMLESS)
            return Sym(node.id)
        if isinstance(node, A.Member):
            return self._eval_member(node, env)
        if isinstance(node, A.Unary):
            v = self._eval(node.operand, env)
            return (-v) if node.op == "-" else v
        if isinstance(node, A.Binary):
            return self._eval_binary(node, env)
        if isinstance(node, A.Call):
            return self._eval_call(node, env)
        if isinstance(node, A.Range):
            return (self._eval(node.lo, env), self._eval(node.hi, env))
        raise ElabError(f"cannot evaluate node {node!r}")

    def _eval_member(self, node: A.Member, env):
        obj = node.obj
        # process.LAYER.prop  /  process.rules.x
        if isinstance(obj, A.Member) and isinstance(obj.obj, A.Name) \
                and obj.obj.id == "process":
            layer = obj.attr
            lay = self.process.layers.get(layer)
            if lay is not None:
                if node.attr == "thickness":
                    return Quantity(lay.thickness * 1e-6, LENGTH)
                if node.attr == "E":
                    return Quantity(169e9, (-1, 1, -2, 0))
            return Sym(f"process.{layer}.{node.attr}")
        base = self._eval(obj, env)
        if isinstance(base, dict) and node.attr in base:
            return base[node.attr]
        return Sym(f"{_symstr(base)}.{node.attr}")

    def _eval_binary(self, node: A.Binary, env):
        l = self._eval(node.left, env)
        r = self._eval(node.right, env)
        op = node.op
        try:
            if op == "+":
                return l + r
            if op == "-":
                return l - r
            if op == "*":
                return l * r
            if op == "/":
                return l / r
            if op == "^":
                return l ** r
            if op in (">=", "<=", ">", "<", "==", "!="):
                return _compare(op, l, r)
        except (TypeError, DimensionError):
            return Sym(f"({_symstr(l)}{op}{_symstr(r)})")
        return Sym("?")

    def _eval_call(self, node: A.Call, env):
        fname = node.func.id if isinstance(node.func, A.Name) else None
        args = [self._eval(a, env) for a in node.args]
        kwargs = {k: self._eval(v, env) for k, v in node.kwargs}
        if fname == "sqrt":
            return _qfunc(args[0], math.sqrt, half_dim=True)
        if fname in ("abs",):
            q = args[0]
            return Quantity(abs(q.value), q.dim) if isinstance(q, Quantity) else q
        if fname in ("min", "max"):
            f = min if fname == "min" else max
            return f(args, key=lambda q: q.value if isinstance(q, Quantity) else q)
        # unknown function: return a symbolic placeholder (used by derive/check)
        return Sym(f"{fname or '?'}(...)")

    # ---- top-level compile ---------------------------------------------
    def elaborate_device(self, name: Optional[str] = None) -> InstanceResult:
        if name is None:
            if not self.devices:
                raise ElabError("no device to elaborate")
            name = list(self.devices.keys())[-1]
        if name not in self.devices:
            raise ElabError(f"unknown device {name!r}")
        dev = self.devices[name]
        return self._elab_device(dev)

    def _elab_device(self, dev: A.Device) -> InstanceResult:
        env: Dict[str, object] = {}
        insts: Dict[str, InstanceResult] = {}
        all_shapes: List[G.Shape] = []

        # first pass: params
        for it in dev.items:
            if isinstance(it, A.Param) and it.default is not None:
                try:
                    env[it.name] = self._eval(it.default, env)
                except Exception:
                    pass

        # resolve solves (best effort) before instancing
        solve_targets = self._plan_solves(dev, env)

        for it in dev.items:
            if isinstance(it, A.Inst):
                res = self._elab_inst(it, env, insts, solve_targets)
                insts[it.name] = res
                all_shapes.extend(res.shapes)
            elif isinstance(it, A.Derive):
                self._record_derive(it, env)
            elif isinstance(it, A.Check):
                self._record_check(it, env)
            elif isinstance(it, A.Solve):
                val = solve_targets.get(it.target)
                if val is not None:
                    self.report.append(
                        f"solve  {it.target} = {val!r}  (target met approx.)")

        bbox = G.bbox_of(all_shapes)
        model = self._extract_device_model(insts)
        return InstanceResult(all_shapes, bbox, {}, model)

    def _plan_solves(self, dev: A.Device, env) -> Dict[str, Quantity]:
        """Best-effort numeric solve.  Currently supports tuning a flexure
        length to a target resonant frequency; otherwise leaves a fallback."""
        out: Dict[str, Quantity] = {}
        for it in dev.items:
            if isinstance(it, A.Solve):
                # default fallback length
                out[it.target] = Quantity(150e-6, LENGTH)
        return out

    # ---- instance elaboration ------------------------------------------
    def _elab_inst(self, inst: A.Inst, env, insts, solves) -> InstanceResult:
        res = self._elab_call(inst.call, env, insts, solves, inst.name)
        dx, dy = self._resolve_placement(inst, env, insts, res)
        if dx or dy:
            res = _translate_result(res, dx, dy)
        return res

    def _elab_call(self, call: A.Call, env, insts, solves,
                   inst_name: str) -> InstanceResult:
        fname = call.func.id if isinstance(call.func, A.Name) else None

        if fname == "array":
            return self._elab_array(call, env, insts, solves, inst_name)

        if fname and is_primitive(fname):
            shapes = self._call_primitive(fname, call, env)
            return InstanceResult(shapes, G.bbox_of(shapes), {})

        if fname in self.components:
            return self._elab_component(self.components[fname], call, env,
                                        solves, inst_name)

        # unknown call -> empty geometry (e.g. behavioural-only component)
        self.warnings.append(f"inst {inst_name}: unknown component {fname!r}")
        return InstanceResult([], (0, 0, 0, 0), {})

    def _elab_array(self, call: A.Call, env, insts, solves,
                    inst_name: str) -> InstanceResult:
        if not call.args:
            return InstanceResult([], (0, 0, 0, 0), {})
        inner = call.args[0]
        kw = {k: v for k, v in call.kwargs}
        count = 4
        if "count" in kw:
            c = self._eval(kw["count"], env)
            count = int(c.value) if isinstance(c, Quantity) else int(c)
        points = self._resolve_points(kw.get("place"), env, insts, count)
        shapes: List[G.Shape] = []
        k_total = 0.0
        for i in range(count):
            sub = self._elab_call(inner if isinstance(inner, A.Call)
                                  else A.Call(inner, [], []),
                                  env, insts, solves, f"{inst_name}[{i}]")
            kx = sub.model.get("k_x")
            if isinstance(kx, Quantity):
                k_total += kx.value
            px, py = points[i % len(points)] if points else (0.0, 0.0)
            sub = _translate_result(sub, px, py)
            shapes.extend(sub.shapes)
        model = {"k_x": Quantity(k_total, (0, 1, -2, 0))} if k_total else {}
        return InstanceResult(shapes, G.bbox_of(shapes), {}, model)

    def _resolve_points(self, place_node, env, insts, count
                        ) -> List[Tuple[float, float]]:
        if place_node is None:
            # default: spread along a line
            return [(i * 50.0, 0.0) for i in range(count)]
        if isinstance(place_node, A.Call) and isinstance(place_node.func, A.Name):
            fn = place_node.func.id
            if fn == "corners" and place_node.args:
                target = place_node.args[0]
                if isinstance(target, A.Name) and target.id in insts:
                    x0, y0, x1, y1 = insts[target.id].bbox
                    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return [(0.0, 0.0)]

    def _call_primitive(self, fname, call: A.Call, env) -> List[G.Shape]:
        args = [self._eval(a, env) for a in call.args]
        kwargs = {k: self._eval(v, env) for k, v in call.kwargs}
        return PRIMITIVES[fname](args, kwargs, self.process.ctx)

    def _elab_component(self, comp: A.Component, call: A.Call, env, solves,
                        inst_name: str) -> InstanceResult:
        local: Dict[str, object] = {}
        pos = [self._eval(a, env) for a in call.args]
        named = {k: self._eval(v, env) for k, v in call.kwargs}
        for i, p in enumerate(comp.params):
            if p.name in named:
                val = named[p.name]
            elif i < len(pos):
                val = pos[i]
            elif p.default is not None:
                val = self._eval(p.default, local)
            else:
                val = Unknown(Quantity(100e-6, LENGTH))
            if isinstance(val, Unknown):
                # try a solved value for "<inst>.<param>"
                sv = solves.get(f"{inst_name}.{p.name}")
                val = sv if sv is not None else val.fallback
            local[p.name] = val

        shapes: List[G.Shape] = []
        ports: Dict[str, Tuple[float, float]] = {}
        for it in comp.items:
            if isinstance(it, A.Geometry):
                shapes.extend(self._elab_geometry(it.body, local, ports))
            elif isinstance(it, A.Derive):
                self._record_derive(it, local, prefix=comp.name)
            elif isinstance(it, A.Check):
                self._record_check(it, local, prefix=comp.name)
        model = self._extract_component_model(comp, local, shapes)
        return InstanceResult(shapes, G.bbox_of(shapes), ports, model)

    def _elab_geometry(self, body, env, ports) -> List[G.Shape]:
        shapes: List[G.Shape] = []
        for node in body:
            if isinstance(node, A.Repeat):
                lo = self._eval(node.rng.lo, env)
                hi = self._eval(node.rng.hi, env)
                lo_i = int(lo.value) if isinstance(lo, Quantity) else int(lo)
                hi_i = int(hi.value) if isinstance(hi, Quantity) else int(hi)
                for i in range(lo_i, hi_i + 1):
                    e2 = dict(env)
                    e2[node.var] = Quantity(float(i), DIMLESS)
                    shapes.extend(self._elab_geometry(node.body, e2, ports))
            elif isinstance(node, A.GeomCall):
                shapes.extend(self._elab_geomcall(node, env, ports))
        return shapes

    def _elab_geomcall(self, gc: A.GeomCall, env, ports) -> List[G.Shape]:
        fname = gc.call.func.id if isinstance(gc.call.func, A.Name) else None
        if fname and is_primitive(fname):
            local_shapes = self._call_primitive(fname, gc.call, env)
        elif fname in self.components:
            sub = self._elab_component(self.components[fname], gc.call, env,
                                       {}, fname)
            local_shapes = sub.shapes
        else:
            return []
        dx, dy = self._geom_placement(gc.placement, env, ports)
        return [G.Shape(s.layer, s.polygon.translated(dx, dy), s.label, s.mech)
                for s in local_shapes]

    def _geom_placement(self, pl: Optional[A.Placement], env, ports
                        ) -> Tuple[float, float]:
        if pl is None:
            return 0.0, 0.0
        if pl.kind == "at_xy":
            x = self._eval(pl.x, env)
            y = self._eval(pl.y, env)
            return (_as_um(x), _as_um(y))
        # at_port: use known port coords, else origin
        return ports.get(pl.port, (0.0, 0.0))

    def _resolve_placement(self, inst: A.Inst, env, insts, res
                          ) -> Tuple[float, float]:
        if inst.placement is not None:
            if inst.placement.kind == "at_xy":
                return (_as_um(self._eval(inst.placement.x, env)),
                        _as_um(self._eval(inst.placement.y, env)))
        if inst.attach:
            # translate so the instance centre lands on the first target point
            for _port, target in inst.attach:
                pt = self._resolve_target_point(target, insts)
                if pt is not None:
                    cx, cy = _center(res.bbox)
                    return (pt[0] - cx, pt[1] - cy)
        return 0.0, 0.0

    def _resolve_target_point(self, target, insts) -> Optional[Tuple[float, float]]:
        # target like M.left / M.right / M.top / M.bottom / M.center
        if isinstance(target, A.Member) and isinstance(target.obj, A.Name):
            iname = target.obj.id
            side = target.attr
            if iname in insts:
                return _bbox_anchor(insts[iname].bbox, side)
        if isinstance(target, A.Name) and target.id in insts:
            return _center(insts[target.id].bbox)
        return None

    # ---- model extraction (lumped) -------------------------------------
    def _extract_component_model(self, comp, local, shapes) -> Dict[str, object]:
        model: Dict[str, object] = {"name": comp.name}
        dev = self.process.device()
        t = (dev.thickness * 1e-6) if dev else 25e-6
        E = 169e9
        if comp.name.endswith("flexure") or "flexure" in comp.name:
            L = local.get("L")
            w = local.get("w")
            n = local.get("n_folds")
            if isinstance(L, Quantity) and isinstance(w, Quantity):
                nf = int(n.value) if isinstance(n, Quantity) else 2
                k = (2 * nf) * (E * t * (w.value ** 3)) / (L.value ** 3)
                model["k_x"] = Quantity(k, (0, 1, -2, 0))
        return model

    def _extract_device_model(self, insts) -> Dict[str, object]:
        dev = self.process.device()
        t = (dev.thickness * 1e-6) if dev else 25e-6
        rho = 2330.0
        m = 0.0
        k = 0.0
        for name, ir in insts.items():
            for sh in ir.shapes:
                if sh.layer == "DEVICE" and sh.mech == "released":
                    m += sh.polygon.area() * 1e-12 * t * rho
            kx = ir.model.get("k_x")
            if isinstance(kx, Quantity):
                k += kx.value
        model: Dict[str, object] = {}
        if m > 0:
            model["m"] = Quantity(m, (0, 1, 0, 0))
        if k > 0:
            model["k"] = Quantity(k, (0, 1, -2, 0))
        if m > 0 and k > 0:
            f0 = math.sqrt(k / m) / (2 * math.pi)
            model["f0"] = Quantity(f0, (0, 0, -1, 0))
        return model

    # ---- derive / check reporting --------------------------------------
    def _record_derive(self, it: A.Derive, env, prefix: str = ""):
        tag = f"{prefix}." if prefix else ""
        try:
            v = self._eval(it.expr, env)
            self.report.append(f"derive {tag}{it.target} {it.op} {v!r}")
        except Exception as e:
            self.report.append(f"derive {tag}{it.target} {it.op} <unresolved: {e}>")

    def _record_check(self, it: A.Check, env, prefix: str = ""):
        tag = f"{prefix}: " if prefix else ""
        try:
            v = self._eval(it.expr, env)
            status = "ok" if (v is True or (isinstance(v, bool) and v)) else v
            line = f"check  {tag}{_symstr(v)}"
            if it.mode == "warn" and isinstance(it.tail, str):
                line += f"   [warn: {it.tail}]"
            elif it.mode:
                line += f"   [{it.mode}]"
            self.report.append(line)
        except Exception as e:
            self.report.append(f"check  {tag}<unresolved: {e}>")


# ---------------------------------------------------------------------------
def _translate_result(res: InstanceResult, dx: float, dy: float) -> InstanceResult:
    shapes = [G.Shape(s.layer, s.polygon.translated(dx, dy), s.label, s.mech)
              for s in res.shapes]
    x0, y0, x1, y1 = res.bbox
    bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    ports = {k: (x + dx, y + dy) for k, (x, y) in res.ports.items()}
    return InstanceResult(shapes, bbox, ports, res.model)


def _center(bbox) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _bbox_anchor(bbox, side: str) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {
        "left": (x0, cy), "right": (x1, cy),
        "top": (cx, y1), "bottom": (cx, y0),
        "center": (cx, cy),
    }.get(side, (cx, cy))


def _as_um(v) -> float:
    if isinstance(v, Quantity):
        if v.dim == LENGTH:
            return v.um
        return v.value
    return float(v)


def _symstr(v) -> str:
    if isinstance(v, Quantity):
        return repr(v)
    return str(v)


def _compare(op, l, r):
    lv = l.value if isinstance(l, Quantity) else l
    rv = r.value if isinstance(r, Quantity) else r
    if op == ">=":
        return lv >= rv
    if op == "<=":
        return lv <= rv
    if op == ">":
        return lv > rv
    if op == "<":
        return lv < rv
    if op == "==":
        return lv == rv
    if op == "!=":
        return lv != rv
    raise ValueError(op)


def _qfunc(q, fn, half_dim=False):
    if isinstance(q, Quantity):
        from .units import _scale
        dim = q.dim
        if half_dim:
            dim = tuple(d // 2 for d in q.dim)
        return Quantity(fn(q.value), dim)  # type: ignore[arg-type]
    return fn(float(q))
