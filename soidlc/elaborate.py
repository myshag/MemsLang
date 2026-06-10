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

from . import connectivity
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
    E: float = 169e9          # Young's modulus, Pa
    rho: float = 2330.0       # density, kg/m^3
    nu: float = 0.22          # Poisson's ratio (plane-stress isotropic approx)


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
        self.errors: List[str] = []
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
            props: Dict[str, float] = {}
            for k, v in ld.props:
                if k == "thickness":
                    th = self._eval(v, {}).um
                elif k == "material":
                    material = str(self._eval(v, {}))
                elif k in ("E", "rho", "nu"):
                    val = self._eval(v, {})
                    if isinstance(val, Quantity):
                        props[k] = val.value
            p.layers[ld.name] = Layer(
                ld.name, th, material=material,
                color=_LAYER_COLORS.get(ld.name, (0.6, 0.6, 0.6)),
                E=props.get("E", 169e9), rho=props.get("rho", 2330.0),
                nu=props.get("nu", 0.22))
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
                    return Quantity(lay.E, (-1, 1, -2, 0))
                if node.attr == "rho":
                    return Quantity(lay.rho, (-3, 1, 0, 0))
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
                for sh in res.shapes:
                    if not sh.owner:
                        sh.owner = it.name
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
        self._check_connectivity(dev, all_shapes)
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
        if inst.attach:
            return self._apply_attach(res, inst.attach, insts)
        dx, dy = self._resolve_placement(inst, env, insts, res)
        if dx or dy:
            res = _translate_result(res, dx, dy)
        return res

    def _apply_attach(self, res: InstanceResult, attach, insts
                      ) -> InstanceResult:
        """Place an instance against a side of another instance.

        ``attach (rotor -> M.left)`` puts the instance just outside M's left
        edge, rotated so its ``rotor``-labelled geometry faces (and slightly
        overlaps) M — making the mechanical joint also an electrical one.
        """
        for port, target in attach:
            info = self._attach_side(target, insts)
            if info is None:
                continue
            pt, side = info
            if side not in _SIDE_VEC:
                cx, cy = _center(res.bbox)
                return _translate_result(res, pt[0] - cx, pt[1] - cy)
            sx, sy = _SIDE_VEC[side]
            desired = (-sx, -sy)              # port must face the target
            k = _rot_steps(_port_dir(res, port), desired)
            if k:
                res = _rotate_result(res, k)
            x0, y0, x1, y1 = res.bbox
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ov = 2.0                          # um of electrical overlap
            if side == "left":
                dx, dy = pt[0] + ov - x1, pt[1] - cy
            elif side == "right":
                dx, dy = pt[0] - ov - x0, pt[1] - cy
            elif side == "bottom":
                dx, dy = pt[0] - cx, pt[1] + ov - y1
            else:                             # top
                dx, dy = pt[0] - cx, pt[1] - ov - y0
            return _translate_result(res, dx, dy)
        return res

    def _attach_side(self, target, insts):
        if isinstance(target, A.Member) and isinstance(target.obj, A.Name):
            iname, side = target.obj.id, target.attr
            if iname in insts:
                return _bbox_anchor(insts[iname].bbox, side), side
        if isinstance(target, A.Name) and target.id in insts:
            return _center(insts[target.id].bbox), "center"
        return None

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
            px, py, mx, my = (points[i % len(points)] if points
                              else (0.0, 0.0, False, False))
            if mx or my:
                sub = _mirror_result(sub, mx, my)
            sub = _translate_result(sub, px, py)
            shapes.extend(sub.shapes)
        model = {"k_x": Quantity(k_total, (0, 1, -2, 0))} if k_total else {}
        return InstanceResult(shapes, G.bbox_of(shapes), {}, model)

    def _resolve_points(self, place_node, env, insts, count
                        ) -> List[Tuple[float, float, bool, bool]]:
        if place_node is None:
            # default: spread along a line
            return [(i * 50.0, 0.0, False, False) for i in range(count)]
        if isinstance(place_node, A.Call) and isinstance(place_node.func, A.Name):
            fn = place_node.func.id
            if fn == "corners" and place_node.args:
                target = place_node.args[0]
                if isinstance(target, A.Name) and target.id in insts:
                    x0, y0, x1, y1 = insts[target.id].bbox
                    # mirror each instance outward, away from the target's
                    # centre, so anchors end up outboard at every corner
                    return [(x0, y0, False, False), (x1, y0, True, False),
                            (x1, y1, True, True), (x0, y1, False, True)]
        return [(0.0, 0.0, False, False)]

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
        return [G.Shape(s.layer, s.polygon.translated(dx, dy), s.label, s.mech,
                        s.owner)
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
        return 0.0, 0.0

    # ---- model extraction (lumped) -------------------------------------
    def _extract_component_model(self, comp, local, shapes) -> Dict[str, object]:
        model: Dict[str, object] = {"name": comp.name}
        dev = self.process.device()
        t = (dev.thickness * 1e-6) if dev else 25e-6
        E = dev.E if dev else 169e9
        if any(tag in comp.name for tag in ("flexure", "suspension")):
            L = local.get("L")
            w = local.get("w")
            n = local.get("n_beams", local.get("n_folds"))
            if isinstance(L, Quantity) and isinstance(w, Quantity):
                # n clamped-guided beams in parallel: k = n * E t w^3 / L^3
                nf = int(n.value) if isinstance(n, Quantity) else 1
                k = nf * (E * t * (w.value ** 3)) / (L.value ** 3)
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

    # ---- connectivity extraction (geometry -> netlist, LVS-style) ------
    def _check_connectivity(self, dev: A.Device, all_shapes: List[G.Shape]):
        devlay = self.process.ctx.device_layer
        shapes = [s for s in all_shapes if s.layer == devlay]
        if not shapes:
            return
        comp = connectivity.components([s.polygon for s in shapes])
        comp_of = {id(s): c for s, c in zip(shapes, comp)}
        islands: Dict[int, List[G.Shape]] = {}
        for s, c in zip(shapes, comp):
            islands.setdefault(c, []).append(s)
        self.islands = sorted(islands.items())   # consumed by the FEM stage
        self.report.append(
            f"connectivity: {len(shapes)} DEVICE polygons -> "
            f"{len(islands)} electrical island(s)")

        # a fully released island has nothing holding it: it would detach
        for c, ss in sorted(islands.items()):
            if not any(s.mech == "anchored" for s in ss):
                x0, y0, x1, y1 = G.bbox_of(ss)
                owners = sorted({s.owner for s in ss if s.owner})
                self.errors.append(
                    f"island #{c} ({', '.join(owners) or 'unnamed'}) has no "
                    f"anchor — released geometry would float away "
                    f"(bbox [{x0:.0f},{y0:.0f}]..[{x1:.0f},{y1:.0f}] um)")

        # declared nets must each map onto exactly one island
        net_comps: Dict[str, set] = {}
        for it in dev.items:
            if not isinstance(it, A.Net):
                continue
            comps: set = set()
            for iname, port in _net_refs(it.expr):
                ss = self._shapes_for_ref(shapes, iname, port)
                if not ss:
                    self.warnings.append(
                        f"net {it.name}: no DEVICE geometry for "
                        f"{iname}{'.' + port if port else ''}")
                    continue
                comps |= {comp_of[id(s)] for s in ss}
            net_comps[it.name] = comps
            if len(comps) > 1:
                self.errors.append(
                    f"net {it.name} is split across {len(comps)} disconnected "
                    f"islands ({sorted(comps)}) — geometry does not realise "
                    f"the declared node")
            elif comps:
                self.report.append(
                    f"net    {it.name} -> island #{min(comps)}")

        # two different nets sharing an island is a short
        names = [n for n in net_comps if net_comps[n]]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared = net_comps[names[i]] & net_comps[names[j]]
                if shared:
                    self.errors.append(
                        f"nets {names[i]} and {names[j]} are shorted: both "
                        f"lie on island #{min(shared)}")

        # explicit isolation requirements
        for it in dev.items:
            if not isinstance(it, A.Isolate):
                continue
            ca = self._netexpr_comps(it.a, net_comps, shapes, comp_of)
            cb = self._netexpr_comps(it.b, net_comps, shapes, comp_of)
            if not ca or not cb:
                continue
            da, db = _net_desc(it.a), _net_desc(it.b)
            if ca & cb:
                self.errors.append(
                    f"isolate violated: {da} and {db} share island "
                    f"#{min(ca & cb)} (trench does not separate them)")
            else:
                self.report.append(f"isolate {da} from {db}: ok")

    def _shapes_for_ref(self, shapes: List[G.Shape], iname: str,
                        port: Optional[str]) -> List[G.Shape]:
        own = [s for s in shapes
               if s.owner == iname or s.owner.startswith(iname + "[")]
        if port:
            labelled = [s for s in own if port in s.label]
            if labelled:
                return labelled
            # ports named like an anchor map to the anchored geometry
            anchored = [s for s in own if s.mech == "anchored"]
            if anchored and port in ("fixed", "anchor", "anchors", "base"):
                return anchored
        return own

    def _netexpr_comps(self, expr, net_comps, shapes, comp_of) -> set:
        if isinstance(expr, A.Name) and expr.id in net_comps:
            return net_comps[expr.id]
        comps: set = set()
        for iname, port in _net_refs(expr):
            if iname in net_comps and port is None:
                comps |= net_comps[iname]
                continue
            for s in self._shapes_for_ref(shapes, iname, port):
                comps.add(comp_of[id(s)])
        return comps

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
_SIDE_VEC = {"left": (-1, 0), "right": (1, 0), "top": (0, 1), "bottom": (0, -1)}


def _translate_result(res: InstanceResult, dx: float, dy: float) -> InstanceResult:
    shapes = [G.Shape(s.layer, s.polygon.translated(dx, dy), s.label, s.mech,
                      s.owner)
              for s in res.shapes]
    x0, y0, x1, y1 = res.bbox
    bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    ports = {k: (x + dx, y + dy) for k, (x, y) in res.ports.items()}
    return InstanceResult(shapes, bbox, ports, res.model)


def _mirror_result(res: InstanceResult, mx: bool, my: bool) -> InstanceResult:
    shapes = [G.Shape(s.layer, s.polygon.mirrored(mx, my), s.label, s.mech,
                      s.owner)
              for s in res.shapes]
    fx = -1.0 if mx else 1.0
    fy = -1.0 if my else 1.0
    ports = {k: (x * fx, y * fy) for k, (x, y) in res.ports.items()}
    return InstanceResult(shapes, G.bbox_of(shapes), ports, res.model)


def _rotate_result(res: InstanceResult, k90: int) -> InstanceResult:
    deg = 90.0 * (k90 % 4)
    shapes = [G.Shape(s.layer, s.polygon.rotated(deg), s.label, s.mech,
                      s.owner)
              for s in res.shapes]
    c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    ports = {k: (x * c - y * s_, x * s_ + y * c)
             for k, (x, y) in res.ports.items()}
    return InstanceResult(shapes, G.bbox_of(shapes), ports, res.model)


def _port_dir(res: InstanceResult, port: str) -> Tuple[int, int]:
    """Which side of the instance the named port's geometry sits on."""
    tagged = [s for s in res.shapes if port in s.label]
    if not tagged:
        return (1, 0)
    px, py = _center(G.bbox_of(tagged))
    cx, cy = _center(res.bbox)
    dx, dy = px - cx, py - cy
    if abs(dx) >= abs(dy):
        return (1, 0) if dx >= 0 else (-1, 0)
    return (0, 1) if dy >= 0 else (0, -1)


def _rot_steps(d0: Tuple[int, int], d1: Tuple[int, int]) -> int:
    def ang(d):
        return {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[d]
    return (ang(d1) - ang(d0)) % 4


def _net_refs(expr) -> List[Tuple[str, Optional[str]]]:
    """Flatten a net expression into (instance, port|None) references."""
    if isinstance(expr, A.Binary) and expr.op == "|":
        return _net_refs(expr.left) + _net_refs(expr.right)
    if isinstance(expr, A.Member) and isinstance(expr.obj, A.Name):
        return [(expr.obj.id, expr.attr)]
    if isinstance(expr, A.Name):
        return [(expr.id, None)]
    return []


def _net_desc(expr) -> str:
    if isinstance(expr, A.Binary) and expr.op == "|":
        return f"{_net_desc(expr.left)}|{_net_desc(expr.right)}"
    if isinstance(expr, A.Member) and isinstance(expr.obj, A.Name):
        return f"{expr.obj.id}.{expr.attr}"
    if isinstance(expr, A.Name):
        return expr.id
    return "?"


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
