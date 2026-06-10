"""Abstract syntax tree node definitions for SOIDL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---- expressions ----------------------------------------------------------
@dataclass
class Num:
    value: float
    unit: Optional[str] = None


@dataclass
class Str:
    value: str


@dataclass
class Name:
    id: str


@dataclass
class Member:        # a.b.c
    obj: "Expr"
    attr: str


@dataclass
class Call:
    func: "Expr"
    args: List["Expr"]
    kwargs: List[Tuple[str, "Expr"]] = field(default_factory=list)


@dataclass
class Unary:
    op: str
    operand: "Expr"


@dataclass
class Binary:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass
class Range:
    lo: "Expr"
    hi: "Expr"


@dataclass
class Hole:          # the '?' placeholder used by solve
    pass


Expr = object


# ---- statements / items ---------------------------------------------------
@dataclass
class LayerDef:
    name: str
    props: List[Tuple[str, "Expr"]]


@dataclass
class MaskDef:
    name: str
    action: "Expr"


@dataclass
class RuleDef:
    name: str
    value: object       # Expr or nested dict of rules


@dataclass
class Process:
    name: str
    layers: List[LayerDef]
    masks: List[MaskDef]
    rules: List[RuleDef]


@dataclass
class Port:
    domain: str         # mech | elec
    names: List[str]


@dataclass
class Param:
    name: str
    default: Optional["Expr"]


@dataclass
class Derive:
    target: str
    op: str             # '=' or '>=' (contract)
    expr: "Expr"


@dataclass
class Placement:
    kind: str           # 'at_xy' | 'at_port'
    x: Optional["Expr"] = None
    y: Optional["Expr"] = None
    port: Optional[str] = None


@dataclass
class GeomCall:         # a primitive or component call inside geometry{}
    call: Call
    placement: Optional[Placement] = None


@dataclass
class Repeat:
    var: str
    rng: Range
    body: List[object]  # list of GeomCall | Repeat


@dataclass
class Geometry:
    body: List[object]


@dataclass
class Inst:
    name: str
    call: Call
    placement: Optional[Placement] = None
    attach: Optional[List[Tuple[str, "Expr"]]] = None   # (port -> target)


@dataclass
class Net:
    name: str
    expr: "Expr"


@dataclass
class Isolate:
    a: "Expr"
    b: "Expr"
    by: str


@dataclass
class Constraint:
    call: Call


@dataclass
class Check:
    expr: "Expr"
    mode: Optional[str] = None   # within | warn | report
    tail: object = None


@dataclass
class Solve:
    target: str
    expr: "Expr"
    within: Optional["Expr"] = None


@dataclass
class Component:
    name: str
    params: List[Param]
    items: List[object]


@dataclass
class Device:
    name: str
    items: List[object]


@dataclass
class File:
    decls: List[object]
