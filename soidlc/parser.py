"""Recursive-descent parser: tokens -> SOIDL AST."""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import sast as A
from .lexer import Token, tokenize


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, toks: List[Token]):
        self.toks = toks
        self.pos = 0

    # ---- token helpers ---------------------------------------------------
    @property
    def cur(self) -> Token:
        return self.toks[self.pos]

    def at(self, kind: str, text: Optional[str] = None) -> bool:
        t = self.cur
        return t.kind == kind and (text is None or t.text == text)

    def at_text(self, text: str) -> bool:
        return self.cur.text == text

    def eat(self, kind: str, text: Optional[str] = None) -> Token:
        t = self.cur
        if t.kind != kind or (text is not None and t.text != text):
            want = text or kind
            raise ParseError(
                f"expected {want!r} but found {t.kind} {t.text!r} "
                f"at line {t.line}, col {t.col}"
            )
        self.pos += 1
        return t

    def accept(self, kind: str, text: Optional[str] = None) -> Optional[Token]:
        if self.at(kind, text):
            return self.eat(kind, text)
        return None

    def accept_text(self, text: str) -> bool:
        if self.cur.text == text:
            self.pos += 1
            return True
        return False

    # ---- entry -----------------------------------------------------------
    def parse_file(self) -> A.File:
        decls = []
        while not self.at("EOF"):
            if self.at("KEYWORD", "process"):
                decls.append(self.parse_process())
            elif self.at("KEYWORD", "component"):
                decls.append(self.parse_component())
            elif self.at("KEYWORD", "device"):
                decls.append(self.parse_device())
            elif self.at("KEYWORD", "chip"):
                decls.append(self.parse_chip())
            else:
                t = self.cur
                raise ParseError(
                    f"unexpected {t.kind} {t.text!r} at top level "
                    f"(line {t.line})"
                )
        return A.File(decls)

    # ---- process ---------------------------------------------------------
    def parse_process(self) -> A.Process:
        self.eat("KEYWORD", "process")
        name = self.eat("IDENT").text
        self.eat("PUNCT", "{")
        layers, masks, rules = [], [], []
        while not self.at("PUNCT", "}"):
            if self.accept("KEYWORD", "stack"):
                self.eat("PUNCT", "{")
                while not self.at("PUNCT", "}"):
                    layers.append(self.parse_layer())
                self.eat("PUNCT", "}")
            elif self.accept("KEYWORD", "masks"):
                self.eat("PUNCT", "{")
                while not self.at("PUNCT", "}"):
                    masks.append(self.parse_mask())
                self.eat("PUNCT", "}")
            elif self.accept("KEYWORD", "rules"):
                self.eat("PUNCT", "{")
                while not self.at("PUNCT", "}"):
                    rules.append(self.parse_rule())
                self.eat("PUNCT", "}")
            else:
                t = self.cur
                raise ParseError(
                    f"unexpected {t.text!r} in process (line {t.line})"
                )
        self.eat("PUNCT", "}")
        return A.Process(name, layers, masks, rules)

    def parse_layer(self) -> A.LayerDef:
        self.eat("KEYWORD", "layer")
        name = self._name_text()
        self.eat("PUNCT", "{")
        props = []
        while not self.at("PUNCT", "}"):
            key = self._name_text()
            self.eat("PUNCT", "=")
            val = self.parse_expr()
            props.append((key, val))
            # properties separated by ';' (optional trailing)
            self.accept("PUNCT", ";")
        self.eat("PUNCT", "}")
        return A.LayerDef(name, props)

    def parse_mask(self) -> A.MaskDef:
        self.eat("KEYWORD", "mask")
        name = self._name_text()
        self.eat("PUNCT", "->")
        action = self.parse_expr()
        self.accept("PUNCT", ";")
        return A.MaskDef(name, action)

    def parse_rule(self):
        name = self._name_text()
        if self.at("PUNCT", "{"):
            self.eat("PUNCT", "{")
            sub = []
            while not self.at("PUNCT", "}"):
                sub.append(self.parse_rule())
            self.eat("PUNCT", "}")
            return A.RuleDef(name, sub)
        # name(...) = expr   or   name = expr
        call_arg = None
        if self.at("PUNCT", "("):
            self.eat("PUNCT", "(")
            call_arg = self._name_text()
            self.eat("PUNCT", ")")
        self.eat("PUNCT", "=")
        val = self.parse_expr()
        self.accept("PUNCT", ";")
        rd = A.RuleDef(name, val)
        rd.arg = call_arg  # type: ignore[attr-defined]
        return rd

    # ---- component / device ---------------------------------------------
    def parse_component(self) -> A.Component:
        self.eat("KEYWORD", "component")
        name = self.eat("IDENT").text
        params = []
        if self.accept("PUNCT", "("):
            params = self.parse_params()
            self.eat("PUNCT", ")")
        self.eat("PUNCT", "{")
        items = self.parse_items()
        self.eat("PUNCT", "}")
        return A.Component(name, params, items)

    def parse_device(self) -> A.Device:
        self.eat("KEYWORD", "device")
        name = self.eat("IDENT").text
        self.eat("PUNCT", "{")
        items = self.parse_items()
        self.eat("PUNCT", "}")
        return A.Device(name, items)

    def parse_chip(self):
        # chip is out of geometry scope for v0.1: skip its body
        self.eat("KEYWORD", "chip")
        self.eat("IDENT")
        self._skip_braced_block()
        return A.Device("__chip__", [])

    def parse_params(self) -> List[A.Param]:
        params = []
        if self.at("PUNCT", ")"):
            return params
        while True:
            pname = self._name_text()
            default = None
            if self.accept("PUNCT", "="):
                default = self.parse_expr()
            params.append(A.Param(pname, default))
            if not self.accept("PUNCT", ","):
                break
        return params

    # ---- items (component/device body) ----------------------------------
    def parse_items(self) -> List[object]:
        items = []
        while not self.at("PUNCT", "}"):
            items.append(self.parse_item())
        return items

    def parse_item(self):
        t = self.cur
        if t.kind == "KEYWORD" and t.text in ("mech", "elec"):
            return self.parse_port()
        if self.at("KEYWORD", "param"):
            return self.parse_param_item()
        if self.at("KEYWORD", "derive"):
            return self.parse_derive()
        if self.at("KEYWORD", "geometry"):
            return self.parse_geometry()
        if self.at("KEYWORD", "inst"):
            return self.parse_inst()
        if self.at("KEYWORD", "net"):
            return self.parse_net()
        if self.at("KEYWORD", "isolate"):
            return self.parse_isolate()
        if self.at("KEYWORD", "constraint"):
            return self.parse_constraint()
        if self.at("KEYWORD", "check"):
            return self.parse_check()
        if self.at("KEYWORD", "require"):
            return self.parse_require()
        if self.at("KEYWORD", "solve"):
            return self.parse_solve()
        raise ParseError(
            f"unexpected {t.kind} {t.text!r} in body (line {t.line})"
        )

    def parse_port(self) -> A.Port:
        domain = self.eat("KEYWORD").text
        self.eat("KEYWORD", "port")
        names = [self._name_text()]
        while self.accept("PUNCT", ","):
            names.append(self._name_text())
        self.accept("PUNCT", ";")
        return A.Port(domain, names)

    def parse_param_item(self) -> A.Param:
        self.eat("KEYWORD", "param")
        name = self._name_text()
        default = None
        if self.accept("PUNCT", "="):
            default = self.parse_expr()
        self.accept("PUNCT", ";")
        return A.Param(name, default)

    def parse_derive(self) -> A.Derive:
        self.eat("KEYWORD", "derive")
        target = self.parse_lvalue()
        if self.accept("PUNCT", ">="):
            op = ">="
        elif self.accept("PUNCT", ":"):
            op = ":"
        else:
            self.eat("PUNCT", "=")
            op = "="
        expr = self.parse_expr()
        self.accept("PUNCT", ";")
        return A.Derive(target, op, expr)

    def parse_geometry(self) -> A.Geometry:
        self.eat("KEYWORD", "geometry")
        self.eat("PUNCT", "{")
        body = self.parse_geom_body()
        self.eat("PUNCT", "}")
        return A.Geometry(body)

    def parse_geom_body(self) -> List[object]:
        body = []
        while not self.at("PUNCT", "}"):
            if self.at("KEYWORD", "repeat"):
                body.append(self.parse_repeat())
            else:
                body.append(self.parse_geom_call())
        return body

    def parse_repeat(self) -> A.Repeat:
        self.eat("KEYWORD", "repeat")
        var = self._name_text()
        self.eat("KEYWORD", "in")
        lo = self.parse_additive()
        self.eat("PUNCT", "..")
        hi = self.parse_additive()
        self.eat("PUNCT", "{")
        body = self.parse_geom_body()
        self.eat("PUNCT", "}")
        return A.Repeat(var, A.Range(lo, hi), body)

    def parse_geom_call(self) -> A.GeomCall:
        call = self.parse_postfix()
        placement = None
        if self.accept("KEYWORD", "at"):
            placement = self.parse_placement()
        self.accept("PUNCT", ";")
        if not isinstance(call, A.Call):
            raise ParseError("expected a primitive call in geometry block")
        return A.GeomCall(call, placement)

    def parse_placement(self) -> A.Placement:
        if self.at("PUNCT", "("):
            self.eat("PUNCT", "(")
            x = self.parse_expr()
            self.eat("PUNCT", ",")
            y = self.parse_expr()
            self.eat("PUNCT", ")")
            return A.Placement("at_xy", x=x, y=y)
        # at <port> (possibly dotted)
        name = self._name_text()
        while self.accept("PUNCT", "."):
            name += "." + self._name_text()
        return A.Placement("at_port", port=name)

    def parse_inst(self) -> A.Inst:
        self.eat("KEYWORD", "inst")
        name = self._name_text()
        self.eat("PUNCT", "=")
        call = self.parse_postfix()
        if not isinstance(call, A.Call):
            call = A.Call(call, [], [])
        placement = None
        attach = None
        # optional placement / attach (any order)
        while self.at("KEYWORD", "at") or self.at("KEYWORD", "attach"):
            if self.accept("KEYWORD", "at"):
                placement = self.parse_placement()
            elif self.accept("KEYWORD", "attach"):
                attach = self.parse_attach()
        self.accept("PUNCT", ";")
        return A.Inst(name, call, placement, attach)

    def parse_attach(self) -> List[Tuple[str, object]]:
        self.eat("PUNCT", "(")
        pairs = []
        while not self.at("PUNCT", ")"):
            port = self._name_text()
            self.eat("PUNCT", "->")
            target = self.parse_postfix()
            pairs.append((port, target))
            if not self.accept("PUNCT", ","):
                break
        self.eat("PUNCT", ")")
        return pairs

    def parse_net(self) -> A.Net:
        self.eat("KEYWORD", "net")
        name = self._name_text()
        self.eat("PUNCT", "=")
        expr = self.parse_netexpr()
        self.accept("PUNCT", ";")
        return A.Net(name, expr)

    def parse_netexpr(self):
        # union of nodes with '|'
        left = self.parse_postfix()
        while self.accept("PUNCT", "|"):
            right = self.parse_postfix()
            left = A.Binary("|", left, right)
        return left

    def parse_isolate(self) -> A.Isolate:
        self.eat("KEYWORD", "isolate")
        a = self.parse_netexpr()
        self.eat("KEYWORD", "from")
        b = self.parse_netexpr()
        self.eat("KEYWORD", "by")
        self.eat("KEYWORD", "trench")
        self.accept("PUNCT", ";")
        return A.Isolate(a, b, "trench")

    def parse_constraint(self) -> A.Constraint:
        self.eat("KEYWORD", "constraint")
        call = self.parse_postfix()
        self.accept("PUNCT", ";")
        if not isinstance(call, A.Call):
            call = A.Call(call, [], [])
        return A.Constraint(call)

    def parse_check(self) -> A.Check:
        self.eat("KEYWORD", "check")
        expr = self.parse_expr()
        mode = None
        tail = None
        if self.accept("KEYWORD", "within"):
            mode = "within"
            tail = self.parse_expr()
        elif self.accept("KEYWORD", "warn"):
            mode = "warn"
            if self.at("STRING"):
                tail = self.eat("STRING").text
        elif self.accept("KEYWORD", "report"):
            mode = "report"
        self.accept("PUNCT", ";")
        return A.Check(expr, mode, tail)

    def parse_require(self) -> A.Require:
        self.eat("KEYWORD", "require")
        expr = self.parse_expr()
        self.accept("PUNCT", ";")
        return A.Require(expr)

    def parse_solve(self) -> A.Solve:
        self.eat("KEYWORD", "solve")
        target = self.parse_lvalue()
        self.eat("KEYWORD", "such")
        self.eat("KEYWORD", "that")
        expr = self.parse_expr()
        within = None
        if self.accept("KEYWORD", "within"):
            within = self.parse_expr()
        self.accept("PUNCT", ";")
        return A.Solve(target, expr, within)

    # ---- expressions -----------------------------------------------------
    def parse_lvalue(self) -> str:
        name = self._name_text()
        while self.accept("PUNCT", "."):
            name += "." + self._name_text()
        return name

    def parse_expr(self):
        return self.parse_comparison()

    def parse_comparison(self):
        left = self.parse_additive()
        while self.cur.text in (">=", "<=", "==", "!=", "<", ">"):
            op = self.eat("PUNCT").text
            right = self.parse_additive()
            left = A.Binary(op, left, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.cur.text in ("+", "-"):
            op = self.eat("PUNCT").text
            right = self.parse_multiplicative()
            left = A.Binary(op, left, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_power()
        while self.cur.text in ("*", "/"):
            op = self.eat("PUNCT").text
            right = self.parse_power()
            left = A.Binary(op, left, right)
        return left

    def parse_power(self):
        left = self.parse_unary()
        if self.cur.text == "^":
            self.eat("PUNCT", "^")
            right = self.parse_power()   # right-assoc
            return A.Binary("^", left, right)
        return left

    def parse_unary(self):
        if self.cur.text in ("-", "+"):
            op = self.eat("PUNCT").text
            return A.Unary(op, self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_atom()
        if self.at("PUNCT", "%"):
            self.eat("PUNCT", "%")
            return A.Binary("*", node, A.Num(0.01, None))
        while True:
            if self.at("PUNCT", "."):
                self.eat("PUNCT", ".")
                attr = self._name_text()
                node = A.Member(node, attr)
            elif self.at("PUNCT", "("):
                self.eat("PUNCT", "(")
                args, kwargs = self.parse_call_args()
                self.eat("PUNCT", ")")
                node = A.Call(node, args, kwargs)
            else:
                break
        return node

    def parse_call_args(self):
        args, kwargs = [], []
        if self.at("PUNCT", ")"):
            return args, kwargs
        while True:
            # keyword arg: IDENT '=' expr   (but not '==')
            if (self.cur.kind in ("IDENT", "KEYWORD")
                    and self.toks[self.pos + 1].text == "="):
                key = self._name_text()
                self.eat("PUNCT", "=")
                val = self.parse_expr()
                kwargs.append((key, val))
            else:
                args.append(self.parse_expr())
            if not self.accept("PUNCT", ","):
                break
        return args, kwargs

    def parse_atom(self):
        t = self.cur
        if t.kind == "NUMBER":
            self.pos += 1
            return A.Num(float(t.text), t.unit)
        if t.kind == "STRING":
            self.pos += 1
            return A.Str(t.text)
        if t.text == "?":
            self.pos += 1
            return A.Hole()
        if self.at("PUNCT", "("):
            self.eat("PUNCT", "(")
            e = self.parse_expr()
            self.eat("PUNCT", ")")
            return e
        if t.kind in ("IDENT", "KEYWORD"):
            self.pos += 1
            return A.Name(t.text)
        raise ParseError(
            f"unexpected {t.kind} {t.text!r} in expression (line {t.line})"
        )

    # ---- misc ------------------------------------------------------------
    def _name_text(self) -> str:
        t = self.cur
        if t.kind in ("IDENT", "KEYWORD"):
            self.pos += 1
            return t.text
        raise ParseError(
            f"expected a name but found {t.kind} {t.text!r} (line {t.line})"
        )

    def _skip_braced_block(self):
        self.eat("PUNCT", "{")
        depth = 1
        while depth > 0 and not self.at("EOF"):
            if self.at("PUNCT", "{"):
                depth += 1
            elif self.at("PUNCT", "}"):
                depth -= 1
            self.pos += 1


def parse(src: str) -> A.File:
    return Parser(tokenize(src)).parse_file()
