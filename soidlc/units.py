"""Dimensional quantities and unit parsing for SOIDL.

Physical units are part of the type system: ``200 um``, ``20 kHz``,
``0.005 ohm*cm``.  Arithmetic propagates dimensions, and a dimension
mismatch (e.g. ``length + freq``) raises :class:`DimensionError`.

Internally every quantity stores its value in SI base units together with a
dimension vector ``(length, mass, time, current)``.  Geometry code asks for
``.um`` to get a plain float in micrometres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# dimension exponent vector: (length, mass, time, current)
Dim = Tuple[int, int, int, int]

DIMLESS: Dim = (0, 0, 0, 0)
LENGTH: Dim = (1, 0, 0, 0)
MASS: Dim = (0, 1, 0, 0)
TIME: Dim = (0, 0, 1, 0)
CURRENT: Dim = (0, 0, 0, 1)


class DimensionError(Exception):
    """Raised when an operation mixes incompatible physical dimensions."""


def _add(a: Dim, b: Dim) -> Dim:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def _sub(a: Dim, b: Dim) -> Dim:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])


def _scale(a: Dim, n: int) -> Dim:
    return (a[0] * n, a[1] * n, a[2] * n, a[3] * n)


_DIM_NAMES = {
    DIMLESS: "1",
    LENGTH: "length",
    MASS: "mass",
    TIME: "time",
    CURRENT: "current",
    (1, 1, -2, 0): "force",
    (0, 1, -2, 0): "stiffness",
    (-1, 1, -2, 0): "pressure",
    (0, 0, -1, 0): "freq",
    (2, 1, -3, -1): "voltage",
    (-2, -1, 4, 2): "capacitance",
    (2, 1, -3, -2): "resistance",
}


def dim_name(d: Dim) -> str:
    return _DIM_NAMES.get(d, "dim" + str(d))


@dataclass(frozen=True)
class Quantity:
    """A scalar value with a physical dimension (stored in SI base units)."""

    value: float          # SI base value
    dim: Dim = DIMLESS

    # ---- construction helpers -------------------------------------------
    @staticmethod
    def dimensionless(x: float) -> "Quantity":
        return Quantity(float(x), DIMLESS)

    # ---- introspection ---------------------------------------------------
    @property
    def is_dimensionless(self) -> bool:
        return self.dim == DIMLESS

    @property
    def um(self) -> float:
        """Value in micrometres (requires length dimension)."""
        if self.dim != LENGTH:
            raise DimensionError(f"expected length, got {dim_name(self.dim)}")
        return self.value / 1e-6

    def to(self, unit: str) -> float:
        """Return the plain number this quantity represents in ``unit``."""
        factor, d = parse_unit(unit)
        if d != self.dim:
            raise DimensionError(
                f"cannot express {dim_name(self.dim)} in {unit} "
                f"({dim_name(d)})"
            )
        return self.value / factor

    def as_float(self) -> float:
        if self.dim != DIMLESS:
            raise DimensionError(
                f"expected a pure number, got {dim_name(self.dim)}"
            )
        return self.value

    # ---- arithmetic ------------------------------------------------------
    def _unify_dim(self, o: "Quantity") -> Dim:
        if self.dim == o.dim:
            return self.dim
        # a literal zero is compatible with any dimension
        if self.dim == DIMLESS and self.value == 0:
            return o.dim
        if o.dim == DIMLESS and o.value == 0:
            return self.dim
        return None  # type: ignore[return-value]

    def __add__(self, o: "Quantity") -> "Quantity":
        o = _q(o)
        d = self._unify_dim(o)
        if d is None:
            raise DimensionError(
                f"cannot add {dim_name(self.dim)} + {dim_name(o.dim)}"
            )
        return Quantity(self.value + o.value, d)

    def __sub__(self, o: "Quantity") -> "Quantity":
        o = _q(o)
        d = self._unify_dim(o)
        if d is None:
            raise DimensionError(
                f"cannot subtract {dim_name(self.dim)} - {dim_name(o.dim)}"
            )
        return Quantity(self.value - o.value, d)

    def __mul__(self, o: "Quantity") -> "Quantity":
        o = _q(o)
        return Quantity(self.value * o.value, _add(self.dim, o.dim))

    def __truediv__(self, o: "Quantity") -> "Quantity":
        o = _q(o)
        if o.value == 0:
            raise ZeroDivisionError("division by zero in quantity")
        return Quantity(self.value / o.value, _sub(self.dim, o.dim))

    def __neg__(self) -> "Quantity":
        return Quantity(-self.value, self.dim)

    def __pow__(self, n) -> "Quantity":
        n = _q(n)
        if n.dim != DIMLESS:
            raise DimensionError("exponent must be dimensionless")
        e = n.value
        if self.dim != DIMLESS and abs(e - round(e)) > 1e-9:
            raise DimensionError("non-integer power of a dimensioned value")
        return Quantity(self.value ** e, _scale(self.dim, int(round(e))))

    # comparisons (same dimension required)
    def _cmp_ok(self, o: "Quantity") -> "Quantity":
        o = _q(o)
        if self.dim != o.dim:
            raise DimensionError(
                f"cannot compare {dim_name(self.dim)} and {dim_name(o.dim)}"
            )
        return o

    def __lt__(self, o):
        return self.value < self._cmp_ok(o).value

    def __le__(self, o):
        return self.value <= self._cmp_ok(o).value

    def __gt__(self, o):
        return self.value > self._cmp_ok(o).value

    def __ge__(self, o):
        return self.value >= self._cmp_ok(o).value

    def __repr__(self) -> str:
        if self.dim == LENGTH:
            return f"{self.value / 1e-6:g} um"
        if self.dim == DIMLESS:
            return f"{self.value:g}"
        return f"{self.value:g} [{dim_name(self.dim)}]"


def _q(x) -> Quantity:
    if isinstance(x, Quantity):
        return x
    if isinstance(x, (int, float)):
        return Quantity(float(x), DIMLESS)
    raise TypeError(f"not a quantity: {x!r}")


# ---------------------------------------------------------------------------
# Unit table.  Each entry: name -> (SI factor, dimension vector).
# ---------------------------------------------------------------------------
_BASE_UNITS: Dict[str, Tuple[float, Dim]] = {
    # length
    "m": (1.0, LENGTH),
    "cm": (1e-2, LENGTH),
    "mm": (1e-3, LENGTH),
    "um": (1e-6, LENGTH),
    "nm": (1e-9, LENGTH),
    # mass
    "kg": (1.0, MASS),
    "g": (1e-3, MASS),
    # time
    "s": (1.0, TIME),
    "ms": (1e-3, TIME),
    "us": (1e-6, TIME),
    # current
    "A": (1.0, CURRENT),
    # frequency
    "Hz": (1.0, (0, 0, -1, 0)),
    "kHz": (1e3, (0, 0, -1, 0)),
    "MHz": (1e6, (0, 0, -1, 0)),
    "GHz": (1e9, (0, 0, -1, 0)),
    # force
    "N": (1.0, (1, 1, -2, 0)),
    # pressure / modulus
    "Pa": (1.0, (-1, 1, -2, 0)),
    "kPa": (1e3, (-1, 1, -2, 0)),
    "MPa": (1e6, (-1, 1, -2, 0)),
    "GPa": (1e9, (-1, 1, -2, 0)),
    # electrical
    "V": (1.0, (2, 1, -3, -1)),
    "mV": (1e-3, (2, 1, -3, -1)),
    "F": (1.0, (-2, -1, 4, 2)),
    "pF": (1e-12, (-2, -1, 4, 2)),
    "fF": (1e-15, (-2, -1, 4, 2)),
    "ohm": (1.0, (2, 1, -3, -2)),
    # special dimensionless / per-square markers
    "sq": (1.0, DIMLESS),
    "rad": (1.0, DIMLESS),
    # temperature carries no dimension vector of its own in v0.1: kelvin
    # values flow into metric functions (e.g. arw(300 K)) as plain numbers
    "K": (1.0, DIMLESS),
}


def _parse_atom(tok: str) -> Tuple[float, Dim]:
    """Parse ``name`` or ``name^k`` into (factor, dim)."""
    if "^" in tok:
        base, _, exp = tok.partition("^")
        k = int(exp)
    else:
        base, k = tok, 1
    if base not in _BASE_UNITS:
        raise KeyError(base)
    f, d = _BASE_UNITS[base]
    return f ** k, _scale(d, k)


def parse_unit(s: str) -> Tuple[float, Dim]:
    """Parse a compound unit string like ``ohm*cm``, ``kg/m^3``, ``ohm/sq``.

    Returns ``(si_factor, dimension)``.  Raises ``KeyError`` if an atom is
    unknown (used by the lexer to reject non-units).
    """
    s = s.strip()
    factor = 1.0
    dim = DIMLESS
    # split on * and / while remembering the operator
    num, den = [], []
    cur, target = "", num
    for ch in s:
        if ch == "*":
            target.append(cur)
            cur, target = "", num
        elif ch == "/":
            target.append(cur)
            cur, target = "", den
        else:
            cur += ch
    target.append(cur)
    for tok in num:
        if not tok:
            continue
        f, d = _parse_atom(tok)
        factor *= f
        dim = _add(dim, d)
    for tok in den:
        if not tok:
            continue
        f, d = _parse_atom(tok)
        factor /= f
        dim = _sub(dim, d)
    return factor, dim


def make_quantity(value: float, unit: str | None) -> Quantity:
    if unit is None:
        return Quantity(float(value), DIMLESS)
    factor, dim = parse_unit(unit)
    return Quantity(float(value) * factor, dim)


def is_unit(s: str) -> bool:
    try:
        parse_unit(s)
        return True
    except (KeyError, ValueError):
        return False
