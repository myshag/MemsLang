"""Tokenizer for SOIDL.

Notable rule: a numeric literal may be immediately followed (on the same
line) by a *unit* — e.g. ``200 um``, ``2330 kg/m^3``, ``0.06 ohm/sq``.  The
lexer greedily tries to attach a unit and silently rewinds if the trailing
word is not a recognised unit (so ``2 * n_folds`` stays a bare number).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .units import is_unit

KEYWORDS = {
    "process", "component", "device", "chip",
    "stack", "layer", "masks", "mask", "rules", "release", "drie",
    "geometry", "repeat", "in", "at", "attach",
    "mech", "elec", "port", "param", "derive", "inst", "net",
    "constraint", "isolate", "from", "by", "trench", "check",
    "solve", "such", "that", "within", "warn", "report",
    "deposit", "etch", "through", "backside", "exposed", "where", "auto",
}

# multi-char punctuation first
_PUNCT = [
    "->", "..", ">=", "<=", "==", "!=",
    "{", "}", "(", ")", "[", "]",
    "=", ",", ";", ".", "+", "-", "*", "/", "^", "<", ">", "|", "?", ":", "%",
]


@dataclass
class Token:
    kind: str          # NUMBER, IDENT, KEYWORD, PUNCT, STRING, EOF
    text: str
    line: int
    col: int
    unit: Optional[str] = None  # for NUMBER tokens carrying a unit

    def __repr__(self) -> str:
        u = f"<{self.unit}>" if self.unit else ""
        return f"{self.kind}:{self.text}{u}"


class LexError(Exception):
    pass


def tokenize(src: str) -> List[Token]:
    toks: List[Token] = []
    i, n = 0, len(src)
    line, col = 1, 1

    def adv(k: int = 1):
        nonlocal i, line, col
        for _ in range(k):
            if i < n and src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        c = src[i]
        # whitespace
        if c in " \t\r\n":
            adv()
            continue
        # comments
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                adv()
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            adv(2)
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                adv()
            adv(2)
            continue
        # string
        if c == '"':
            start_line, start_col = line, col
            adv()
            buf = ""
            while i < n and src[i] != '"':
                buf += src[i]
                adv()
            adv()  # closing quote
            toks.append(Token("STRING", buf, start_line, start_col))
            continue
        # number (optionally followed by a unit)
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            start_line, start_col = line, col
            num = ""
            while i < n and (src[i].isdigit() or src[i] in ".eE+-"):
                # stop at the range operator ".."
                if src[i] == "." and i + 1 < n and src[i + 1] == ".":
                    break
                # only allow +/- right after e/E (exponent)
                if src[i] in "+-" and not (num and num[-1] in "eE"):
                    break
                num += src[i]
                adv()
            unit = _try_unit(src, i, n)
            if unit is not None:
                # skip inline spaces then the unit text
                j = i
                while j < n and src[j] in " \t":
                    j += 1
                adv(j - i + len(unit))
            toks.append(Token("NUMBER", num, start_line, start_col, unit=unit))
            continue
        # identifier / keyword
        if c.isalpha() or c == "_":
            start_line, start_col = line, col
            buf = ""
            while i < n and (src[i].isalnum() or src[i] in "_"):
                buf += src[i]
                adv()
            kind = "KEYWORD" if buf in KEYWORDS else "IDENT"
            toks.append(Token(kind, buf, start_line, start_col))
            continue
        # punctuation
        matched = False
        for p in _PUNCT:
            if src.startswith(p, i):
                toks.append(Token("PUNCT", p, line, col))
                adv(len(p))
                matched = True
                break
        if matched:
            continue
        raise LexError(f"unexpected character {c!r} at line {line}, col {col}")

    toks.append(Token("EOF", "", line, col))
    return toks


def _try_unit(src: str, i: int, n: int) -> Optional[str]:
    """Peek past inline spaces for a unit word; return it or None."""
    j = i
    while j < n and src[j] in " \t":
        j += 1
    if j >= n or not (src[j].isalpha()):
        return None
    k = j
    while k < n and (src[k].isalnum() or src[k] in "*/^"):
        k += 1
    word = src[j:k]
    # a trailing '*' or '/' would have been grabbed greedily by a following
    # expression; trim operators that are not part of a real unit
    while word and word[-1] in "*/^":
        word = word[:-1]
    if word and is_unit(word):
        return word
    return None
