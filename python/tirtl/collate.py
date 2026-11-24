"""R string ordering (ICU "root" collation, R's default when built with ICU).

Used to reproduce list.files() order and the tie-break of names(sort(-table(x)))[1].
"""
from __future__ import annotations

from ._native import core

_PUNCT = " _-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$"
_PRIM = {}
for _i, _c in enumerate(_PUNCT):
    _PRIM[_c] = (0, _i)
for _d in "0123456789":
    _PRIM[_d] = (1, ord(_d) - 48)
for _l in "abcdefghijklmnopqrstuvwxyz":
    _PRIM[_l] = (2, ord(_l) - 97)
    _PRIM[_l.upper()] = (2, ord(_l) - 97)


def _prim(c):
    p = _PRIM.get(c)
    if p is not None:
        return p
    return (0, 100 + ord(c)) if ord(c) < 128 else (3, ord(c))


def r_sort_key(s: str):
    return (tuple(_prim(c) for c in s), tuple(1 if c.isascii() and c.isupper() else 0 for c in s), s)


def r_order(xs):
    """Stable ordering permutation of `xs` under R collation (like R's order())."""
    c = core()
    xs = [str(x) for x in xs]
    if c is not None:
        return list(c.r_order(xs))
    return sorted(range(len(xs)), key=lambda i: r_sort_key(xs[i]))


def r_sorted(xs):
    xs = list(xs)
    return [xs[i] for i in r_order(xs)]
