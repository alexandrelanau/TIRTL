"""Access to the compiled Rust kernels (tirtl._core), with a switch to force the pure-Python path.

Set TIRTL_NO_RUST=1 to disable the Rust extension (pure-Python fallback, mainly for testing).
"""
from __future__ import annotations

import os

try:  # pragma: no cover - depends on the build
    from . import _core as _core_mod
except ImportError:  # pragma: no cover
    _core_mod = None


def core():
    """The Rust extension module, or None when unavailable/disabled."""
    if os.environ.get("TIRTL_NO_RUST", "") == "1":
        return None
    return _core_mod


def have_rust() -> bool:
    return core() is not None
