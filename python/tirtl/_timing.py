"""Lightweight per-stage wall-clock timing (read by benchmarks/bench.py)."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

ENABLED = True
TIMES: dict = defaultdict(float)


def reset():
    TIMES.clear()


@contextmanager
def stage(name):
    t = time.perf_counter()
    try:
        yield
    finally:
        if ENABLED:
            TIMES[name] += time.perf_counter() - t
