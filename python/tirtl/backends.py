"""Array backends for the dense float32 products: numpy (CPU), cupy (NVIDIA), mlx (Apple).

Each backend only has to provide float32 matrix products and a per-row top-k selection; exactness
is restored on the host (see pairing.py), so every backend yields the same pairs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ._native import core


def _ensure_cuda_path():
    # conda's cuda headers live in $PREFIX/targets/<arch>/include; cupy needs CUDA_PATH for NVRTC
    if "CUDA_PATH" not in os.environ:
        for arch in ("x86_64-linux", "sbsa-linux", "aarch64-linux"):
            cand = Path(sys.prefix) / "targets" / arch
            if (cand / "include" / "cuda_fp16.h").exists():
                os.environ["CUDA_PATH"] = str(cand)
                break


_ensure_cuda_path()  # before any cupy import


class Backend:
    name = "numpy"
    chunk_rows = 2048

    def to_dev(self, x):
        return x

    def to_host(self, x):
        return np.asarray(x)

    def matmul(self, a, b_t):
        return a @ b_t

    def topk(self, s, k):
        """(idx, values) of the k largest per row of the device matrix s, on the host."""
        c = core()
        s = self.to_host(s)
        if c is not None:
            return c.topk_rows_f32(np.ascontiguousarray(s, dtype=np.float32), int(k))
        k = min(k, s.shape[1])
        key = np.where(np.isnan(s), -np.inf, s)
        part = np.argpartition(-key, k - 1, axis=1)[:, :k] if k < s.shape[1] else np.tile(np.arange(s.shape[1]), (s.shape[0], 1))
        vals = np.take_along_axis(s, part, axis=1)
        return part.astype(np.int64), vals.astype(np.float32)

    def free(self):
        pass


class CupyBackend(Backend):
    name = "cupy"
    chunk_rows = 4096

    def __init__(self):
        _ensure_cuda_path()
        import cupy as cp
        self.cp = cp
        os.environ.setdefault("CUPY_TF32", "0")  # full float32 products (error bound assumes fp32)

    def to_dev(self, x):
        return self.cp.asarray(x)

    def to_host(self, x):
        return self.cp.asnumpy(x)

    def topk(self, s, k):
        cp = self.cp
        k = min(k, s.shape[1])
        key = cp.where(cp.isnan(s), -cp.inf, s)
        if k < s.shape[1]:
            part = cp.argpartition(-key, k - 1, axis=1)[:, :k]
        else:
            part = cp.broadcast_to(cp.arange(s.shape[1]), s.shape)
        vals = cp.take_along_axis(s, part, axis=1)
        return cp.asnumpy(part).astype(np.int64), cp.asnumpy(vals).astype(np.float32)

    def free(self):
        self.cp.get_default_memory_pool().free_all_blocks()


class MlxBackend(Backend):
    name = "mlx"
    chunk_rows = 4096

    def __init__(self):
        import mlx.core as mx
        self.mx = mx

    def to_dev(self, x):
        return self.mx.array(x)

    def to_host(self, x):
        return np.array(x)

    def matmul(self, a, b_t):
        return self.mx.matmul(a, b_t)

    def topk(self, s, k):
        mx = self.mx
        k = min(k, s.shape[1])
        key = mx.where(mx.isnan(s), -mx.inf, s)
        if k < s.shape[1]:
            part = mx.argpartition(-key, k - 1, axis=1)[:, :k]
        else:
            part = mx.broadcast_to(mx.arange(s.shape[1]), s.shape)
        vals = mx.take_along_axis(s, part, axis=1)
        return np.array(part).astype(np.int64), np.array(vals).astype(np.float32)


def get_backend(name, strict=False):
    """Backend by name. When cupy / mlx cannot be used, fall back to numpy with a warning (results
    are identical: exactness is restored on the host) unless strict=True."""
    if name in (None, "numpy", "cpu"):
        return Backend()
    if name not in ("cupy", "mlx"):
        raise ValueError("backend must be 'numpy', 'cupy', or 'mlx'")
    try:
        be = CupyBackend() if name == "cupy" else MlxBackend()
        s = be.matmul(be.to_dev(np.ones((4, 4), np.float32)), be.to_dev(np.ones((4, 4), np.float32)))
        be.topk(s, 2)  # also compiles device kernels
        return be
    except Exception as e:  # noqa: BLE001
        if strict:
            raise
        import warnings
        warnings.warn(f"backend {name!r} unavailable ({type(e).__name__}: {e}); using numpy (same results)")
        return Backend()
