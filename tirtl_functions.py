"""Compatibility module: the flat `tirtl_functions` API, now backed by the `tirtl` package.

    import tirtl_functions as tf
    tf.run_single_point_analysis_sub_gpu("data/", backend="cupy")

Install the package once (`pip install -e .` or `maturin develop --release`) to get the Rust
kernels; without them a pure-Python fallback (same results, slower) is used.
The module-level `madhyper_surface` and `estimate_pair_prob_batch` may be monkeypatched (e.g. to
force a cells-per-well value): run_single_point_analysis_sub_gpu looks them up at call time.
"""
from __future__ import annotations

import os
import sys

try:
    import tirtl as _tirtl  # noqa: F401
except ImportError:  # repository checkout without installation
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
    import tirtl as _tirtl  # noqa: F401

from tirtl.legacy import *  # noqa: F401,F403,E402
from tirtl.madhyper import estimate_pair_prob, estimate_pair_prob_batch, madhyper_surface  # noqa: F401,E402
from tirtl.pipeline import get_well_subset, list_files_r  # noqa: F401,E402
from tirtl.pipeline import run_single_point_analysis_sub_gpu as _run_single_point  # noqa: E402


def run_single_point_analysis_sub_gpu(folder_path, prefix="tmp", well_filter_thres=0.5, min_reads=0,
                                      min_wells=2, well_pos=3, wellset1=None, compute=True, backend="numpy",
                                      pval_thres_tshell=1e-10, wij_thres_tshell=2, pseudobulk=False, **kwargs):
    mod = sys.modules[__name__]
    kwargs.setdefault("surface_fn", lambda **k: mod.madhyper_surface(**k))
    kwargs.setdefault("score_fn", lambda **k: mod.estimate_pair_prob_batch(**k))
    return _run_single_point(folder_path, prefix=prefix, well_filter_thres=well_filter_thres,
                             min_reads=min_reads, min_wells=min_wells, well_pos=well_pos, wellset1=wellset1,
                             compute=compute, backend=backend, pval_thres_tshell=pval_thres_tshell,
                             wij_thres_tshell=wij_thres_tshell, pseudobulk=pseudobulk, **kwargs)
