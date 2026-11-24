"""TIRTL-seq paired TCR analysis.

Pairing pipeline faithful to the original R implementation (pogorely/TIRTL), with Rust kernels
(tirtl._core) and CPU (numpy) / NVIDIA (cupy) / Apple (mlx) backends.
"""
from __future__ import annotations

from ._native import have_rust
from .collate import r_order, r_sorted
from .madhyper import (
    derivative_prob_function,
    estimate_match_frequencies,
    estimate_nonmatch_frequencies,
    estimate_pair_prob,
    estimate_pair_prob_batch,
    estimate_probs_vec,
    find_freq,
    get_input,
    get_input_mat,
    madhyper_surface,
    multinom_coeff,
    well_freq,
    wij_adjustment,
)
from .pipeline import get_well_subset, list_files_r, run_single_point_analysis_sub_gpu
from .run_pairing import get_tshell_settings, run_pairing
from .diversity import calculate_diversity, get_all_div_metrics
from .legacy import (  # small helpers kept from the previous Python port
    add_VJ_aa,
    big_merge_freqs2,
    combineTCR,
    geta,
    geta_sm,
    getb,
    getb_sm,
    get_good_wells_sub,
    get_most_popularV,
    na_to0,
    read_gpu,
    read_gpu_corr,
    write_dat,
)
from .multirun import (  # faithful to the original R (multi-run analyses)
    add_sign,
    concordance,
    get_clonotypes_10x,
    merge_TIRTL,
    process_10x,
    run_longitudinal_analysis_sub,
)
from .tools import (  # TIRTLtools data processing
    TIRTLseqDataSet,
    add_metadata,
    clean_pairs,
    combine_TIRTLseqData,
    create_thimble_df,
    read_external_bulk,
    write_stitchr_tsv,
    filter_dataset,
    filter_duplicate_tcrs,
    filter_mait,
    filter_nonfunctional_TCRs,
    filter_short_cdr3s,
    get_all_tcrs,
    get_expanded_clones,
    get_pair_stats,
    get_paired_by_read_fraction_range,
    get_samples,
    load_tirtlseq,
    remove_duplicates,
    reorder_samples,
    summarize_data,
)

__version__ = "0.2.0"
