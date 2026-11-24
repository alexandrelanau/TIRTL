"""Speed of tirtl vs the original R code on public data (Zenodo 10.5281/zenodo.14010377, exp3).

    python docs/make_benchmark_plot.py   # docs/benchmark_public.json -> docs/benchmark_public.png

Timings (seconds, best of 3 for tirtl) were measured on a Xeon w7-3465X (28 cores). The R
references are the original TIRTL code (pogorely/TIRTL @ 6d6528d, numpy backend) and TIRTLtools
0.2.9 (backend "cpp").
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = {
    "pipe ID03": "run_single_point_analysis_sub_gpu (TIRTL)\nID03, 192 wells",
    "run_pairing ID03": "run_pairing (TIRTLtools)\nID03, 192 wells",
    "run_longitudinal_analysis_sub": "run_longitudinal_analysis_sub (TIRTL)\nID02 vs ID03",
    "process_10x": "process_10x (TIRTL)\n10x exp3 tp2",
    "load_tirtlseq": "load_tirtlseq (TIRTLtools)\nID02 + ID03",
    "clean_pairs": "clean_pairs (TIRTLtools)",
    "get_expanded_clones": "get_expanded_clones (TIRTLtools)\nID02 vs ID03",
    "calculate_diversity": "calculate_diversity (TIRTLtools)\nbeta, 14 metrics",
}


def fmt(s):
    return f"{s * 1000:.0f} ms" if s < 1 else (f"{s:.1f} s" if s < 10 else (f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min"))


def main():
    t = json.loads((HERE / "benchmark_public.json").read_text())
    keys = [k for k in LABELS if k in t]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(keys) + 1.8), dpi=150)
    for a in (fig, ax):
        a.set_facecolor("#fcfcfb")
    h = 0.36
    ys = np.arange(len(keys))[::-1]
    for y, k in zip(ys, keys):
        r, p = t[k]
        ax.barh(y + h / 2 + 0.02, r, height=h, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.5,
                label="R (original TIRTL / TIRTLtools)" if y == ys[0] else None)
        ax.barh(y - h / 2 - 0.02, p, height=h, color="#1baf7a", edgecolor="#fcfcfb", linewidth=1.5,
                label="tirtl (Python + Rust)" if y == ys[0] else None)
        ax.text(r * 1.1, y + h / 2 + 0.02, fmt(r), va="center", fontsize=8, color="#0b0b0b")
        ax.text(p * 1.1, y - h / 2 - 0.02, fmt(p), va="center", fontsize=8, color="#0b0b0b")
        ax.text(1.0, y, f"×{r / p:,.0f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#0b0b0b",
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(0.03, 12000)
    ax.set_yticks(ys, [LABELS[k] for k in keys])
    ax.set_xlabel("Wall-clock time (s, log scale) — lower is better", color="#52514e")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    ax.set_title("tirtl vs R on public TIRTL-seq data (Zenodo exp3)\nsame pairs / tables as the R code; speed-up on the right",
                 loc="left", fontsize=11.5, color="#0b0b0b", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(HERE / "benchmark_public.png", facecolor=fig.get_facecolor())
    print(HERE / "benchmark_public.png")


if __name__ == "__main__":
    main()
