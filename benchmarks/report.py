"""Turn results/runs.jsonl into results/summary.csv, results/REPORT.md and figures.

    python report.py

Works on a partial sweep: cells that have not run are simply absent.
"""
import json
import os

import numpy as np
import pandas as pd

import config
import data

RESULTS_DIR = data.RESULTS_DIR
ORDER = ["tsne-h2pack", "FIt-SNE", "openTSNE", "UMAP"]
# Validated categorical palette (colorblind-safe in this fixed order); color follows the library
COLORS = {"tsne-h2pack": "#2a78d6", "FIt-SNE": "#eb6834", "openTSNE": "#1baf7a", "UMAP": "#eda100"}
MARKERS = {"tsne-h2pack": "o", "FIt-SNE": "s", "openTSNE": "^", "UMAP": "D"}
# Labels used in the figures. The method is written h2-tsne (the name used in the paper);
# tsne-h2pack is the distribution name and stays the key everywhere in the data.
LABEL = {m: m for m in ORDER}
LABEL["tsne-h2pack"] = "h2-tsne"


def load():
    rows = [json.loads(line) for line in open(os.path.join(RESULTS_DIR, "runs.jsonl"))]
    df = pd.DataFrame([r for r in rows if "error" not in r and r["dataset"] == config.DATASET])
    key = lambda r: (r["method"], r["n"], r["threads"], r["rep"], r.get("k"))
    succeeded = {key(r) for r in rows if "error" not in r}
    failed = [r for r in rows if "error" in r and key(r) not in succeeded]   # not fixed by a later retry
    return df, failed


def fmt_n(n):
    return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n // 1000}K"


def mean_std(values, digits=1):
    values = [v for v in values if v is not None and not pd.isna(v)]
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_summary(df):
    """One row per (method, n, threads): means over reps, plus the shared k-NN times."""
    knn = df[df.method == "knn"].groupby(["n", "threads", "k"]).t_knn.mean()
    native = df[df.method == "FIt-SNE-native-knn"].groupby(["n", "threads"]).t_setup.mean()
    runs = df[df.method.isin(ORDER)]
    rows = []
    for (method, n, threads), g in runs.groupby(["method", "n", "threads"]):
        k = config.K_UMAP if method == "UMAP" else config.K_TSNE
        t_knn = knn.get((n, threads, k), np.nan)
        t_setup = g.t_setup.astype(float).mean()
        t_opt = g.t_optimize.mean()
        if method == "FIt-SNE":
            # its own front end (Annoy k-NN + P); it cannot consume the shared graph end to end
            front, t_total = native.get((n, threads), np.nan), native.get((n, threads), np.nan) + t_opt
        else:
            front, t_total = t_knn + t_setup, t_knn + t_setup + t_opt
        rows.append(dict(method=method, n=n, threads=threads, reps=len(g), t_knn_shared=t_knn, t_setup=t_setup,
                         t_front_end=front, t_optimize=t_opt,
                         t_optimize_std=g.t_optimize.std(ddof=1) if len(g) > 1 else np.nan,
                         t_total=t_total, kl=g.kl.mean(), knn_preservation=g.knn_preservation.mean(),
                         span=g.span.mean(), peak_rss_gb=g.peak_rss_gb.mean()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e3e6ea", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#9aa3ad", "axes.labelcolor": "#2b3138",
                         "xtick.color": "#4a525b", "ytick.color": "#4a525b"})
    threads = sorted(summary.threads.unique())

    # 1. optimization time vs N, one panel per thread count, shared y axis
    # (a 2 x 2 grid once there are four thread counts, so the panels stay readable)
    ncols = 2 if len(threads) == 4 else len(threads)
    nrows = -(-len(threads) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), sharey=True, squeeze=False)
    ends = []
    for ax, t in zip(axes.flat, threads):
        tips = []
        for m in ORDER:
            s = summary[(summary.method == m) & (summary.threads == t)].sort_values("n")
            if s.empty:
                continue
            ax.plot(s.n, s.t_optimize, color=COLORS[m], marker=MARKERS[m], markersize=6, linewidth=2,
                    label=LABEL[m])
            tips.append([np.log10(s.t_optimize.iloc[-1]), s.n.iloc[-1], LABEL[m]])
        ends.append((ax, tips))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(sorted(summary.n.unique())); ax.set_xticklabels([fmt_n(n) for n in sorted(summary.n.unique())])
        ax.minorticks_off()
        ax.set_xlim(right=max(summary.n) * 2.6)
        ax.set_title(f"{t} thread{'s' if t > 1 else ''}", loc="left", fontsize=11)
        ax.set_xlabel("points")
    for row in axes:
        row[0].set_ylabel("optimization time (s), shared k-NN graph")
    axes[0][0].legend(frameon=False, fontsize=9, loc="upper left")
    # End-of-line labels, pushed apart when two series finish at the same height (the y axis is
    # shared and logarithmic, so a fixed separation in decades is resolution independent).
    lo, hi = axes[0][0].get_ylim()
    min_sep = 0.055 * (np.log10(hi) - np.log10(lo))
    for ax, tips in ends:
        tips.sort()
        for i in range(1, len(tips)):
            if tips[i][0] - tips[i - 1][0] < min_sep:
                tips[i][0] = tips[i - 1][0] + min_sep
        for logy, x, lab in tips:
            ax.annotate(lab, (x, 10 ** logy), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color="#2b3138")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_optimization_time.png"), dpi=160)
    plt.close(fig)

    # 2. end-to-end breakdown, stacked bars: front end (k-NN + affinities) and optimization
    sizes = sorted(summary.n.unique())
    fig, axes = plt.subplots(len(threads), 1, figsize=(1.9 * len(sizes) + 3, 3.6 * len(threads)), squeeze=False)
    width = 0.19
    for ax, t in zip(axes[:, 0], threads):
        for j, m in enumerate(ORDER):
            s = summary[(summary.method == m) & (summary.threads == t)].set_index("n").reindex(sizes)
            xs = np.arange(len(sizes)) + (j - 1.5) * (width + 0.02)
            ax.bar(xs, s.t_front_end, width, color=COLORS[m], alpha=0.45, edgecolor="white", linewidth=1)
            ax.bar(xs, s.t_optimize, width, bottom=s.t_front_end, color=COLORS[m], edgecolor="white", linewidth=1,
                   label=LABEL[m])
        ax.set_xticks(np.arange(len(sizes))); ax.set_xticklabels([fmt_n(n) for n in sizes])
        ax.set_yscale("log")
        ax.set_ylabel("end-to-end time (s)")
        ax.set_title(f"{t} thread{'s' if t > 1 else ''}: light = k-NN + affinities, solid = optimization",
                     loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0][0].legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_time_breakdown.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- report
def report(df, summary, failed):
    sizes, threads = sorted(summary.n.unique()), sorted(summary.threads.unique())
    runs = df[df.method.isin(ORDER)]
    dim_note = ("" if config.DIM == 2 else
                f" **{config.DIM}D embeddings**: FIt-SNE and openTSNE have no FFT gradient beyond 2D and run their "
                f"Barnes-Hut gradient here; tsne-h2pack and UMAP run unchanged with n_components={config.DIM}.")
    L = [f"# t-SNE benchmark ({config.DIM}D): tsne-h2pack vs FIt-SNE vs openTSNE vs UMAP\n",
         f"Dataset `{config.DATASET}`, {config.DIM}D, perplexity {config.PERPLEXITY}, {config.N_ITER} iterations "
         f"({config.EARLY_EXAGGERATION_ITER} at exaggeration {config.EARLY_EXAGGERATION}), each library's own "
         f"'auto' learning-rate rule (see README: N/12 for tsne-h2pack and FIt-SNE; openTSNE N/12 then N). Times in seconds, mean ± sample std over reps. "
         f"Machine: `{df.machine.iloc[0]}`.{dim_note}\n",
         "All three t-SNE libraries optimize the **same P**, built from one shared pynndescent k-NN graph "
         "(FIt-SNE loads it as precomputed affinities), so the optimization columns differ only in how the "
         "gradient is computed. UMAP runs its defaults on a k=15 graph from the same k-NN code.\n"]

    L.append("## 1. Optimization time (shared graph, identical P)\n")
    rows = []
    for t in threads:
        for n in sizes:
            cell = {m: runs[(runs.method == m) & (runs.n == n) & (runs.threads == t)].t_optimize.tolist() for m in ORDER}
            if not any(cell.values()):
                continue
            h = np.mean(cell["tsne-h2pack"]) if cell["tsne-h2pack"] else np.nan
            ratio = lambda m: f"{np.mean(cell[m]) / h:.1f}×" if cell[m] and h == h else "–"
            rows.append([t, fmt_n(n)] + [mean_std(cell[m]) for m in ORDER] + [ratio("FIt-SNE"), ratio("openTSNE")])
    L.append(md_table(["threads", "N"] + ORDER + ["FIt-SNE / h2pack", "openTSNE / h2pack"], rows) + "\n")
    L.append("![optimization time](fig_optimization_time.png)\n")

    L.append("## 2. Embedding quality (one common P for every library)\n")
    L.append(f"KL against the common P and kNN-preservation@10. The three t-SNE embeddings count as matched when "
             f"each is within {config.KL_TOLERANCE:.0%} (KL) and {config.KNN_PRES_TOLERANCE} (kNN-preservation) of "
             f"the best of the three. UMAP optimizes a different objective; its scores are shown for reference.\n")
    rows = []
    for t in threads:
        for n in sizes:
            s = summary[(summary.n == n) & (summary.threads == t)].set_index("method")
            if s.empty:
                continue
            tsne = s.reindex(config.TSNE_METHODS).dropna(subset=["kl"])
            matched = "–"
            if len(tsne) > 1:
                ok = ((tsne.kl <= tsne.kl.min() * (1 + config.KL_TOLERANCE)) &
                      (tsne.knn_preservation >= tsne.knn_preservation.max() - config.KNN_PRES_TOLERANCE))
                matched = "yes" if ok.all() else "NO: " + ", ".join(tsne.index[~ok])
            cells = [f"{s.kl.get(m, np.nan):.3f} / {s.knn_preservation.get(m, np.nan):.3f}" if m in s.index else "–"
                     for m in ORDER]
            rows.append([t, fmt_n(n)] + cells + [matched])
    L.append(md_table(["threads", "N"] + [f"{m} (KL / kNN-pres)" for m in ORDER] + ["t-SNE matched?"], rows) + "\n")

    L.append("## 3. Time breakdown, end to end\n")
    L.append("`k-NN` is the shared pynndescent build (k=90 for t-SNE, k=15 for UMAP), `setup` is everything between "
             "the graph and the first iteration (perplexity calibration, symmetrization, initialization). "
             "FIt-SNE's wrapper accepts finished affinities but not a k-NN graph, so its front end is measured "
             "separately as its own bundled Annoy search plus calibration (one rep per size). "
             f"numba's JIT compilation is excluded from the pynndescent and UMAP times: it is a per-process "
             f"constant (measured here: {df[df.method == 'knn'].t_jit_warmup.mean():.1f} s for pynndescent), "
             "paid on a toy input before the timer starts.\n")
    rows = []
    for t in threads:
        for n in sizes:
            for m in ORDER:
                s = summary[(summary.method == m) & (summary.n == n) & (summary.threads == t)]
                if s.empty:
                    continue
                s = s.iloc[0]
                knn_txt = f"{s.t_front_end:.1f} (own Annoy + P)" if m == "FIt-SNE" else f"{s.t_knn_shared:.1f}"
                setup_txt = "–" if m in ("FIt-SNE", "UMAP") else f"{s.t_setup:.1f}"
                share = s.t_front_end / s.t_total if s.t_total == s.t_total and s.t_total > 0 else np.nan
                rows.append([t, fmt_n(n), m, knn_txt, setup_txt, f"{s.t_optimize:.1f}", f"{s.t_total:.1f}",
                             f"{share:.0%}" if share == share else "–", f"{s.peak_rss_gb:.1f}"])
    L.append(md_table(["threads", "N", "library", "k-NN", "setup", "optimize", "total", "front-end share",
                       "peak RSS (GB)"], rows) + "\n")
    L.append("![time breakdown](fig_time_breakdown.png)\n")

    if len(threads) > 1:
        lo, hi = threads[0], threads[-1]
        L.append(f"## 4. Thread scaling from {lo} thread\n")
        for hi in threads[1:]:
            rows = []
            for n in sizes:
                row = [fmt_n(n)]
                for m in ORDER:
                    a = summary[(summary.method == m) & (summary.n == n) & (summary.threads == lo)]
                    b = summary[(summary.method == m) & (summary.n == n) & (summary.threads == hi)]
                    row.append(f"{a.t_optimize.iloc[0] / b.t_optimize.iloc[0]:.1f}×" if len(a) and len(b) else "–")
                k = df[(df.method == "knn") & (df.n == n) & (df.k == config.K_TSNE)].groupby("threads").t_knn.mean()
                row.append(f"{k[lo] / k[hi]:.1f}×" if lo in k.index and hi in k.index else "–")
                rows.append(row)
            L.append(f"{lo} → {hi} threads:\n")
            L.append(md_table(["N"] + [f"{m} optimize" for m in ORDER] + ["pynndescent k-NN (k=90)"], rows) + "\n")

    L.append("## 5. Inside tsne-h2pack's optimization\n")
    rows = []
    h2 = runs[runs.method == "tsne-h2pack"]
    for t in threads:
        for n in sizes:
            g = h2[(h2.n == n) & (h2.threads == t)]
            if g.empty:
                continue
            d = pd.DataFrame(g.detail.tolist())
            opt = g.t_optimize.mean()
            parts = [d.attractive.mean(), d.h2_rebuild.mean(), d.h2_matvec.mean()]
            rows.append([t, fmt_n(n), f"{opt:.1f}"] + [f"{p:.1f} ({p / opt:.0%})" for p in parts] +
                        [f"{opt - sum(parts):.1f}"])
    L.append(md_table(["threads", "N", "optimize", "attractive forces", "H² rebuild", "H² matvec", "other"], rows) + "\n")

    if failed:
        L.append("## Failed cells\n")
        L.append(md_table(["library", "N", "threads", "rep", "error"],
                          [[r["method"], fmt_n(r["n"]), r["threads"], r["rep"], r["error"][:160]] for r in failed]) + "\n")

    with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(L))


def main():
    df, failed = load()
    summary = build_summary(df)
    summary.sort_values(["threads", "n", "method"]).to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    figures(summary)
    report(df, summary, failed)
    print(f"{len(df)} records -> results/summary.csv, results/REPORT.md, results/fig_*.png")


if __name__ == "__main__":
    main()
