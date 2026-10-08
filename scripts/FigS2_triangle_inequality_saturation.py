#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from markov_response import CTMC


def random_nstate_ctmc(rng: np.random.Generator, N: int) -> CTMC:
    W = np.zeros((N, N), float)
    lo, hi = np.log(5e-2 ), np.log(2e1)
    for j in range(N):
        for i in range(N):
            if i != j:
                W[i, j] = float(np.exp(rng.uniform(lo, hi)))
    return CTMC(W)


def remove_direct_edge_and_build_ctmc(ctmc: CTMC, m: int, n: int) -> CTMC:
    """Remove directed edge m->n (matrix index [n,m]) and rebuild CTMC."""
    G = ctmc.W.copy()
    R = np.zeros_like(G)
    for j in range(G.shape[1]):
        for i in range(G.shape[0]):
            if i != j:
                R[i, j] = max(0.0, G[i, j])
    R[n, m] = 0.0
    return CTMC(R)


def remove_node_and_build_ctmc(ctmc: CTMC, n: int) -> tuple[CTMC, dict[int, int]]:
    """Remove state n entirely and return reduced CTMC + old->new index map."""
    G = ctmc.W.copy()
    N = G.shape[0]
    keep = [i for i in range(N) if i != n]
    idx_map = {old: new for new, old in enumerate(keep)}

    R = np.zeros((N - 1, N - 1), float)
    for new_j, old_j in enumerate(keep):
        for new_i, old_i in enumerate(keep):
            if old_i == old_j:
                continue
            R[new_i, new_j] = max(0.0, G[old_i, old_j])

    return CTMC(R), idx_map


def sample_points(N: int, n_keep: int, seed: int, trial_factor: int = 2000):
    rng = np.random.default_rng(seed)
    x_sum, y_direct, ratio = [], [], []
    trials, skipped = 0, 0

    while len(x_sum) < n_keep and trials < n_keep * trial_factor:
        trials += 1
        ctmc = random_nstate_ctmc(rng, N)

        m = int(rng.integers(0, N))
        n = int(rng.integers(0, N - 1))
        if n >= m:
            n += 1

        candidates = [i for i in range(N) if i not in (m, n)]
        if not candidates:
            continue
        k = int(candidates[int(rng.integers(0, len(candidates)))])

        # notation: t_ab = MFPT from b to a
        # triangle inequality: <t_kn> <= <t_mn> + <t_km>
        # i.e. n->k direct <= n->m + m->k via intermediate m
        t_kn = ctmc.mfpt(i=k, j=n)  # n -> k
        t_km = ctmc.mfpt(i=k, j=m)  # m -> k
        t_mn = ctmc.mfpt(i=m, j=n)  # n -> m

        # color indicator: node-removal ratio on n->k MFPT (remove node m)
        ctmc_removed_m, idx_map = remove_node_and_build_ctmc(ctmc, n=m)
        try:
            t_kn_removed_m = ctmc_removed_m.mfpt(i=idx_map[k], j=idx_map[n])
        except Exception:
            skipped += 1
            continue

        if not np.isfinite(t_kn_removed_m) or t_kn <= 0:
            skipped += 1
            continue

        rho = float(t_kn / t_kn_removed_m)

        x_sum.append(t_mn + t_km)
        y_direct.append(t_kn)
        ratio.append(rho)

    x = np.array(x_sum)
    y = np.array(y_direct)
    c = np.array(ratio)
    # draw high-ratio first, low-ratio last (on top)
    idx = np.argsort(-c)
    return x[idx], y[idx], c[idx], trials, skipped


def get_cmap():
    return LinearSegmentedColormap.from_list(
        'contrast_warm', ['#f08a24', '#e4572e', '#c5283d', '#8f2d56', '#3a0f4a'], N=256
    )


def draw_figure(N: int, x: np.ndarray, y: np.ndarray, c: np.ndarray, out_png: Path, out_pdf: Path, out_svg: Path):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig = plt.figure(figsize=(3.9, 3.0), dpi=600)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    
    cmap = get_cmap()
    cmap.set_over(cmap(1.0))
    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=Normalize(vmax=1.2), s=7, alpha=0.75, edgecolors='none', rasterized=True)

    # robust shared axis range to reduce outlier domination
    q_lo = float(min(np.quantile(x, 0.01), np.quantile(y, 0.01)))
    q_hi = float(max(np.quantile(x, 0.99), np.quantile(y, 0.99)))
    lo_raw, hi_raw = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    if q_hi <= q_lo:
        q_lo, q_hi = lo_raw, hi_raw
    pad = 0.05 * (q_hi - q_lo if q_hi > q_lo else max(1e-9, hi_raw - lo_raw))
    lo, hi = q_lo - pad, q_hi + pad
    ax.plot([lo, hi], [lo, hi], ls='--', lw=1.0, color='#1b998b', label=r'$y=x$')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r'$\langle t_{km}\rangle+\langle t_{mn}\rangle +$')
    ax.set_ylabel(r'$\langle t_{kn}\rangle$')
    ax.set_title('')
    ax.legend(frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cb = fig.colorbar(sc, cax=cax, extend='max')
    cb.set_label(r'$\rho_{kn}^{\setminus m}=\langle t_{kn}\rangle/\langle t_{kn}\rangle^{\setminus m}$')
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.subplots_adjust(left=0.14, right=0.94, top=0.95, bottom=0.16, wspace=0.25)
    fig.savefig(out_png, bbox_inches='tight')
    #fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--states', nargs='+', type=int, default=[6])
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=2026021329)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    figdir = root / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)

    lines = ['# Fig. S2: method and statistics', '- Verify: <t_kn> <= <t_mn> + <t_km>.', '- Color uses node-removal ratio on n->k MFPT: <t_kn>/<t_kn>^(remove m).']

    for N in args.states:
        if N < 3:
            continue
        x, y, c, trials, skipped = sample_points(N, args.samples, args.seed + N)
        draw_figure(
            N,
            x,
            y,
            c,
            figdir / f'FigS2_triangle_inequality_saturation_N{N}.png',
            figdir / f'FigS2_triangle_inequality_saturation_N{N}.pdf',
            figdir / f'FigS2_triangle_inequality_saturation_N{N}.svg',
        )
        margin = x - y
        lines.append(
            f'N={N}, kept={len(x)}, trials={trials}, skipped_unreachable={skipped}, min_margin={margin.min():.3e}, median_margin={np.median(margin):.3e}, violations={(margin < -1e-12).sum()}, rho_median={np.median(c):.4f}'
        )

    (figdir / 'FigS2_method_and_stats.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
