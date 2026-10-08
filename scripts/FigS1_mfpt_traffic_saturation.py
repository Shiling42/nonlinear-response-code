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


def random_nstate_gate_continuous(rng: np.random.Generator, N: int, m: int, n: int) -> CTMC:
    W = np.zeros((N, N), float)
    base_lo, base_hi = np.log(5e-3), np.log(2e1)
    gamma = float(rng.uniform(0.0, 6.0))

    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            is_gate = ((i == m and j == n) or (i == n and j == m))
            shift = 0.5 * gamma if is_gate else -0.25 * gamma
            val = rng.uniform(base_lo + shift, base_hi + shift)
            W[i, j] = float(np.exp(np.clip(val, np.log(1e-4), np.log(1e3))))

    return CTMC(W)


def mean_time_to_first_direct_edge(W: np.ndarray, m: int, n: int) -> float:
    N = W.shape[0]
    lam = -np.diag(W)

    A = np.zeros((N, N), float)
    b = np.zeros(N, float)

    for i in range(N):
        A[i, i] = 1.0
        b[i] = 1.0 / lam[i]
        for j in range(N):
            if j == i:
                continue
            p_ji = W[j, i] / lam[i]
            if i == m and j == n:
                continue
            A[i, j] -= p_ji

    E = np.linalg.solve(A, b)
    return float(E[m])


def remove_direct_edge_and_build_ctmc(ctmc: CTMC, m: int, n: int) -> CTMC:
    """Return modified CTMC with directed edge m->n removed.

    Matrix convention in this project: W[to, from]. So m->n is entry [n, m].
    """
    G = ctmc.W.copy()
    R = np.zeros_like(G)
    for j in range(G.shape[1]):
        for i in range(G.shape[0]):
            if i != j:
                R[i, j] = max(0.0, G[i, j])
    R[n, m] = 0.0
    return CTMC(R)


def sample_points(N: int, n_keep: int, seed: int, phi_min: float, phi_max: float, trial_factor: int = 1800):
    rng = np.random.default_rng(seed)

    x, y_eq, y_ineq, ratio = [], [], [], []
    trials = 0
    skipped_unreachable = 0

    while len(x) < n_keep and trials < n_keep * trial_factor:
        trials += 1
        m = int(rng.integers(0, N))
        n = int(rng.integers(0, N - 1))
        if n >= m:
            n += 1

        ctmc = random_nstate_gate_continuous(rng, N=N, m=m, n=n)
        phi_nm = ctmc.flux(n, m)  # m -> n
        if not (phi_min <= phi_nm <= phi_max):
            continue

        invphi = 1.0 / phi_nm
        t_mn = ctmc.mfpt(i=m, j=n)
        t_nm = ctmc.mfpt(i=n, j=m)
        tau_edge = mean_time_to_first_direct_edge(ctmc.W, m=m, n=n)

        # ratio from user criterion
        ctmc_removed = remove_direct_edge_and_build_ctmc(ctmc, m=m, n=n)
        try:
            t_nm_removed = ctmc_removed.mfpt(i=n, j=m)
        except Exception:
            skipped_unreachable += 1
            continue

        if not np.isfinite(t_nm_removed) or t_nm_removed <= 0:
            skipped_unreachable += 1
            continue

        rho = t_nm / t_nm_removed  # saturation => rho -> 1
        rho = float(np.clip(rho, 0.0, 1.0))

        x.append(invphi)
        y_eq.append(t_mn + tau_edge)
        y_ineq.append(t_mn + t_nm)
        ratio.append(rho)

    x, y_eq, y_ineq, ratio = map(np.array, (x, y_eq, y_ineq, ratio))
    idx = np.argsort(ratio)
    return x[idx], y_eq[idx], y_ineq[idx], ratio[idx], trials, skipped_unreachable


def get_cmap():
    return LinearSegmentedColormap.from_list(
        'contrast_warm', ['#f08a24', '#e4572e', '#c5283d', '#8f2d56', '#3a0f4a'], N=256
    )


def draw_figure(N: int, x: np.ndarray, y_eq: np.ndarray, y_ineq: np.ndarray, ratio: np.ndarray, out_png: Path, out_pdf: Path, out_svg: Path):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    cmap = get_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(7.2, 3.0), dpi=600)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    sc = ax0.scatter(x, y_eq, c=ratio, cmap=cmap, norm=norm, s=7, alpha=0.75, edgecolors='none', rasterized=True)
    lo0, hi0 = min(x.min(), y_eq.min()), max(x.max(), y_eq.max())
    pad0 = 0.03 * (hi0 - lo0)
    lo0, hi0 = lo0 - pad0, hi0 + pad0
    ax0.plot([lo0, hi0], [lo0, hi0], ls='--', lw=1.0, color='#1b998b', label=r'$y=x$')
    ax0.set_xlim(lo0, hi0)
    ax0.set_ylim(lo0, hi0)
    ax0.set_xlabel(r'$1/\phi_{nm}$')
    ax0.set_ylabel(r'$\langle t_{mn}\rangle + \langle t^{\rm edge}_{nm}\rangle$')
    ax0.set_title('')
    ax0.legend(frameon=False, loc='upper left')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    ax1.scatter(x, y_ineq, c=ratio, cmap=cmap, norm=norm, s=7, alpha=0.75, edgecolors='none', rasterized=True)
    lo1, hi1 = min(x.min(), y_ineq.min()), max(x.max(), y_ineq.max())
    pad1 = 0.03 * (hi1 - lo1)
    lo1, hi1 = lo1 - pad1, hi1 + pad1
    ax1.plot([lo1, hi1], [lo1, hi1], ls='--', lw=1.0, color='#1b998b', label=r'$y=x$')
    ax1.set_xlim(lo1, hi1)
    ax1.set_ylim(lo1, hi1)
    ax1.set_xlabel(r'$1/\phi_{nm}$')
    ax1.set_ylabel(r'$\langle t_{mn}\rangle + \langle t_{nm}\rangle$')
    ax1.set_title('')
    ax1.legend(frameon=False, loc='upper left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r'$\rho_{nm}=\langle t_{nm}\rangle/\langle t_{nm}\rangle^{\setminus(m\to n)}$')
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.subplots_adjust(left=0.08, right=0.93, top=0.90, bottom=0.16, wspace=0.30)
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)


def run_one(N: int, out_root: Path, samples: int, seed: int, phi_min: float, phi_max: float):
    x, y_eq, y_ineq, ratio, trials, skipped = sample_points(
        N=N, n_keep=samples, seed=seed + N, phi_min=phi_min, phi_max=phi_max
    )

    figdir = out_root / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)

    out_png = figdir / f'FigS1_mfpt_traffic_saturation_N{N}.png'
    out_pdf = figdir / f'FigS1_mfpt_traffic_saturation_N{N}.pdf'
    out_svg = figdir / f'FigS1_mfpt_traffic_saturation_N{N}.svg'
    draw_figure(N, x, y_eq, y_ineq, ratio, out_png, out_pdf, out_svg)

    eq_res = x - y_eq
    report = (
        f'N={N}, kept={len(x)}, trials={trials}, skipped_unreachable={skipped}\n'
        f'equality residual min={eq_res.min():.3e}, median={np.median(eq_res):.3e}, max_abs={np.max(np.abs(eq_res)):.3e}\n'
        f'ratio rho min/median/max={ratio.min():.4f}/{np.median(ratio):.4f}/{ratio.max():.4f}\n'
        f'Pr(rho>0.95)={(ratio > 0.95).mean():.4f}, Pr(rho>0.99)={(ratio > 0.99).mean():.4f}\n'
    )
    (out_root / 'figures' / f'FigS1_stats_N{N}.txt').write_text(report, encoding='utf-8')
    return report


def parse_args():
    p = argparse.ArgumentParser(description='Unified N-state plot with saturation ratio color.')
    p.add_argument('--states', nargs='+', type=int, default=[4])
    p.add_argument('--samples', type=int, default=7000)
    p.add_argument('--seed', type=int, default=2026021317)
    p.add_argument('--phi-min', type=float, default=5e-3)
    p.add_argument('--phi-max', type=float, default=0.95)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    lines = [
        '# Fig. S1: method and statistics',
        '- Panel A: equality check x=1/phi_nm vs y=t_mn+t_nm^{edge}.',
        '- Panel B: inequality panel x=1/phi_nm vs y=t_mn+t_nm.',
        '- Colour: rho = t_nm / t_nm(edge m->n removed), in [0,1].',
        ''
    ]

    for N in args.states:
        if N < 3:
            continue
        lines.append(run_one(N, root, args.samples, args.seed, args.phi_min, args.phi_max))

    summary = '\n'.join(lines)
    (root / 'figures' / 'FigS1_method_and_stats.txt').write_text(summary, encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
