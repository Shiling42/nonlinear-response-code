"""
Fig4_push_pull_hidden_network.py -- Fig. 4: the push-pull motif embedded in a hidden network.

(a) Scheme. The kinase-catalyzed transition S <-> S* is the only transition that depends on [K] and the only
    ATP-consuming step (B-type perturbation, Delta B = ln([K]/[K]')). S and S* connect to a hidden network of
    unknown size and rates (phosphatase complexes, binding partners, ...).
(b) Measured readout: phosphorylated fraction <O> vs [K]'/[K] for five hidden networks.
(c) Fold change of the ATP-consumption rate j_ATP (= net current through the kinase step): actual value (lines)
    and value inferred from (b) with Eq. (18) (symbols), j'/j = Delta<O> / [(1 - [K]/[K]') d<O>/dln[K]];
    gray = range excluded by Eq. (8), min(1, c) <= j'/j <= max(1, c).

Hidden networks (all reproducible from the seeds below):
  states 0 = S, 1 = S*, 2..H+1 hidden; each hidden state is phosphorylated with probability 1/2;
  topology = random spanning tree avoiding the pair (S, S*) + n random extra edges (never S-S*);
  hidden rates W_ij = exp[-(B_ij - E_j)], B_ij = B_ji ~ U(-3, 3), E_j ~ U(-1, 1)  (detailed balance, no drive);
  kinase W_{S*,S} = [K] exp[-(B_K - E_S - dmu/2)],  W_{S,S*} = [K] exp[-(B_K - E_S* + dmu/2)]  (the only drive).
  The five networks were selected from a random search (search(), fixed seed) as those with the largest readout
  change among networks whose ATP-cost fold change at [K]'/[K] = 10 is close to 1.2, 2, 3.5, 5.5 and 8.

The page is the final printed size (\\linewidth = 247 pt = 246.1 bp); all lettering sizes are printed sizes.
Panel (a) is typeset by pdflatex/TikZ in the idiom of Fig. 3(b); everything else is matplotlib; composed by pdflatex.

usage:  python3 Fig4_push_pull_hidden_network.py [--outdir DIR] [--keep DIR]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, LogLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "figures")

# ============================================================================ model
def steady(W):
    M = W.copy()
    M[-1, :] = 1.0
    b = np.zeros(len(W))
    b[-1] = 1.0
    return np.linalg.solve(M, b)


def make_net(H, BK, dmu, seed):
    rng = np.random.default_rng(seed)
    n = H + 2
    phos = np.zeros(n, bool)
    phos[1] = True
    phos[2:] = rng.random(H) < 0.5
    E = rng.uniform(-1, 1, n)
    A = np.zeros((n, n), bool)
    order = rng.permutation(n)
    if set(order[:2]) == {0, 1}:
        order[[1, 2]] = order[[2, 1]]
    for k in range(1, n):                       # random spanning tree, never using the pair (S, S*)
        while True:
            j = order[rng.integers(0, k)]
            i = order[k]
            if {i, j} != {0, 1}:
                break
        A[i, j] = A[j, i] = True
    extra = n
    while extra:                                # extra random edges (cycles), never S-S*
        i, j = rng.integers(0, n, 2)
        if i != j and {i, j} != {0, 1} and not A[i, j]:
            A[i, j] = A[j, i] = True
            extra -= 1
    B = rng.uniform(-3, 3, (n, n))
    B = (B + B.T) / 2
    W = np.where(A, np.exp(-(B - E[None, :])), 0.0)
    np.fill_diagonal(W, 0.0)

    def gen(K):
        Wk = W.copy()
        Wk[1, 0] = K * np.exp(-(BK - E[0] - dmu / 2))       # S -> S*   (ATP -> ADP + ...)
        Wk[0, 1] = K * np.exp(-(BK - E[1] + dmu / 2))       # S* -> S
        return Wk - np.diag(Wk.sum(axis=0))

    return gen, phos.astype(float)


def pj(gen, K):
    W = gen(K)
    p = steady(W)
    return p, W[1, 0] * p[0] - W[0, 1] * p[1]


def curves(gen, O, cs):
    """readout <O>(c), actual j'/j, and j'/j inferred from the readout with Eq. (18)."""
    p1, j1 = pj(gen, 1.0)
    f1 = O @ p1
    h = 1e-3
    f = lambda s: O @ pj(gen, np.exp(s))[0]
    chi = (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)     # d<O>/dln[K] at [K]' = [K]
    fc = np.array([O @ pj(gen, c)[0] for c in cs])
    jc = np.array([pj(gen, c)[1] for c in cs]) / j1
    with np.errstate(divide="ignore", invalid="ignore"):
        inf = (fc - f1) / ((1.0 - 1.0 / cs) * chi)
    return fc, jc, inf, chi, j1, f1


TARGETS = [1.2, 2.0, 3.5, 5.5, 8.0]


def search(n_iter=4000, seed=11):
    rng = np.random.default_rng(seed)
    best = {t: None for t in TARGETS}
    for _ in range(n_iter):
        BK = rng.uniform(-2, 7)
        dmu = rng.uniform(2, 8)
        H = int(rng.integers(20, 41))
        s = int(rng.integers(1e9))
        gen, O = make_net(H, BK, dmu, s)
        (p01, j01), (p1, j1), (p10, j10) = pj(gen, 0.1), pj(gen, 1.0), pj(gen, 10.0)
        r10, span = j10 / j1, abs(O @ p10 - O @ p01)
        for t in TARGETS:
            if abs(np.log(r10 / t)) < 0.1 and (best[t] is None or span > best[t][4]):
                best[t] = (H, BK, dmu, s, span, r10)
    return [best[t] for t in TARGETS]


# ============================================================================ layout (bp, final size)
W_PAGE, H_PAGE = 246.1, 172.0
AX_B = (140.0, 104.0, 229.0, 167.0)          # (x0, y0, x1, y1)
AX_C = (140.0, 25.0, 229.0, 92.0)
COLORS = ["#93C4E8", "#5A9FD4", "#2B7BBF", "#12508F", "#0A2A5C"]   # light -> dark = cheap -> costly
EDGE = "#1B2631"
GRAY_BOUND = "#BDC3C7"
INACC = "#6E6F70"
FS_TICK, FS_LAB, FS_PANEL, FS_SMALL = 6.3, 7.5, 7.2, 6.2

# scheme geometry (bp): kinase edge S <-> S* on top; the hidden network is a U-shaped band below it that
# carries the return path S* -> ... -> S; the ATP-driven cycle current j_ATP circulates in the hole of the U.
RING_C, RING_R, RING_r = np.array([57.0, 109.0]), 55.0, 25.0
TH0, TH1 = 168.0, 372.0                      # angular extent of the band (degrees; ends slightly above centre)
P_S, P_SS = np.array([17.0, 147.0]), np.array([94.0, 147.0])
# the band is drawn narrower and taller than the construction ring (keeps "(c)" clear, fills panel (a))
CX_NEW, SX, Y_ANCH, KY = 55.5, 53.0 / 55.0, 132.0, 1.33


def T(P):
    P = np.atleast_2d(np.asarray(P, float)).copy()
    P[:, 0] = CX_NEW + (P[:, 0] - 57.0) * SX
    P[:, 1] = Y_ANCH - KY * (Y_ANCH - P[:, 1])
    return P


ARC_C, ARC_R = np.array([55.5, 97.0]), 13.0


def band_graph(seed=3, n=21, sep=9.3):
    """schematic hidden network: nodes inside the U-shaped band (one fixed node at each end, next to S and
    S*), each joined to its three nearest neighbours, so that the network has branches and loops."""
    rng = np.random.default_rng(seed)
    mid = 0.5 * (RING_R + RING_r)
    endL = RING_C + mid * np.array([np.cos(np.radians(TH0 + 5)), np.sin(np.radians(TH0 + 5))])
    endR = RING_C + mid * np.array([np.cos(np.radians(TH1 - 5)), np.sin(np.radians(TH1 - 5))])
    pts = [endL, endR]
    tries = 0
    while len(pts) < n and tries < 200000:
        tries += 1
        th = np.radians(rng.uniform(TH0 + 4, TH1 - 4))
        rr = rng.uniform(RING_r + 4.8, RING_R - 4.8)
        q = RING_C + rr * np.array([np.cos(th), np.sin(th)])
        if all(np.linalg.norm(q - p) > sep for p in pts):
            pts.append(q)
    pts = np.array(pts)
    edges = set()
    for i in range(len(pts)):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.argsort(d)[1:4]:
            if d[j] < 2.3 * sep:
                edges.add(tuple(sorted((i, int(j)))))
    near = lambda P: [int(k) for k in np.argsort(np.linalg.norm(pts - P, axis=1))[:2]]
    return pts, sorted(edges), near(P_S), near(P_SS)


def band_patch():
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    th = np.radians(np.linspace(TH0, TH1, 200))
    outer = RING_C + RING_R * np.c_[np.cos(th), np.sin(th)]
    inner = RING_C + RING_r * np.c_[np.cos(th[::-1]), np.sin(th[::-1])]
    # rounded ends: half discs joining outer and inner rims
    def cap(t, sgn):
        c = RING_C + 0.5 * (RING_R + RING_r) * np.array([np.cos(t), np.sin(t)])
        h = 0.5 * (RING_R - RING_r)
        a = np.linspace(0, np.pi, 40)
        u = np.array([np.cos(t), np.sin(t)])          # radial direction
        v = sgn * np.array([-np.sin(t), np.cos(t)])   # outward tangent at this end
        return c[None, :] + h * (np.cos(a)[:, None] * (-u if sgn > 0 else u)[None, :]
                                 + np.sin(a)[:, None] * v[None, :])
    capR = cap(np.radians(TH1), +1)
    capL = cap(np.radians(TH0), -1)
    verts = np.vstack([outer, capR[::-1] if np.linalg.norm(capR[0]-outer[-1]) > np.linalg.norm(capR[-1]-outer[-1]) else capR,
                       inner, capL[::-1] if np.linalg.norm(capL[0]-inner[-1]) > np.linalg.norm(capL[-1]-inner[-1]) else capL,
                       outer[:1]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    verts = T(verts)
    return PathPatch(Path(verts, codes), facecolor="#EEF0F2", edgecolor="#C9CFD5", lw=0.5, zorder=0)


def make_base(nets, path):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": FS_LAB, "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.minor.width": 0.35, "ytick.minor.width": 0.35,
                         "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                         "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
                         "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
                         "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK})
    fig = plt.figure(figsize=(W_PAGE / 72, H_PAGE / 72))

    # ---- panel (a) background: hidden network (the typeset parts are added by TikZ)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, W_PAGE)
    bg.set_ylim(0, H_PAGE)
    bg.axis("off")
    bg.add_patch(band_patch())
    pts, edges, nS, nSS = band_graph()
    pts = T(pts)
    edges = sorted(set(edges) | {(13, 20)})           # removes the only bridge next to S
    for i, j in edges:
        bg.plot(*zip(pts[i], pts[j]), color="#AEB6BF", lw=0.45, zorder=1, solid_capstyle="round")
    for k in nS:
        bg.plot(*zip(P_S + np.array([0.0, -6.5]), pts[k]), color="#85929E", lw=0.55, zorder=1)
    for k in nSS:
        bg.plot(*zip(P_SS + np.array([-1.6, -6.5]), pts[k]), color="#85929E", lw=0.55, zorder=1)
    bg.scatter(pts[:, 0], pts[:, 1], s=6.5, c="#85929E", edgecolors="none", zorder=2)
    bg.text(CX_NEW, T([0, RING_C[1] - RING_R])[0, 1] - 7.0, "hidden network", fontfamily="Verdana", style="italic",
            fontsize=FS_SMALL + 0.3, color=INACC, ha="center", va="center", zorder=3)

    def ax_at(box):
        x0, y0, x1, y1 = box
        return fig.add_axes([x0 / W_PAGE, y0 / H_PAGE, (x1 - x0) / W_PAGE, (y1 - y0) / H_PAGE])

    def logx(ax):
        ax.set_xscale("log")
        ax.set_xlim(10 ** -1.045, 10 ** 1.045)          # padded: end markers are not clipped
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10"]))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color="#B0B0B0", alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)

    # ---- (b) measured readout
    ab = ax_at(AX_B)
    logx(ab)
    for net, col in zip(nets, COLORS):
        ab.plot(net["cf"], net["ff"], color=col, lw=1.0, zorder=2, solid_capstyle="round")
        ab.plot(net["cm_all"], net["fm_all"], "o", ms=3.0, mfc=col, mec=EDGE, mew=0.35, zorder=3)
    ab.set_ylim(0.38, 0.82)
    ab.yaxis.set_major_locator(FixedLocator([0.4, 0.6, 0.8]))
    ab.tick_params(labelbottom=False)
    ab.set_ylabel(r"$\langle\mathcal{O}\rangle$", fontsize=FS_LAB + 0.5, labelpad=2.0)
    ab.text(0.035, 0.94, "phosphorylated fraction", transform=ab.transAxes, fontsize=FS_SMALL,
            color="#4D5656", ha="left", va="top")

    # ---- (c) ATP-consumption fold change: actual vs inferred, inside the bound of Eq. (8)
    ac = ax_at(AX_C)
    logx(ac)
    ac.set_yscale("log")
    ac.set_ylim(10 ** -1.06, 10 ** 1.06)
    ac.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    ac.yaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10"]))
    ac.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ac.yaxis.set_minor_formatter(NullFormatter())
    for a_ in (ab, ac):                                   # no grid lines next to the spines
        for loc, gl in zip(a_.xaxis.get_majorticklocs(), a_.get_xgridlines()):
            gl.set_visible(loc == 1)
    for loc, gl in zip(ac.yaxis.get_majorticklocs(), ac.get_ygridlines()):
        gl.set_visible(loc == 1)
    cb = np.logspace(-1.05, 1.05, 400)
    ac.fill_between(cb, np.maximum(1, cb), 12, color=GRAY_BOUND, alpha=0.3, lw=0, zorder=0)
    ac.fill_between(cb, 0.08, np.minimum(1, cb), color=GRAY_BOUND, alpha=0.3, lw=0, zorder=0)
    ac.plot(cb, np.maximum(1, cb), color=GRAY_BOUND, lw=1.3, zorder=1, solid_capstyle="butt")
    ac.plot(cb, np.minimum(1, cb), color=GRAY_BOUND, lw=1.3, zorder=1, solid_capstyle="butt")
    for net, col in zip(nets, COLORS):
        ac.plot(net["cf"], net["jf"], color=col, lw=1.0, zorder=2, solid_capstyle="round")
        ac.plot(net["cm"], net["inf"], "o", ms=3.0, mfc=col, mec=EDGE, mew=0.35, zorder=3)
    for k in (0, len(nets) - 1):                          # the two extreme costs at [K]'/[K] = 10
        ac.annotate("%.1f\u00d7" % nets[k]["r10"], (10 ** 1.045, nets[k]["r10"]), xytext=(1.5, 0),
                    textcoords="offset points", fontsize=FS_SMALL, color="#4D5656",
                    ha="left", va="center", annotation_clip=False)
    ac.text(0.115, 4.6, "inaccessible", fontfamily="Verdana", style="italic", fontsize=FS_SMALL + 0.3,
            color=INACC, ha="left", va="center")
    handles = [Line2D([], [], color="#5D6D7E", lw=1.0, label="actual"),
               Line2D([], [], color="none", marker="o", ms=3.0, mfc="#AEB6BF", mec=EDGE, mew=0.35,
                      label="inferred from (b)")]
    ac.legend(handles=handles, loc="lower right", fontsize=FS_SMALL, frameon=False, handlelength=1.3,
              handletextpad=0.4, borderaxespad=0.25, labelspacing=0.25)
    ac.set_xlabel(r"$[K]'/[K]$", fontsize=FS_LAB, labelpad=1.0)
    ac.set_ylabel(r"$j_{\mathrm{ATP}}'/j_{\mathrm{ATP}}$", fontsize=FS_LAB, labelpad=1.0)

    for lab, x, y in (("(a)", 1.0, H_PAGE - 1.0), ("(b)", 112.0, H_PAGE - 1.0), ("(c)", 112.0, 97.0)):
        fig.text(x / W_PAGE, y / H_PAGE, lab, fontfamily="Verdana", fontsize=FS_PANEL, ha="left", va="top")
    fig.savefig(path)
    plt.close(fig)


# ============================================================================ panel (a): TikZ overlay
SCALE = 0.90                     # CM at 10 pt x 0.9: state labels 9 pt, rate labels (scriptstyle) 6.3 pt
PT2BP = 72.0 / 72.27
AXIS = 2.5 * SCALE * PT2BP       # math axis -> centre line of the harpoon pair
CAPH = 6.83 * SCALE * PT2BP      # cap height of cmmi10 'S'


# Coupled ATP <-> ADP arc on the kinase step, and P_i release from the hidden network.
# Every line in panel (a) is a vector stroke of the same width LW: the two reaction harpoons (redrawn as strokes at the
# exact position of the \xrightleftharpoons glyphs, so that they and the arc look equally thick in every PDF viewer),
# the ATP <-> ADP arc, the P_i arrow and the j_ATP arrow (red).
LW = 0.40                        # common stroke width (bp)
TOP_Y, BOT_Y = 147.81, 146.13    # centre lines of the forward / backward harpoon
TOP_X = (26.26, 83.20)           # forward harpoon: tail x, tip x
BOT_X = (82.77, 25.85)           # backward harpoon: tail x, tip x
HTIP = "{Computer Modern Rightarrow[harpoon,length=2.4bp,width=4.4bp]}"   # half head of the reaction harpoons
TIP = "{Computer Modern Rightarrow[length=2.4bp,width=4.0bp]}"            # full head of every other arrow
LAB_UP = (42.70, 152.27)         # baseline origin of w_+[K] (bp); lifted by LABEL_RAISE so the arc fits under it
LAB_DN = (43.80, 136.90)         # baseline origin of w_-[K] (bp)
LABEL_RAISE = 2.6
CPL_R = 26.0                     # radius of the arc (bp)
THETA_UP = 55.0                  # half opening angle of the arc (deg)
PI_END = np.array([55.6, 74.4])  # tip of the P_i arrow, inside the loop below j_ATP
PI_START = np.array([62.6, 61.2])  # start, inside the grey cloud in the empty area right of the loop's bottom
PI_WAVE = ((0.30, 1.25), (0.60, -0.70))   # two uneven bends: (fraction along the arrow, sideways offset in bp)


def assert_in_cloud(p, pts, edges, d_edge=2.6, d_node=4.6):
    """the P_i arrow must start inside the grey band (the cloud), not in the loop, away from network edges and nodes."""
    def seg_dist(a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        return np.linalg.norm(p - (a + t * ab))
    X = 57.0 + (p[0] - CX_NEW) / SX                        # back to construction coordinates
    Y = Y_ANCH - (Y_ANCH - p[1]) / KY
    rad = np.hypot(X - RING_C[0], Y - RING_C[1])
    assert RING_r + 1.5 < rad < RING_R - 1.5, "P_i start not inside the band"
    assert min(seg_dist(pts[i], pts[j]) for i, j in edges) >= d_edge, "P_i start too close to an edge"
    assert np.min(np.linalg.norm(pts - p, axis=1)) >= d_node, "P_i start too close to a node"


def tikz(base_pdf):
    L = []
    for pos, tex in ((P_S, r"S"), (P_SS, r"S^{*}")):
        L.append(r"\node[anchor=base,inner sep=0,outer sep=0] at (%.3f,%.3f) {\scalebox{\sch}{$%s$}};"
                 % (pos[0], pos[1] - CAPH / 2, tex))
    s0, s1 = P_S[0] + 8.5, P_SS[0] - 10.5
    # reaction harpoons S <-> S* as strokes (forward: barb up at the right end; backward: barb down at the left end)
    L.append(r"\draw[line width=%.2fbp,-%s] (%.3f,%.3f) -- (%.3f,%.3f);" % (LW, HTIP, TOP_X[0], TOP_Y, TOP_X[1], TOP_Y))
    L.append(r"\draw[line width=%.2fbp,-%s] (%.3f,%.3f) -- (%.3f,%.3f);" % (LW, HTIP, BOT_X[0], BOT_Y, BOT_X[1], BOT_Y))
    rate = r"\node[anchor=base west,inner sep=0,outer sep=0] at (%.3f,%.3f) {\scalebox{\sch}{$\scriptstyle %s$}};"
    L.append(rate % (LAB_UP[0], LAB_UP[1] + LABEL_RAISE, r"w_+\hl{[K]}"))
    L.append(rate % (LAB_DN[0], LAB_DN[1], r"w_-\hl{[K]}"))
    # coupled reaction: one arc above, touching the forward harpoon; full heads at both ends (ATP <-> ADP)
    xm = 0.5 * (s0 + s1)
    cu = TOP_Y + LW + CPL_R
    pt = lambda c, ang: (xm + CPL_R * np.cos(np.radians(ang)), c + CPL_R * np.sin(np.radians(ang)))
    u0, u1 = pt(cu, 270.0 - THETA_UP), pt(cu, 270.0 + THETA_UP)
    L.append(r"\draw[line width=%.2fbp,%s-%s] (%.3f,%.3f) arc[start angle=%.2f,end angle=%.2f,radius=%.3f];"
             % (LW, TIP, TIP, u0[0], u0[1], 270.0 - THETA_UP, 270.0 + THETA_UP, CPL_R))
    lab = r"\node[anchor=%s,inner sep=0] at (%.3f,%.3f) {\scalebox{\sch}{$\scriptstyle\mathrm{%s}$}};"
    L.append(lab % ("east", u0[0] - 1.4, u0[1] + 1.2, "ATP"))
    L.append(lab % ("west", u1[0] + 1.4, u1[1] + 1.2, "ADP"))
    # P_i release: a slightly wavy arrow (two uneven bends) from inside the grey cloud into the loop
    pts, edges, _, _ = band_graph()
    edges = sorted(set(edges) | {(13, 20)})
    P = T(pts)
    q = PI_START
    assert_in_cloud(q, P, edges)
    u = (PI_END - q) / np.linalg.norm(PI_END - q)
    nrm = np.array([-u[1], u[0]])
    Lq = np.linalg.norm(PI_END - q)
    way = [q] + [q + f * Lq * u + off * nrm for f, off in PI_WAVE] + [q + 0.84 * Lq * u, PI_END]
    L.append(r"\draw[line width=%.2fbp,-%s] plot[smooth,tension=0.7] coordinates {%s};"
             % (LW, TIP, " ".join("(%.3f,%.3f)" % tuple(w) for w in way)))
    L.append(r"\node[anchor=south,inner sep=0] at (%.3f,%.3f) {\scalebox{\sch}{$\scriptstyle\mathrm{P_i}$}};"
             % (PI_END[0], PI_END[1] + 0.9))
    r, th0, th1 = ARC_R, 120.0, -180.0                 # clockwise: S -> S* -> hidden -> S
    L.append(r"\draw[jred,line width=%.2fbp,-%s] "
             r"(%.3f,%.3f) arc[start angle=%.1f,end angle=%.1f,radius=%.2f];"
             % (LW, TIP, ARC_C[0] + r * np.cos(np.radians(th0)), ARC_C[1] + r * np.sin(np.radians(th0)), th0, th1, r))
    L.append(r"\node[inner sep=0,text=jred] at (%.3f,%.3f) {\scalebox{%.3f}{$j_{\mathrm{ATP}}$}};"
             % (ARC_C[0], ARC_C[1], SCALE))
    body = "\n".join(L)
    return r"""\documentclass[border=0pt]{standalone}
\usepackage{graphicx}
\usepackage{mathtools}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,decorations.pathmorphing}
\definecolor{pinkbase}{HTML}{FFC9C9}
\colorlet{pinkK}{pinkbase!76!white}
\definecolor{jred}{HTML}{E03131}
\newcommand{\sch}{%(scale)s}
\newcommand{\hl}[1]{{\setlength{\fboxsep}{0.9pt}\colorbox{pinkK}{$\scriptstyle #1$}}}
\newlength\harppad
\newcommand{\edge}[6]{%%
  \pgfmathsetlengthmacro{\mbw}{(#4bp)/\sch - \harppad}%%
  \node[rotate=#3,anchor=base,inner sep=0,outer sep=0] at (#1,#2)
    {\scalebox{\sch}{$\xrightleftharpoons[{\mathmakebox[\mbw]{#6}}]{\mathmakebox[\mbw]{#5}}$}};}
\begin{document}%%
\sbox0{$\xrightleftharpoons[{\mathmakebox[50pt]{}}]{\mathmakebox[50pt]{}}$}%%
\setlength\harppad{\dimexpr\wd0-50pt\relax}%%
\begin{tikzpicture}[x=1bp,y=1bp]
\useasboundingbox (0bp,0bp) rectangle (%(w)sbp,%(h)sbp);
\node[anchor=south west,inner sep=0,outer sep=0] at (0,0) {\includegraphics{%(base)s}};
%(body)s
\end{tikzpicture}
\end{document}
""" % dict(scale=SCALE, w=W_PAGE, h=H_PAGE, base=base_pdf, body=body)


def pdflatex(texfile, cwd):
    exe = shutil.which("pdflatex") or "/Library/TeX/texbin/pdflatex"
    r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", os.path.basename(texfile)],
                       cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:])
        raise RuntimeError("pdflatex failed")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=PROJECT)
    ap.add_argument("--keep", default=None)
    args = ap.parse_args()

    sel = search()
    cf = np.logspace(-1, 1, 241)
    cm_all = np.logspace(-1, 1, 9)                      # measured concentrations (incl. the reference)
    cm = cm_all[np.abs(cm_all - 1) > 1e-9]
    nets = []
    print("  H   B_K    dmu    seed        j'/j(10)  j'/j(0.1)  Delta<O>(0.1->10)  max rel. error")
    for (H, BK, dmu, s, span, r10) in sel:
        gen, O = make_net(H, BK, dmu, s)
        ff, jf, _, _, _, _ = curves(gen, O, cf)
        fm_all, _, _, _, _, _ = curves(gen, O, cm_all)
        fm, jm, inf, chi, j1, f1 = curves(gen, O, cm)
        err = np.max(np.abs(inf - jm) / jm)
        assert err < 1e-6, err                                             # Eq. (18) exact
        assert np.all((jf >= np.minimum(1, cf) - 1e-9) & (jf <= np.maximum(1, cf) + 1e-9))   # Eq. (8)
        nets.append(dict(cf=cf, ff=ff, jf=jf, cm_all=cm_all, fm_all=fm_all, cm=cm, inf=inf, r10=jf[-1],
                         span=abs(ff[-1] - ff[0]), H=H))
        print("%3d %6.3f %6.3f %10d   %8.3f  %8.3f   %8.3f   %.1e" % (H, BK, dmu, s, jf[-1], jf[0],
                                                                    abs(ff[-1] - ff[0]), err))
    r10 = [n["r10"] for n in nets]
    assert np.all(np.diff(r10) > 0)                     # colours ordered by cost
    print("hidden states: %d-%d;  j'/j at [K]'/[K]=10: %.2f-%.2f;  smallest readout change in network %d "
          "(largest cost %d)" % (min(n["H"] for n in nets), max(n["H"] for n in nets), min(r10), max(r10),
                                 int(np.argmin([n["span"] for n in nets])), int(np.argmax(r10))))

    work = args.keep or tempfile.mkdtemp(prefix="fig4_")
    os.makedirs(work, exist_ok=True)
    make_base(nets, os.path.join(work, "base.pdf"))
    with open(os.path.join(work, "fig.tex"), "w") as fh:
        fh.write(tikz("base.pdf"))
    pdflatex(os.path.join(work, "fig.tex"), work)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "Fig4_push_pull.pdf")
    shutil.copyfile(os.path.join(work, "fig.pdf"), out)
    print("written", out)
    if args.keep is None:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
