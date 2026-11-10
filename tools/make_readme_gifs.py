"""Render the text-free illustrative GIFs used in README.md.

Each animation is a schematic of one PyRPMD workflow driven by a small toy
simulation (ring-polymer normal modes, Langevin or surface-hopping dynamics on
model potentials). They illustrate the ideas only; they are not results.

    python tools/make_readme_gifs.py            # all animations
    python tools/make_readme_gifs.py pimd pcet  # a subset

Requires matplotlib and Pillow. Output: docs/_static/gifs/*.gif
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "_static" / "gifs"

W, H, DPI = 4.8, 3.0, 100
BG = "#ffffff"
INK = "#1f2a44"
GRID = "#d9dee8"
BLUE = "#2f6fb2"
TEAL = "#1d9a8a"
ORANGE = "#ef8a2f"
RED = "#d1495b"
PURPLE = "#7b5ea7"
GRAY = "#9aa3b2"
LIGHT = "#eef2f7"


# --------------------------------------------------------------------------- helpers


def new_fig(ncols=1, width_ratios=None):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    if ncols == 1:
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
        axes = [ax]
    else:
        gs = fig.add_gridspec(1, ncols, width_ratios=width_ratios, left=0.03, right=0.97,
                              bottom=0.06, top=0.94, wspace=0.12)
        axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    for ax in axes:
        clean(ax)
    return fig, axes


def clean(ax, frame=False):
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for side, spine in ax.spines.items():
        spine.set_visible(frame and side in ("left", "bottom"))
        spine.set_color(GRAY)
        spine.set_linewidth(1.2)


def plot_frame(ax):
    """Bare axes: two arrows instead of tick labels."""
    clean(ax, frame=True)


def grab(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(name, frames, fps=20, hold_last=0):
    OUT.mkdir(parents=True, exist_ok=True)
    frames = list(frames) + [frames[-1]] * hold_last
    # One shared palette keeps colours stable between frames and the file small.
    strip = Image.new("RGB", (frames[0].width, frames[0].height * min(len(frames), 8)))
    picks = np.linspace(0, len(frames) - 1, min(len(frames), 8)).astype(int)
    for k, i in enumerate(picks):
        strip.paste(frames[i], (0, k * frames[0].height))
    palette = strip.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    pal = palette.getpalette()[: 96 * 3]
    cols = np.array(pal).reshape(-1, 3)
    bg = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)])
    j = int(np.argmin(((cols - bg) ** 2).sum(axis=1)))
    pal[3 * j: 3 * j + 3] = bg.tolist()
    palette.putpalette(pal)
    quant = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path = OUT / f"{name}.gif"
    quant[0].save(path, save_all=True, append_images=quant[1:], duration=int(1000 / fps),
                  loop=0, optimize=True, disposal=1)
    print(f"{path.relative_to(OUT.parents[2])}: {len(frames)} frames, "
          f"{path.stat().st_size / 1024:.0f} KiB")


def ring_offsets(nbeads, radius, phase, rng_modes, dim=2):
    """Smooth closed-path bead displacements built from a few normal modes."""
    s = np.linspace(0, 2 * np.pi, nbeads, endpoint=False)
    out = np.zeros((nbeads, dim))
    out[:, 0] += radius * np.cos(s + phase)
    if dim > 1:
        out[:, 1] += radius * np.sin(s + phase)
    for k, (ax_, ay_, px, py) in enumerate(rng_modes, start=1):
        amp = 0.45 * radius / k
        out[:, 0] += amp * ax_ * np.cos(k * s + px + phase * (1 + 0.3 * k))
        if dim > 1:
            out[:, 1] += amp * ay_ * np.sin(k * s + py + phase * (1 + 0.25 * k))
    out -= out.mean(axis=0)
    return out


def draw_ring(ax, centre, offsets, color, bead_size=22, lw=1.4, alpha=1.0, z=5):
    pts = centre + offsets
    closed = np.vstack([pts, pts[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw, alpha=0.55 * alpha, zorder=z)
    ax.scatter(pts[:, 0], pts[:, 1], s=bead_size, color=color, edgecolor="white",
               linewidth=0.6, alpha=alpha, zorder=z + 1)
    ax.scatter([centre[0]], [centre[1]], s=bead_size * 0.45, color=INK, alpha=0.8 * alpha,
               zorder=z + 2, marker="x", linewidths=1.0)


def modes(rng, n=3):
    return [(rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0), rng.uniform(0, 6.3),
             rng.uniform(0, 6.3)) for _ in range(n)]


def axis_arrows(ax, x0, y0, x1, y1):
    for (a, b) in (((x0, y0), (x1, y0)), ((x0, y0), (x0, y1))):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=GRAY,
                                     lw=1.2, zorder=1))


# ------------------------------------------------------------------ 1. ring polymer


def muller_like(x, y):
    """Two basins joined by a curved saddle region."""
    return (1.2 * (x ** 2 - 1) ** 2 + 2.2 * (y - 0.35 * x ** 2 + 0.2) ** 2
            - 0.25 * np.exp(-((x + 1) ** 2 + (y + 0.0) ** 2) * 2)
            - 0.25 * np.exp(-((x - 1) ** 2 + (y + 0.0) ** 2) * 2))


def gif_ring_polymer():
    rng = np.random.default_rng(3)
    xs = np.linspace(-2.0, 2.0, 220)
    ys = np.linspace(-1.2, 1.6, 160)
    X, Y = np.meshgrid(xs, ys)
    V = muller_like(X, Y)
    n = 90
    t = np.linspace(0, 1, n, endpoint=False)
    # centroid: rests in A, crosses, rests in B, returns
    u = 0.5 - 0.5 * np.cos(2 * np.pi * t)
    cx = -1.0 + 2.0 * (u ** 1.0)
    cy = 0.35 * cx ** 2 - 0.2 + 0.05 * np.sin(6 * np.pi * t)
    m = modes(rng, 4)
    frames = []
    trail = []
    for i in range(n):
        fig, (ax,) = new_fig()
        ax.contourf(X, Y, V, levels=np.linspace(-0.3, 2.4, 14), cmap="Blues_r", alpha=0.55)
        ax.contour(X, Y, V, levels=np.linspace(-0.3, 2.4, 14), colors="white", linewidths=0.6)
        ax.scatter([0], [-0.2], marker="*", s=90, color=ORANGE, zorder=4, edgecolor="white")
        barrier = np.exp(-(cx[i] / 0.45) ** 2)
        radius = 0.16 + 0.22 * barrier  # the ring spreads (tunnels) at the barrier top
        off = ring_offsets(16, radius, 2 * np.pi * 3 * t[i], m)
        off[:, 0] *= 1 + 1.2 * barrier
        trail.append((cx[i], cy[i]))
        tr = np.array(trail[-25:])
        if len(tr) > 1:
            segs = np.stack([tr[:-1], tr[1:]], axis=1)
            lc = LineCollection(segs, colors=INK, linewidths=1.2,
                                alpha=np.linspace(0.05, 0.6, len(segs)))
            ax.add_collection(lc)
        draw_ring(ax, np.array([cx[i], cy[i]]), off, RED, bead_size=26)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.2, 1.3)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("ring_polymer", frames, fps=18)


# ------------------------------------------------------------------ 2. umbrella / PMF


def gif_umbrella_pmf():
    xs = np.linspace(-0.1, 1.1, 400)

    def pmf(x):
        return 1.6 * np.exp(-((x - 0.55) / 0.17) ** 2) - 0.6 * np.exp(-((x - 1.05) / 0.2) ** 2)

    centres = np.linspace(0.0, 1.0, 13)
    k = 260.0
    beta = 4.0
    frames = []
    per = 6
    rng = np.random.default_rng(1)
    hists = []
    for c in centres:
        p = np.exp(-beta * (pmf(xs) + 0.5 * k * (xs - c) ** 2))
        p /= p.max()
        hists.append(p)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(centres)))
    total = len(centres) * per + 25
    for f in range(total):
        w = min(f // per, len(centres) - 1)
        g = (f % per + 1) / per if f < len(centres) * per else 1.0
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
        ax1 = fig.add_axes([0.05, 0.50, 0.92, 0.46])
        ax2 = fig.add_axes([0.05, 0.05, 0.92, 0.40])
        clean(ax1)
        clean(ax2)
        axis_arrows(ax1, -0.08, -0.9, 1.1, 2.0)
        axis_arrows(ax2, -0.08, 0.0, 1.1, 1.2)
        ax1.plot(xs, pmf(xs), color=GRID, lw=5, zorder=2)
        done = w + 1 if f >= len(centres) * per else w
        for j in range(done + 1 if f < len(centres) * per else len(centres)):
            frac = 1.0 if j < done else g
            amp = frac * (0.95 + 0.05 * rng.random())
            ax2.fill_between(xs, 0, amp * hists[j], color=colors[j], alpha=0.55, lw=0)
            ax2.plot(xs, amp * hists[j], color=colors[j], lw=0.8)
        # current umbrella spring
        if f < len(centres) * per:
            c = centres[w]
            sel = np.abs(xs - c) < 0.11
            ax1.plot(xs[sel], pmf(c) + 0.5 * k * 0.3 * (xs[sel] - c) ** 2 - 0.05,
                     color=colors[w], lw=2, zorder=3)
            ax1.scatter([c], [pmf(c)], color=colors[w], s=40, zorder=5, edgecolor="white")
        # PMF reconstructed from finished windows
        cover = centres[min(done, len(centres) - 1)] + 0.05 if f < len(centres) * per else 1.1
        sel = xs <= cover
        ax1.plot(xs[sel], pmf(xs[sel]), color=INK, lw=2.2, zorder=4)
        ax1.set_xlim(-0.1, 1.12)
        ax1.set_ylim(-0.9, 2.1)
        ax2.set_xlim(-0.1, 1.12)
        ax2.set_ylim(-0.02, 1.25)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("umbrella_pmf", frames, fps=16)


# ------------------------------------------------------------- 3. recrossing / kappa


def gif_recrossing():
    rng = np.random.default_rng(7)

    def force(x, y):
        # V = (x^2-1)^2 + 3 (y - 0.6 x)^2 ... rotated valley gives recrossing
        fx = -(4 * x * (x ** 2 - 1)) + 3.6 * (y - 0.6 * x)
        fy = -6 * (y - 0.6 * x)
        return fx, fy

    def pot(x, y):
        return (x ** 2 - 1) ** 2 + 3 * (y - 0.6 * x) ** 2

    ntraj, nstep, dt, gamma, T = 400, 260, 0.02, 1.2, 0.35
    x = np.zeros(ntraj)
    y = rng.normal(0, np.sqrt(T / 6), ntraj)
    vx = np.abs(rng.normal(0, np.sqrt(T), ntraj)) * rng.choice([-1, 1], ntraj)
    vy = rng.normal(0, np.sqrt(T), ntraj)
    v0 = vx.copy()
    path = np.zeros((nstep, ntraj, 2))
    for s in range(nstep):
        path[s, :, 0], path[s, :, 1] = x, y
        fx, fy = force(x, y)
        vx += dt * fx - gamma * vx * dt + np.sqrt(2 * gamma * T * dt) * rng.normal(size=ntraj)
        vy += dt * fy - gamma * vy * dt + np.sqrt(2 * gamma * T * dt) * rng.normal(size=ntraj)
        x += dt * vx
        y += dt * vy
    side = (path[:, :, 0] > 0).astype(float)
    kappa = (v0[None, :] * side).mean(axis=1) / (v0 * (v0 > 0)).mean()
    tgrid = np.arange(nstep) * dt
    show = rng.choice(ntraj, 14, replace=False)
    xs = np.linspace(-1.8, 1.8, 200)
    ys = np.linspace(-1.6, 1.6, 160)
    X, Y = np.meshgrid(xs, ys)
    Vg = pot(X, Y)
    frames = []
    steps = list(range(2, nstep, 4))
    for s in steps:
        fig, (ax, axk) = new_fig(2, [1.15, 1])
        ax.contourf(X, Y, Vg, levels=np.linspace(0, 3, 13), cmap="Blues_r", alpha=0.5)
        ax.contour(X, Y, Vg, levels=np.linspace(0, 3, 13), colors="white", linewidths=0.5)
        ax.plot([0, 0], [-1.6, 1.6], color=ORANGE, lw=2, ls=(0, (4, 3)), zorder=3)
        for j in show:
            p = path[: s + 1, j]
            col = TEAL if p[-1, 0] > 0 else PURPLE
            ax.plot(p[:, 0], p[:, 1], color=col, lw=1.0, alpha=0.75, zorder=4)
            ax.scatter(p[-1, 0], p[-1, 1], s=10, color=col, zorder=5)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.6, 1.6)
        clean(axk)
        axis_arrows(axk, 0, 0, tgrid[-1] * 1.05, 1.1)
        axk.axhline(kappa[-60:].mean(), color=ORANGE, lw=1.2, ls=(0, (4, 3)))
        axk.plot(tgrid[: s + 1], kappa[: s + 1], color=INK, lw=2.2)
        axk.scatter(tgrid[s], kappa[s], color=INK, s=18, zorder=5)
        axk.set_xlim(-0.1, tgrid[-1] * 1.08)
        axk.set_ylim(-0.05, 1.15)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("recrossing", frames, fps=16, hold_last=20)


# ------------------------------------------------------------------ 4. dRPMD collision


def gif_drpmd():
    rng = np.random.default_rng(5)
    n = 110
    t = np.linspace(0, 1, n)
    # collinear-ish A + BC -> AB + C ; scripted smooth centroid paths
    # A approaches BC; after the exchange AB leaves to the upper left and C to the right.
    A = np.zeros((n, 2))
    vib = 0.12 * np.sin(2 * np.pi * 6 * t)
    B = np.zeros((n, 2))
    C = np.zeros((n, 2))
    for i, ti in enumerate(t):
        if ti < 0.5:
            A[i] = [-3.2 + 2.0 * ti / 0.5 * 1.25, 0.9 - 0.9 * ti / 0.5]
            B[i] = [0.0 - vib[i] / 2, 0.0]
            C[i] = [0.9 + vib[i] / 2, 0.0]
        else:
            d = (ti - 0.5) / 0.5
            com = np.array([-0.4, 0.0]) + d * np.array([-2.4, 0.9])
            B[i] = com + np.array([0.45 + vib[i] / 2, 0.0])
            A[i] = com - np.array([0.45 + vib[i] / 2, 0.0])
            C[i] = np.array([0.9, 0.0]) + d * np.array([2.6, -0.4])
    def smooth(P):  # soften the join at the exchange
        k = np.ones(7) / 7
        out = P.copy()
        for dcol in range(2):
            out[:, dcol] = np.convolve(np.pad(P[:, dcol], 3, mode="edge"), k, mode="valid")
        return out

    A, B, C = smooth(A), smooth(B), smooth(C)
    mA, mB, mC = modes(rng), modes(rng), modes(rng)
    frames = []
    for i in range(n):
        fig, (ax,) = new_fig()
        ph = 2 * np.pi * 4 * t[i]
        for P, Q in ((A, B), (B, C)):
            d = np.linalg.norm(P[i] - Q[i])
            a = float(np.clip((1.4 - d) / 0.5, 0, 1))
            if a > 0:
                ax.plot([P[i, 0], Q[i, 0]], [P[i, 1], Q[i, 1]], color=INK, lw=3, alpha=0.35 * a,
                        solid_capstyle="round", zorder=2)
        for P, m, col, r in ((A, mA, RED, 0.22), (B, mB, BLUE, 0.22), (C, mC, TEAL, 0.22)):
            draw_ring(ax, P[i], ring_offsets(12, r, ph, m), col, bead_size=24)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-1.5, 1.6)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("drpmd_collision", frames, fps=20, hold_last=10)


# ------------------------------------------------------------------ 5. metadynamics 1D


def gif_metadynamics():
    rng = np.random.default_rng(11)
    xs = np.linspace(-2.2, 2.2, 400)

    def F(x):
        return (1.29 * (x ** 2 - 1.6) ** 2 + 0.4 * np.exp(-((x + 0.2) / 0.3) ** 2)
                - 0.5 * np.exp(-((x - 1.3) / 0.3) ** 2))

    def dF(x, h=1e-4):
        return (F(x + h) - F(x - h)) / (2 * h)

    w0, sig, dT, kT = 0.12, 0.18, 6.0, 0.25
    centres, heights = [], []

    def bias(x):
        if not centres:
            return np.zeros_like(x)
        c = np.array(centres)[:, None]
        h = np.array(heights)[:, None]
        return (h * np.exp(-0.5 * ((np.atleast_1d(x)[None, :] - c) / sig) ** 2)).sum(axis=0)

    def dbias(x, e=1e-4):
        return (bias(np.array([x + e])) - bias(np.array([x - e])))[0] / (2 * e)

    x, dt = -1.26, 0.002
    frames = []
    trail = []
    for f in range(150):
        for _ in range(40):
            x += -(dF(x) + dbias(x)) * dt + np.sqrt(2 * kT * dt) * rng.normal()
            x = float(np.clip(x, -2.1, 2.1))
        hb = w0 * np.exp(-bias(np.array([x]))[0] / (kT * dT))
        centres.append(x)
        heights.append(hb)
        trail.append(x)
        if f % 2:
            continue
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
        ax = fig.add_axes([0.03, 0.40, 0.94, 0.58])
        al = fig.add_axes([0.03, 0.02, 0.94, 0.34])
        clean(ax)
        clean(al)
        b = bias(xs)
        base = F(xs)
        ax.fill_between(xs, base, base + b, color=ORANGE, alpha=0.45, lw=0, zorder=2)
        ax.plot(xs, base, color=GRAY, lw=2, zorder=3)
        ax.plot(xs, base + b, color=ORANGE, lw=1.6, zorder=3)
        est = -b * (dT + 1) / dT
        seen = b > 0.03
        al.plot(xs, base, color=GRID, lw=5, zorder=2)
        if seen.any():
            est = est - est[seen].min() + base.min()
            al.plot(xs[seen], est[seen], color=INK, lw=2.0, zorder=3, marker="none")
        al.set_xlim(-2.2, 2.2)
        al.set_ylim(base.min() - 0.3, 3.2)
        yx = F(x) + bias(np.array([x]))[0]
        ax.scatter([x], [yx + 0.08], s=70, color=RED, zorder=6, edgecolor="white")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(base.min() - 0.3, 4.0)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("metadynamics", frames, fps=15, hold_last=15)


# ------------------------------------------------------------- 6. 2D enhanced sampling


def gif_fes2d():
    rng = np.random.default_rng(2)

    def F(x, y):
        return (-2.0 * np.exp(-((x + 1.0) ** 2 + (y + 0.6) ** 2) / 0.35)
                - 1.6 * np.exp(-((x - 1.0) ** 2 + (y - 0.7) ** 2) / 0.35)
                - 1.3 * np.exp(-((x - 0.9) ** 2 + (y + 0.9) ** 2) / 0.30)
                + 0.06 * (x ** 2 + y ** 2) ** 2)

    xs = np.linspace(-2, 2, 120)
    X, Y = np.meshgrid(xs, xs)
    Fg = F(X, Y)
    Fg -= Fg.min()
    sig, w0, dT, dt = 0.22, 0.08, 8.0, 0.002
    C = []
    Hh = []
    Bg = np.zeros_like(X)

    def grad_total(x, y, e=1e-4):
        def tot(a, b):
            v = F(a, b)
            if C:
                c = np.array(C)
                v += (np.array(Hh) * np.exp(-((a - c[:, 0]) ** 2 + (b - c[:, 1]) ** 2) / (2 * sig ** 2))).sum()
            return v
        return ((tot(x + e, y) - tot(x - e, y)) / (2 * e), (tot(x, y + e) - tot(x, y - e)) / (2 * e))

    x, y = -1.0, -0.6
    kT = 0.4
    pts = []
    frames = []
    levels = np.linspace(0, 2.6, 14)
    for f in range(240):
        for _ in range(25):
            gx, gy = grad_total(x, y)
            x += -gx * dt + np.sqrt(2 * kT * dt) * rng.normal()
            y += -gy * dt + np.sqrt(2 * kT * dt) * rng.normal()
            x, y = float(np.clip(x, -1.9, 1.9)), float(np.clip(y, -1.9, 1.9))
            pts.append((x, y))
        bnow = 0.0
        if C:
            c = np.array(C)
            bnow = (np.array(Hh) * np.exp(-((x - c[:, 0]) ** 2 + (y - c[:, 1]) ** 2) / (2 * sig ** 2))).sum()
        h = w0 * np.exp(-bnow / (kT * dT))
        C.append((x, y))
        Hh.append(h)
        Bg += h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sig ** 2))
        if f % 3:
            continue
        fig, (a1, a2) = new_fig(2, [1, 1])
        a1.contourf(X, Y, Fg, levels=levels, cmap="Blues_r", extend="max")
        a1.contour(X, Y, Fg, levels=levels, colors="white", linewidths=0.4)
        P = np.array(pts[-600:])
        a1.plot(P[:, 0], P[:, 1], color=ORANGE, lw=0.7, alpha=0.8)
        a1.scatter([x], [y], s=40, color=RED, edgecolor="white", zorder=5)
        est = -Bg * (dT + 1) / dT
        est -= est.min()
        visited = Bg > 0.02
        est_m = np.where(visited, np.minimum(est, levels[-1]), np.nan)
        a2.set_facecolor(LIGHT)
        a2.contourf(X, Y, est_m, levels=levels, cmap="Blues_r")
        for a in (a1, a2):
            a.set_xlim(-2, 2)
            a.set_ylim(-2, 2)
            a.set_aspect("equal")
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("fes_2d", frames, fps=14, hold_last=15)


# ------------------------------------------------------------------ 7. PIMD / NQE


def gif_pimd():
    rng = np.random.default_rng(4)
    n = 100
    t = np.linspace(0, 1, n, endpoint=False)
    temp = 0.5 + 0.5 * np.cos(2 * np.pi * t)  # 1 = hot, 0 = cold
    O = np.array([0.0, 0.0])
    ang = np.deg2rad(52)
    H1 = np.array([-np.sin(ang), -np.cos(ang)]) * 1.15
    H2 = np.array([np.sin(ang), -np.cos(ang)]) * 1.15
    m1, m2, mo = modes(rng, 4), modes(rng, 4), modes(rng, 4)
    cmap = plt.cm.coolwarm
    frames = []
    for i in range(n):
        fig, (ax,) = new_fig()
        cold = 1 - temp[i]
        rH = 0.08 + 0.30 * cold ** 1.3
        rO = 0.03 + 0.07 * cold ** 1.3
        ph = 2 * np.pi * 5 * t[i]
        vib = 0.06 * np.sin(2 * np.pi * 9 * t[i])
        h1 = H1 * (1 + vib)
        h2 = H2 * (1 - vib)
        for h in (h1, h2):
            ax.plot([O[0], h[0]], [O[1], h[1]], color=INK, lw=4, alpha=0.25, zorder=1,
                    solid_capstyle="round")
        draw_ring(ax, O, ring_offsets(24, rO, ph, mo), RED, bead_size=46)
        draw_ring(ax, h1, ring_offsets(24, rH, ph, m1), BLUE, bead_size=22)
        draw_ring(ax, h2, ring_offsets(24, rH, ph + 1.3, m2), BLUE, bead_size=22)
        # thermometer
        ax.add_patch(Rectangle((2.15, -1.6), 0.18, 2.5, facecolor=LIGHT, edgecolor=GRAY,
                               lw=1.0, zorder=1))
        ax.add_patch(Rectangle((2.15, -1.6), 0.18, 2.5 * (0.08 + 0.92 * temp[i]),
                               facecolor=cmap(0.15 + 0.8 * temp[i]), lw=0, zorder=2))
        ax.add_patch(Circle((2.24, -1.75), 0.2, facecolor=cmap(0.15 + 0.8 * temp[i]),
                            edgecolor=GRAY, lw=1.0, zorder=3))
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.15, 1.25)
        ax.set_aspect("equal")
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("pimd_nqe", frames, fps=18)


# ------------------------------------------------------------------ 8. diffusion


def gif_diffusion():
    rng = np.random.default_rng(9)
    L = 6.0
    npart, dt, kT, nstep, every = 6, 0.004, 0.32, 4400, 40
    amp = 1.0

    def force(p):
        k = 2 * np.pi
        return -0.5 * amp * k * np.sin(k * p)

    # many tracer copies for the MSD curve
    ntr = 300
    pos = rng.uniform(0, L, (ntr, 2))
    pos = np.round(pos) + 0.0
    unwrapped = pos.copy()
    start = pos.copy()
    msd = []
    snaps = []
    for s in range(nstep):
        dr = force(pos) * dt + np.sqrt(2 * kT * dt) * rng.normal(size=pos.shape)
        pos = (pos + dr) % L
        unwrapped += dr
        if s % every == 0:
            msd.append(((unwrapped - start) ** 2).sum(axis=1).mean())
            snaps.append(unwrapped[:npart].copy())
    msd = np.array(msd)
    tt = np.arange(len(msd))
    host = np.array([(i + 0.5, j + 0.5) for i in range(int(L)) for j in range(int(L))])
    ms = [modes(rng) for _ in range(npart)]
    cols = plt.cm.tab10(np.arange(npart))
    frames = []
    for f in range(len(snaps)):
        fig, (ax, am) = new_fig(2, [1.25, 1])
        ax.scatter(host[:, 0], host[:, 1], s=70, color=GRAY, alpha=0.55, zorder=1, edgecolor="white")
        ax.add_patch(Rectangle((0, 0), L, L, fill=False, edgecolor=GRAY, lw=1.0, ls=(0, (3, 3))))
        for p in range(npart):
            hist = np.array([s_[p] for s_ in snaps[max(0, f - 14): f + 1]])
            hw = hist % L
            # break the trail at periodic jumps
            jump = np.r_[False, np.any(np.abs(np.diff(hw, axis=0)) > L / 2, axis=1)]
            seg = []
            for k_ in range(len(hw)):
                if jump[k_] and seg:
                    ax.plot(*np.array(seg).T, color=cols[p], lw=1.0, alpha=0.5)
                    seg = []
                seg.append(hw[k_])
            if len(seg) > 1:
                ax.plot(*np.array(seg).T, color=cols[p], lw=1.0, alpha=0.5)
            draw_ring(ax, hw[-1], ring_offsets(10, 0.17, f * 0.6, ms[p]), cols[p], bead_size=16,
                      lw=1.2)
        ax.set_xlim(-0.2, L + 0.2)
        ax.set_ylim(-0.2, L + 0.2)
        ax.set_aspect("equal")
        clean(am)
        axis_arrows(am, 0, 0, tt[-1] * 1.05, msd.max() * 1.08)
        am.plot(tt[: f + 1], msd[: f + 1], color=INK, lw=2)
        slope = np.polyfit(tt[10:], msd[10:], 1)
        am.plot(tt, np.polyval(slope, tt), color=ORANGE, lw=1.0, ls=(0, (4, 3)), alpha=0.8)
        am.set_xlim(-2, tt[-1] * 1.1)
        am.set_ylim(-0.5, msd.max() * 1.15)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("diffusion", frames, fps=16, hold_last=10)


# ------------------------------------------------------------------ 9. surface rate


def gif_surface():
    rng = np.random.default_rng(21)
    a = 1.0

    def V(x):
        return 0.5 * (1 - np.cos(2 * np.pi * x / a))

    # many independent escapes from the central basin -> survival curve
    nt, dt, kT, gamma = 600, 0.01, 0.3, 1.5
    x = rng.normal(0, 0.05, nt)
    v = rng.normal(0, np.sqrt(kT), nt)
    alive = np.ones(nt, bool)
    surv = []
    for s in range(1600):
        f = -0.5 * (2 * np.pi / a) * np.sin(2 * np.pi * x / a)
        v += f * dt - gamma * v * dt + np.sqrt(2 * gamma * kT * dt) * rng.normal(size=nt)
        x += v * dt
        alive &= np.abs(x) < 0.5 * a + 0.12
        if s % 16 == 0:
            surv.append(alive.mean())
    surv = np.array(surv)
    # displayed walker: thermal rattling inside a site, rare quick hops between sites
    nfr = len(surv)
    hops = [(22, 0, 1), (48, 1, 2), (66, 2, 1), (88, 1, 0)]  # (frame, from, to)
    site = np.zeros(nfr)
    for fh, s0, s1 in hops:
        u = np.clip((np.arange(nfr) - fh) / 5, 0, 1)
        site += (s1 - s0) * (3 * u ** 2 - 2 * u ** 3)
    jig = np.convolve(rng.normal(0, 0.12, nfr), np.ones(5) / 5, mode="same")
    disp = site * a + jig
    m = modes(rng)
    frames = []
    for f in range(len(surv)):
        fig, (ax, ak) = new_fig(2, [1.5, 1])
        # slab: three layers, bottom fixed (darker, hatched)
        for layer, (yy, col) in enumerate(((-0.75, "#c9d3e3"), (-1.55, "#b4c0d6"), (-2.35, "#8e9bb5"))):
            shift = 0.5 * a * (layer % 2)
            for xs_ in np.arange(-3.5, 4.0, a) + shift:
                ax.add_patch(Circle((xs_, yy), 0.38, facecolor=col, edgecolor="white", lw=0.8,
                                    hatch="////" if layer == 2 else None, zorder=1))
        gx = np.linspace(-3.2, 3.2, 300)
        ax.plot(gx, 0.25 + 0.35 * V(gx), color=GRID, lw=1.5, zorder=0)
        ax.add_patch(Rectangle((-0.5 * a, -0.3), a, 1.5, color=ORANGE, alpha=0.12, lw=0))
        xa = disp[f]
        ya = 0.35 + 0.35 * V(xa)
        draw_ring(ax, np.array([xa, ya]), ring_offsets(12, 0.2, f * 0.7, m), RED, bead_size=22)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.85, 1.25)
        ax.set_aspect("equal")
        clean(ak)
        tt = np.arange(len(surv))
        axis_arrows(ak, 0, 0, tt[-1] * 1.05, 1.1)
        good = surv > 0.02
        rate = -np.polyfit(tt[good][5:], np.log(surv[good][5:]), 1)[0]
        ak.plot(tt, np.exp(-rate * tt), color=ORANGE, lw=1.2, ls=(0, (4, 3)))
        ak.plot(tt[: f + 1], surv[: f + 1], color=INK, lw=2)
        ak.set_xlim(-2, tt[-1] * 1.1)
        ak.set_ylim(-0.05, 1.15)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("surface_rate", frames, fps=16, hold_last=10)


# ------------------------------------------------------------------ 10. PCET


def gif_pcet():
    q = np.linspace(-2.4, 2.4, 300)
    lam = 1.0
    etas = np.concatenate([np.linspace(-1.4, 1.4, 60), np.linspace(1.4, -1.4, 60)])

    def lnk(eta, iso):
        # Marcus-like with vibronic offsets; D gives a lower, shifted curve
        tot = 0.0
        w = 1.0 if iso == "H" else 0.35
        for nu, hw in enumerate([0.0, 0.55, 1.1]):
            dg = -eta + hw
            tot += w * np.exp(-nu * 0.9) * np.exp(-(dg + lam) ** 2 / (4 * lam * 0.25))
        return np.log(tot + 1e-12)

    eta_grid = np.linspace(-1.4, 1.4, 200)
    kH = np.array([lnk(e, "H") for e in eta_grid])
    kD = np.array([lnk(e, "D") for e in eta_grid])
    lo = min(kH.min(), kD.min())
    frames = []
    for i, eta in enumerate(etas):
        fig, (ax, ar) = new_fig(2, [1.2, 1])
        GR = lam * (q + 1) ** 2 / 4
        GP = lam * (q - 1) ** 2 / 4 - eta
        ax.plot(q, GR, color=BLUE, lw=2.4)
        ax.plot(q, GP, color=ORANGE, lw=2.4)
        for nu, hw in enumerate([0.55, 1.1]):
            ax.plot(q, GP + hw, color=ORANGE, lw=1.0, alpha=0.45 - 0.15 * nu, ls=(0, (3, 2)))
        # crossing point
        qc = -eta / lam
        ax.scatter([qc], [lam * (qc + 1) ** 2 / 4], s=55, color=RED, zorder=5, edgecolor="white")
        # electrode (left) with charge level
        ax.add_patch(Rectangle((-2.4, -1.8), 0.3, 4.4, facecolor="#c7ced9", edgecolor="none"))
        ax.plot([-2.1, -1.85], [eta * 0.8, eta * 0.8], color=PURPLE, lw=3)
        ax.set_xlim(-2.45, 2.4)
        ax.set_ylim(-1.8, 2.6)
        clean(ar)
        axis_arrows(ar, -1.45, lo - 0.3, 1.5, kH.max() + 0.6)
        sel = eta_grid <= eta if i < 60 else np.ones_like(eta_grid, bool)
        ar.plot(eta_grid[sel], kH[sel], color=BLUE, lw=2.2)
        ar.plot(eta_grid[sel], kD[sel], color=TEAL, lw=2.2, ls=(0, (5, 2)))
        ar.axvline(eta, color=PURPLE, lw=1.0, alpha=0.6)
        ar.scatter([eta, eta], [lnk(eta, "H"), lnk(eta, "D")], s=22, color=[BLUE, TEAL], zorder=5)
        ar.set_xlim(-1.5, 1.6)
        ar.set_ylim(lo - 0.4, kH.max() + 0.8)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("pcet", frames, fps=20)


# ------------------------------------------------------------------ 11. nonadiabatic


def gif_nonadiabatic():
    """Fewest-switches surface hopping on Tully's simple avoided crossing."""
    rng = np.random.default_rng(8)
    A, B, Cc, D = 0.01, 1.6, 0.005, 1.0
    mass = 2000.0

    def H(x):
        v11 = np.sign(x) * A * (1 - np.exp(-B * np.abs(x)))
        v12 = Cc * np.exp(-D * x ** 2)
        return v11, v12

    def adiabats(x):
        v11, v12 = H(x)
        e = np.sqrt(v11 ** 2 + v12 ** 2)
        return -e, e

    def dE(x, h=1e-4):
        lo1, up1 = adiabats(x + h)
        lo0, up0 = adiabats(x - h)
        return (lo1 - lo0) / (2 * h), (up1 - up0) / (2 * h)

    def nac(x, h=1e-4):
        v11, v12 = H(x)
        th = 0.5 * np.arctan2(v12, v11)
        v11b, v12b = H(x + h)
        thb = 0.5 * np.arctan2(v12b, v11b)
        return (thb - th) / h

    ntraj = 80
    k0 = 16.0
    x = rng.normal(-6.0, 0.6, ntraj)
    p = rng.normal(k0, 0.6, ntraj)
    state = np.ones(ntraj, int)  # start on upper surface
    c = np.zeros((ntraj, 2), complex)
    c[:, 1] = 1.0
    dt = 4.0
    nsteps = 2000
    snaps = []
    pops = []
    for s in range(nsteps):
        g_lo, g_up = dE(x)
        F = -np.where(state == 1, g_up, g_lo)
        p += F * dt
        x += p / mass * dt
        lo, up = adiabats(x)
        d = nac(x)
        v = p / mass
        # electronic propagation in adiabatic basis
        Hm = np.zeros((ntraj, 2, 2), complex)
        Hm[:, 0, 0] = lo
        Hm[:, 1, 1] = up
        Hm[:, 0, 1] = -1j * v * (-d)
        Hm[:, 1, 0] = -1j * v * d
        for k in range(ntraj):
            w, U = np.linalg.eigh(Hm[k])
            c[k] = U @ (np.exp(-1j * w * dt) * (U.conj().T @ c[k]))
        # hopping
        for k in range(ntraj):
            a, b = state[k], 1 - state[k]
            bab = -2 * np.real(np.conj(c[k, a]) * c[k, b] * v[k] * (d[k] if b == 1 else -d[k]))
            prob = dt * bab / max(abs(c[k, a]) ** 2, 1e-12)
            if rng.random() < prob:
                gap = (up[k] - lo[k]) * (1 if b == 1 else -1)
                ke = p[k] ** 2 / (2 * mass)
                if ke > gap:
                    p[k] = np.sign(p[k]) * np.sqrt(2 * mass * (ke - gap))
                    state[k] = b
        if np.median(x) > 6.5:
            break
        if s % 5 == 0:
            snaps.append((x.copy(), state.copy()))
            pops.append((state == 1).mean())
    xs = np.linspace(-8, 8, 400)
    lo_g, up_g = adiabats(xs)
    v11, _ = H(xs)
    frames = []
    pops = np.array(pops)
    for f, (xx, st) in enumerate(snaps):
        fig, (ax, ap) = new_fig(2, [1.5, 1])
        ax.plot(xs, v11, color=GRID, lw=1, ls=(0, (3, 3)))
        ax.plot(xs, -v11, color=GRID, lw=1, ls=(0, (3, 3)))
        ax.plot(xs, up_g, color=ORANGE, lw=2.4)
        ax.plot(xs, lo_g, color=BLUE, lw=2.4)
        lo_x, up_x = adiabats(xx)
        yy = np.where(st == 1, up_x, lo_x) + rng.normal(0, 0.0004, len(xx))
        cc = np.where(st == 1, ORANGE, BLUE)
        ax.scatter(xx, yy, s=16, c=cc, edgecolor=INK, linewidth=0.3, alpha=0.85, zorder=4)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-0.0135, 0.0135)
        clean(ap)
        tt = np.arange(len(pops))
        axis_arrows(ap, 0, 0, tt[-1] * 1.05, 1.1)
        ap.plot(tt[: f + 1], pops[: f + 1], color=ORANGE, lw=2.2)
        ap.plot(tt[: f + 1], 1 - pops[: f + 1], color=BLUE, lw=2.2)
        ap.set_xlim(-2, tt[-1] * 1.1)
        ap.set_ylim(-0.05, 1.15)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("nonadiabatic", frames, fps=16, hold_last=12)


# ------------------------------------------------------------------ 12. spectroscopy


def gif_spectroscopy():
    rng = np.random.default_rng(12)
    dt = 0.05
    nt = 1600
    t = np.arange(nt) * dt
    w1, w2, w3 = 1.6, 3.7, 3.9
    # dipole correlation function: sum of damped cosines with noise
    cf = (1.0 * np.cos(w1 * t) * np.exp(-t / 18) + 0.6 * np.cos(w2 * t) * np.exp(-t / 14)
          + 0.45 * np.cos(w3 * t) * np.exp(-t / 14))
    cf += 0.04 * rng.normal(size=nt) * np.exp(-t / 40)
    omega = np.linspace(0.5, 5.0, 300)
    m = modes(rng)
    ang = np.deg2rad(52)
    frames = []
    nf = 80
    for f in range(nf):
        upto = int(40 + (nt - 40) * (f / (nf - 1)) ** 1.5)
        tt = t[:upto]
        win = np.cos(0.5 * np.pi * tt / tt[-1]) ** 2
        spec = np.array([np.sum(cf[:upto] * win * np.cos(w * tt)) * dt for w in omega])
        spec = np.clip(spec, 0, None) * omega ** 2
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
        am = fig.add_axes([0.01, 0.30, 0.30, 0.66])
        ac = fig.add_axes([0.36, 0.56, 0.61, 0.40])
        asp = fig.add_axes([0.36, 0.06, 0.61, 0.44])
        for a in (am, ac, asp):
            clean(a)
        # vibrating molecule (symmetric/asymmetric stretch + bend)
        ph = f * 0.9
        s1 = 0.10 * np.sin(ph)
        s2 = 0.10 * np.sin(2.3 * ph)
        b = 0.12 * np.sin(1.0 * ph * 0.5)
        h1 = np.array([-np.sin(ang + b), -np.cos(ang + b)]) * (1.0 + s1 + s2)
        h2 = np.array([np.sin(ang + b), -np.cos(ang + b)]) * (1.0 + s1 - s2)
        for h in (h1, h2):
            am.plot([0, h[0]], [0, h[1]], color=INK, lw=4, alpha=0.25, solid_capstyle="round")
        draw_ring(am, np.zeros(2), ring_offsets(12, 0.06, ph, m), RED, bead_size=30)
        draw_ring(am, h1, ring_offsets(12, 0.16, ph, m), BLUE, bead_size=14)
        draw_ring(am, h2, ring_offsets(12, 0.16, ph + 1, m), BLUE, bead_size=14)
        mu = (h1 + h2) * 0.5
        tip = -mu * 1.3
        am.add_patch(FancyArrowPatch((0, 0), (tip[0], tip[1]), arrowstyle="-|>",
                                     mutation_scale=12, color=ORANGE, lw=2, zorder=8))
        am.set_xlim(-1.4, 1.4)
        am.set_ylim(-1.5, 1.2)
        am.set_aspect("equal")
        axis_arrows(ac, 0, -1.3, t[-1] * 1.03, 1.3)
        ac.axhline(0, color=GRID, lw=0.8)
        ac.plot(tt, cf[:upto], color=INK, lw=0.9)
        ac.set_xlim(-1, t[-1] * 1.05)
        ac.set_ylim(-1.6, 2.0)
        axis_arrows(asp, 0.5, 0, 5.1, 1.1)
        top = max(spec.max(), 1e-9)
        asp.fill_between(omega, 0, spec / top, color=TEAL, alpha=0.35, lw=0)
        asp.plot(omega, spec / top, color=TEAL, lw=2)
        asp.set_xlim(0.4, 5.2)
        asp.set_ylim(-0.03, 1.2)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("spectroscopy", frames, fps=14, hold_last=20)


# ------------------------------------------------------------------ 13. instanton


def gif_instanton():
    V0, a = 1.0, 1.0

    def V(x):
        return V0 / np.cosh(x / a) ** 2

    def dV(x):
        return -2 * V0 / a * np.tanh(x / a) / np.cosh(x / a) ** 2

    def orbit(xt, nb=32):
        # periodic orbit in the inverted potential, started at turning point xt
        dt = 1e-3
        x, v, ts, xsx = xt, 0.0, [0.0], [xt]
        while True:
            v += dV(x) * dt  # m x'' = +V'(x) in imaginary time
            x += v * dt
            ts.append(ts[-1] + dt)
            xsx.append(x)
            if x <= -xt:
                break
        half = np.array(xsx)
        tau = np.array(ts)
        full_t = np.concatenate([tau, tau[-1] + tau[1:]])
        full_x = np.concatenate([half, half[::-1][1:]])
        grid = np.linspace(0, full_t[-1], nb, endpoint=False)
        return np.interp(grid, full_t, full_x), full_t[-1]

    turns = np.concatenate([np.linspace(0.08, 1.35, 45), np.linspace(1.35, 0.08, 45)])
    xs = np.linspace(-3, 3, 300)
    frames = []
    for xt in turns:
        beads, period = orbit(xt)
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
        a1 = fig.add_axes([0.04, 0.42, 0.92, 0.55])
        a2 = fig.add_axes([0.04, 0.03, 0.92, 0.36])
        clean(a1)
        clean(a2)
        a1.plot(xs, V(xs), color=INK, lw=2.4)
        a1.fill_between(xs, 0, V(xs), color=LIGHT, lw=0)
        E = V(xt)
        a1.plot([-xt, xt], [E, E], color=ORANGE, lw=1.5, ls=(0, (4, 3)))
        cols = plt.cm.twilight(np.linspace(0, 1, len(beads), endpoint=False))
        a1.scatter(beads, V(beads) + 0.04, s=28, c=cols, edgecolor="white", linewidth=0.5,
                   zorder=5)
        a1.set_xlim(-3, 3)
        a1.set_ylim(-0.05, 1.25)
        # imaginary-time ring: x(tau) wrapped on a circle -> closed loop
        th = np.linspace(0, 2 * np.pi, len(beads), endpoint=False)
        rr = 0.35
        px = beads
        py = rr * np.sin(th)
        a2.plot(np.r_[px, px[:1]], np.r_[py, py[:1]], color=GRAY, lw=1.2)
        a2.scatter(px, py, s=28, c=cols, edgecolor="white", linewidth=0.5, zorder=5)
        a2.axvline(0, color=GRID, lw=1)
        a2.set_xlim(-3, 3)
        a2.set_ylim(-0.5, 0.5)
        frames.append(grab(fig))
        plt.close(fig)
    save_gif("instanton", frames, fps=18)


ALL = {
    "ring_polymer": gif_ring_polymer,
    "umbrella_pmf": gif_umbrella_pmf,
    "recrossing": gif_recrossing,
    "drpmd": gif_drpmd,
    "metadynamics": gif_metadynamics,
    "fes_2d": gif_fes2d,
    "pimd": gif_pimd,
    "diffusion": gif_diffusion,
    "surface": gif_surface,
    "pcet": gif_pcet,
    "nonadiabatic": gif_nonadiabatic,
    "spectroscopy": gif_spectroscopy,
    "instanton": gif_instanton,
}


if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for name in names:
        ALL[name]()
