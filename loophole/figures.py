"""Figures for the write-up (matplotlib, static PNG).

Palette: the dataviz reference instance (light mode). Series colours follow a
fixed slot order; text stays in ink colours; grids are recessive.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .levitator import Levitator, plate_clearance, simulate

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
MAGNET = "#9c9a92"
GRAPHITE = "#3b3a37"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2.0, "legend.frameon": False,
    })


def earnshaw_landscape(lev: Levitator, path: str) -> None:
    """Axial energy with and without the graphite: saddle vs. well."""
    _style()
    bare = replace(lev, chi_perp=0.0, chi_par=0.0)
    zmax = 0.97 * lev.gap
    zs = np.linspace(-zmax, zmax, 121)
    u_bare = np.array([bare.axial_energy(z) for z in zs])
    u_full = np.array([lev.axial_energy(z) for z in zs])
    u_bare -= bare.axial_energy(0.0)
    u_full -= lev.axial_energy(0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    zmm = zs * 1e3
    ax.plot(zmm, u_bare * 1e6, color=S2)
    ax.plot(zmm, u_full * 1e6, color=S1)
    ax.axvspan(-lev.gap * 1e3 * 1.3, -lev.gap * 1e3, color=GRAPHITE, alpha=0.15, lw=0)
    ax.axvspan(lev.gap * 1e3, lev.gap * 1e3 * 1.3, color=GRAPHITE, alpha=0.15, lw=0)
    ax.set_xlim(-lev.gap * 1e3 * 1.3, lev.gap * 1e3 * 1.3)
    i = int(0.75 * len(zs))
    ax.annotate("magnets only: energy hill\n(Earnshaw: unstable)", (zmm[i], u_bare[i] * 1e6),
                xytext=(-150, -30), textcoords="offset points", color=INK_2, fontsize=9.5)
    ax.annotate("+ pyrolytic graphite: energy well\n(stable levitation)", (zmm[i], u_full[i] * 1e6),
                xytext=(-190, 6), textcoords="offset points", color=INK_2, fontsize=9.5)
    ax.text(-lev.gap * 1e3 * 1.15, ax.get_ylim()[1] * 0.9, "plate", ha="center", color=INK_2, fontsize=9)
    ax.text(lev.gap * 1e3 * 1.15, ax.get_ylim()[1] * 0.9, "plate", ha="center", color=INK_2, fontsize=9)
    ax.set_xlabel("vertical displacement of the floater from centre [mm]")
    ax.set_ylabel("potential energy [µJ]")
    ax.set_title("Earnshaw's theorem, and the loophole")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def field_cross_section(lev: Levitator, path: str) -> None:
    """Field lines in the x-z plane with the parts drawn to scale."""
    _style()
    top = lev.lifter_height + lev.lifter.length / 2 + 0.25 * lev.lifter.length
    bottom = -(lev.plate_face + lev.plate_thickness) - 0.4 * lev.lifter.length
    half_w = max(lev.lifter.radius * 1.8, 0.35 * (top - bottom))
    xs = np.linspace(-half_w, half_w, 220)
    zs = np.linspace(bottom, top, 420)
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X, np.zeros_like(X), Z], axis=-1)
    B = lev.lifter.field(P, center=(0, 0, lev.lifter_height)) + lev.floater.field(P)
    Bm = np.linalg.norm(B, axis=-1)
    fig, ax = plt.subplots(figsize=(5.2, 7.6))
    ax.grid(False)
    ax.streamplot(X * 1e3, Z * 1e3, B[..., 0], B[..., 2], density=1.6, linewidth=0.8,
                  color=np.log10(np.clip(Bm, 1e-4, None)), cmap=matplotlib.colors.ListedColormap(SEQ),
                  arrowsize=0.6)
    L, F = lev.lifter, lev.floater

    def box(x0, z0, w, h, color, **kw):
        ax.add_patch(Rectangle((x0 * 1e3, z0 * 1e3), w * 1e3, h * 1e3, color=color, zorder=3, **kw))

    box(-L.radius, lev.lifter_height - L.length / 2, 2 * L.radius, L.length, MAGNET)
    box(-F.radius, -F.length / 2, 2 * F.radius, F.length, S1)
    for side in (-1, 1):
        z0 = lev.plate_face if side > 0 else -(lev.plate_face + lev.plate_thickness)
        box(-lev.plate_r, z0, 2 * lev.plate_r, lev.plate_thickness, GRAPHITE)
    ax.text(0, (lev.lifter_height) * 1e3, "lifter\n(fixed)", ha="center", va="center",
            color=INK, fontsize=9, zorder=4)
    label_z = (lev.plate_face + lev.plate_thickness) * 1e3 + 0.08 * (top - bottom) * 1e3
    ax.annotate("floater + payload\n(free, touching nothing)", (F.radius * 1e3 * 0.6, F.length / 2 * 1e3),
                xytext=(F.radius * 1e3 + 4, label_z), textcoords="data",
                color=INK, fontsize=9, zorder=5, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.annotate("pyrolytic graphite plates", (-lev.plate_r * 1e3, -(lev.plate_face + lev.plate_thickness) * 1e3),
                xytext=(-lev.plate_r * 1e3, bottom * 1e3 + 0.02 * (top - bottom) * 1e3),
                textcoords="data", color=INK, fontsize=9, zorder=5)
    ax.set_xlim(-half_w * 1e3, half_w * 1e3)
    ax.set_ylim(bottom * 1e3, top * 1e3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Cross-section and field lines")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.07)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def benchmark_stiffness(make_device, path: str, gaps_mm=None) -> None:
    """Vertical stiffness vs gap for the Simon-Heflinger-Geim device."""
    _style()
    gaps_mm = gaps_mm if gaps_mm is not None else np.linspace(0.3, 1.8, 16)
    k = [make_device(g * 1e-3).hessian(full=False)[2, 2] for g in gaps_mm]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(0, color=INK_2, lw=1)
    ax.plot(gaps_mm, np.array(k) * 1e3, color=S1, marker="o", markersize=4)
    ax.axvline(1.4, color=S2, lw=2, ls=(0, (4, 3)))
    ax.text(1.43, max(k) * 1e3 * 0.8, "experiment: well turns\ndouble-humped\nbeyond ~1.4 mm",
            color=INK_2, fontsize=9.5)
    ax.set_xlabel("gap between floater face and each graphite plate [mm]")
    ax.set_ylabel("net vertical stiffness [mN/m]")
    ax.set_title("Validation against a measured levitator", pad=18)
    ax.text(0.0, 1.015, "model curve has no fitted parameters", transform=ax.transAxes,
            color=INK_2, fontsize=9, va="bottom", ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def disturbance_response(lev: Levitator, path: str, q0, v0=(0, 0, 0, 0, 0), t_end=4.0,
                         damping=0.0) -> dict:
    """Nonlinear release from an offset: sideways motion and plate clearance vs time."""
    _style()
    res = simulate(lev, q0=q0, v0=v0, t_end=t_end, damping=damping)
    t, q = res["t"], res["q"]
    clear = plate_clearance(lev, q) * 1e3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    ax1.plot(t, q[:, 0] * 1e3, color=S1, lw=1.4)
    ax1.set_ylabel("sideways x [mm]")
    ax1.set_title(f"Released {q0[0] * 1e3:.1f} mm off-centre and {q0[2] * 1e3:.2f} mm high, no damping")
    ax2.axhline(0, color=S2, lw=2)
    ax2.text(t[-1], 0, "  contact", color=INK_2, va="center", fontsize=9)
    ax2.plot(t, clear, color=S1, lw=1.0)
    ax2.axhline(lev.gap * 1e3, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax2.text(t[-1], lev.gap * 1e3, "  centred", color=INK_2, va="bottom", fontsize=9)
    ax2.set_ylim(-0.05, lev.gap * 1e3 * 1.15)
    ax2.set_ylabel("rim-to-plate clearance [mm]")
    ax2.set_xlabel("time [s]")
    ax2.set_xlim(0, t[-1] * 1.14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    res["min_clearance"] = float(clear.min()) / 1e3
    return res


def payload_scaling(rows_by_band: dict[str, list[tuple[float, float]]], path: str) -> None:
    """Best payload vs lifter mass, one series per thermal-tolerance requirement."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for color, (label, pts) in zip((S1, S2, S3), rows_by_band.items()):
        pts = sorted(p for p in pts if p[1] > 0)
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, marker="o", markersize=5)
        ax.text(x[-1] * 1.08, y[-1], label, color=INK_2, va="center", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.set_xticks([0.05, 0.1, 0.3, 1, 3, 10, 25])
    ax.set_yticks([3, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.minorticks_off()
    ax.set_xlabel("lifter magnet mass [kg]")
    ax.set_ylabel("max payload [g]")
    ax.set_title("How far passive room-temperature levitation scales", pad=18)
    ax.text(0.0, 1.015, "lines: temperature swing the device must ride out without re-adjustment",
            transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom", ha="left")
    ax.set_xlim(0.035, 60)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
