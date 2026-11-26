"""Build the six KTN paper figures in dark/gold style (near-black background,
gold lettering, like the P=NP joke banner). All numbers are taken from the
KTN bench results files. No number is invented.
Run with system python3 (matplotlib 3.6.3)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("/home/hatch/workspace/your_files/ktn-figures")
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0a0a0a"
GOLD = "#d4af37"
PALE = "#f0d878"
OFFWHITE = "#f2ead8"
DIM = "#8a7326"
GRID = "#3a3325"
GREY = "#8a8a7a"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": GOLD,
    "axes.edgecolor": GOLD,
    "axes.labelcolor": GOLD,
    "xtick.color": GOLD,
    "ytick.color": GOLD,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
})


def style_ax(ax, title):
    ax.set_title(title, color=GOLD, fontsize=16, fontweight="bold", pad=14)
    for spine in ax.spines.values():
        spine.set_color(GOLD)
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor=BG)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- fig 1
# Two traces, R3-E1, results-r3-e1.md
b = np.array([1, 2, 3, 4])
fast = np.array([0.398654, 0.952743, 0.983077, 0.989586])
slow = np.array([0.318923, 0.825979, 0.951657, 0.982000])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(b, fast, color=GOLD, marker="o", markersize=9, linewidth=2.5,
        label="fast trace (live, unrecorded)")
ax.plot(b, slow, color=OFFWHITE, marker="s", markersize=9, linewidth=2.5,
        label="slow trace (the record)")
ax.axvline(3, color=PALE, linestyle="--", linewidth=1.5)
ax.annotate("crossing at boundary 3\nslow takes over",
            xy=(3, 0.951657), xytext=(3.15, 0.55),
            color=PALE, fontsize=12,
            arrowprops=dict(arrowstyle="->", color=PALE, lw=1.5))
ax.annotate("signal = slow from here\nchange 0.0 through 40 poison probes",
            xy=(4, 0.982000), xytext=(1.0, 0.14),
            color=OFFWHITE, fontsize=11)
ax.text(0.02, 0.97, "fast carries impact, slow carries influence\none switch, not two",
        transform=ax.transAxes, color=GOLD, fontsize=11, va="top")
ax.set_xlim(0.8, 4.2)
ax.set_ylim(0.1, 1.05)
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("consolidation boundary")
ax.set_ylabel("probe value")
style_ax(ax, "Figure 1. The two points of state")
leg = ax.legend(loc="lower right", fontsize=11, frameon=True)
leg.get_frame().set_facecolor(BG)
leg.get_frame().set_edgecolor(GOLD)
for text in leg.get_texts():
    text.set_color(GOLD)
fig.tight_layout()
save(fig, "fig1-two-traces.png")

# ---------------------------------------------------------------- fig 2
# Anatomy schematic, no measured numbers. Formula from draft section 3.1.
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("Figure 2. Anatomy of a KT neuron", color=GOLD,
             fontsize=16, fontweight="bold", pad=12)
ax.text(5.5, 6.45, r"$y = \sigma(a \cdot (w \cdot x) + (b - \mathrm{scar}))$",
        color=PALE, fontsize=19, ha="center", va="center")
ax.text(5.5, 6.0, "the signal speaks at its weakest authority",
        color=GOLD, fontsize=11, ha="center", va="center")


def box(x, y, w, h, title, lines, title_color=PALE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                facecolor="#141210", edgecolor=GOLD, linewidth=1.4))
    ax.text(x + w / 2, y + h - 0.32, title, color=title_color, fontsize=12,
            fontweight="bold", ha="center", va="top")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - i * 0.34, ln, color=GOLD,
                fontsize=10.5, ha="center", va="top")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                 mutation_scale=16, color=GOLD, linewidth=1.6))


box(0.2, 4.3, 2.2, 1.5, "INPUT x", ["what the neuron", "is asked"])
box(3.0, 4.3, 2.4, 1.5, "WEIGHTS w", ["assertions,", "not variables"])
box(6.0, 4.3, 2.2, 1.5, "AUTHORITY GATE", ["a = weakest", "authority"])
box(8.8, 4.3, 2.0, 1.5, "OUTPUT y", ["sigmoid of", "gated sum"])
arrow(2.4, 5.05, 3.0, 5.05)
arrow(5.4, 5.05, 6.0, 5.05)
arrow(8.2, 5.05, 8.8, 5.05)

box(0.2, 2.1, 3.2, 1.6, "KID AT BIRTH", ["192-bit id", "no KID, no read,", "no write"])
box(3.9, 2.1, 3.4, 1.6, "RECORD", ["intervals [k_start, k_end)", "SHA-256 chain, close", "is the only mutation"])
box(7.8, 2.1, 3.0, 1.6, "SCAR + GAIN", ["scar: fast trace,", "sits in the bias,", "gain: one scalar per neuron"])
arrow(1.8, 3.7, 4.2, 4.3)
arrow(5.6, 3.7, 5.6, 4.3)
arrow(9.3, 3.7, 9.3, 4.3)

box(2.6, 0.2, 5.8, 1.3, "SLOW TRACE = THE RECORD",
    ["k advances at consolidation, never at the step",
     "fast trace forgets, slow trace is the record"], title_color=GOLD)
arrow(5.5, 2.1, 5.5, 1.5)
fig.tight_layout()
save(fig, "fig2-anatomy.png")

# ---------------------------------------------------------------- fig 3
# Toy run, draft section 4, verified live Nov 26.
stages = ["After teacher", "After liar injury", "After restore"]
out = np.array([0.886, 0.101, 0.995])
auth = np.array([0.9, 0.9, 0.6])

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
ax.plot(x, out, color=GOLD, marker="o", markersize=10, linewidth=2.5,
        label="output on input (1, 1)")
ax.plot(x, auth, color=OFFWHITE, marker="s", markersize=10, linewidth=2.5,
        label="signal authority")
for xi, yi in zip(x, out):
    ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                xytext=(0, 12), color=GOLD, fontsize=12, ha="center")
for xi, yi in zip(x, auth):
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=(0, -20), color=OFFWHITE, fontsize=12, ha="center")
ax.annotate("behavior returns", xy=(2, 0.995), xytext=(1.35, 0.88),
            color=PALE, fontsize=12,
            arrowprops=dict(arrowstyle="->", color=PALE, lw=1.5))
ax.annotate("authority lags at 0.6", xy=(2, 0.6), xytext=(1.35, 0.38),
            color=OFFWHITE, fontsize=12,
            arrowprops=dict(arrowstyle="->", color=OFFWHITE, lw=1.5))
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylim(0.0, 1.1)
ax.set_ylabel("value")
style_ax(ax, "Figure 3. The toy: behavior returns, authority lags")
leg = ax.legend(loc="lower left", fontsize=11, frameon=True)
leg.get_frame().set_facecolor(BG)
leg.get_frame().set_edgecolor(GOLD)
for text in leg.get_texts():
    text.set_color(GOLD)
fig.tight_layout()
save(fig, "fig3-toy.png")

# ---------------------------------------------------------------- fig 4
# E3, results.md table: post-repair, +1, +2, +6
e3_x = ["Post-repair", "+1", "+2", "+6"]
e3_auth = np.array([0.6, 0.6, 0.6, 0.7])
e3_ptrue = np.array([0.820, 0.820, 0.820, 0.881])
e3_def = np.array([0.30, 0.29, 0.28, 0.24])

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(4)
ax.plot(x, e3_auth, color=GOLD, marker="o", markersize=9, linewidth=2.5,
        label="seated authority")
ax.plot(x, e3_ptrue, color=OFFWHITE, marker="s", markersize=9, linewidth=2.5,
        label="mean P(true)")
ax.axhline(0.8, color=PALE, linestyle="--", linewidth=1.2)
ax.axhline(0.878, color=GREY, linestyle=":", linewidth=1.5)
ax.text(0.05, 0.885, "threshold 0.878, missed", color=GREY, fontsize=11)
ax2 = ax.twinx()
ax2.plot(x, e3_def, color=DIM, marker="^", markersize=9, linewidth=2,
         label="raw deficit")
ax2.set_ylabel("raw deficit", color=DIM, fontsize=13)
ax2.tick_params(colors=DIM, labelsize=12)
ax2.set_ylim(0.0, 0.4)
for spine in ax2.spines.values():
    spine.set_color(GOLD)
ax2.spines["top"].set_visible(False)
ax.set_xticks(x)
ax.set_xticklabels(e3_x)
ax.set_ylim(0.5, 0.95)
ax.set_ylabel("authority / mean P(true)")
style_ax(ax, "Figure 4. E3: the law at scale")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
from matplotlib.lines import Line2D
pre_line = Line2D([0], [0], color=PALE, linestyle="--", linewidth=1.2)
lines1 = lines1 + [pre_line]
labels1 = labels1 + ["pre-injury authority 0.8 (dashed)"]
leg = ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
                fontsize=11, frameon=True)
leg.get_frame().set_facecolor(BG)
leg.get_frame().set_edgecolor(GOLD)
for text in leg.get_texts():
    text.set_color(GOLD)
fig.tight_layout()
save(fig, "fig4-authority-lag.png")

# ---------------------------------------------------------------- fig 5
# R3-E2, results-r3-e2.md: deficit 0.30, decay per consolidation.
n = np.arange(0, 31)
fast_d = np.maximum(0.0, 0.30 - 0.05 * n)
slow_d = np.maximum(0.0, 0.30 - 0.01 * n)
perm_d = np.full_like(n, 0.30, dtype=float)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n, fast_d, color=GOLD, linewidth=2.5, label="FAST, decay 0.05")
ax.plot(n, slow_d, color=OFFWHITE, linewidth=2.5, label="SLOW, decay 0.01")
ax.plot(n, perm_d, color=GREY, linewidth=2.5, linestyle="--",
        label="PERMANENT, decay 0.00")
ax.plot([5], [0.05], color=GOLD, marker="o", markersize=10)
ax.annotate("healed +5", xy=(5, 0.05), xytext=(8, 0.14), color=GOLD,
            fontsize=12, arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))
ax.plot([25], [0.05], color=OFFWHITE, marker="s", markersize=9)
ax.annotate("healed +25", xy=(25, 0.05), xytext=(19, 0.14), color=OFFWHITE,
            fontsize=12, arrowprops=dict(arrowstyle="->", color=OFFWHITE, lw=1.5))
ax.annotate("never heals", xy=(30, 0.30), xytext=(22, 0.26), color=GREY,
            fontsize=12, arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))
ax.set_xlim(0, 30)
ax.set_ylim(0.0, 0.38)
ax.set_xlabel("clean consolidations after repair (+n)")
ax.set_ylabel("raw deficit")
style_ax(ax, "Figure 5. R3-E2: the tail sets the healing")
leg = ax.legend(loc="upper right", fontsize=11, frameon=True)
leg.get_frame().set_facecolor(BG)
leg.get_frame().set_edgecolor(GOLD)
for text in leg.get_texts():
    text.set_color(GOLD)
fig.tight_layout()
save(fig, "fig5-tails.png")

# ---------------------------------------------------------------- fig 6
# C2C, results.md: grouped bars, P(true) and P(liar).
cats = ["Post-transfer", "Post-null", "Post-repair"]
# vanilla has no post-null / post-repair: no record to null or repair.
v_true = np.array([0.425, np.nan, np.nan])
v_liar = np.array([0.543, np.nan, np.nan])
k_true = np.array([0.325, 0.943, 0.836])
k_liar = np.array([0.654, 0.002, 0.115])

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(3)
w = 0.19
ax.bar(x - 1.5 * w, v_true, w, color="#b8912f", edgecolor=GOLD, linewidth=1,
       label="vanilla P(true)")
ax.bar(x - 0.5 * w, v_liar, w, color="#6e6e60", edgecolor=GOLD, linewidth=1,
       label="vanilla P(liar)")
ax.bar(x + 0.5 * w, k_true, w, color=GOLD, edgecolor=PALE, linewidth=1,
       label="KT P(true)")
ax.bar(x + 1.5 * w, k_liar, w, color=OFFWHITE, edgecolor=PALE, linewidth=1,
       label="KT P(liar)")
for xi, v in zip(x - 1.5 * w, v_true):
    if not np.isnan(v):
        ax.text(xi, v + 0.02, f"{v:.3f}", color=GOLD, fontsize=10, ha="center")
for xi, v in zip(x - 0.5 * w, v_liar):
    if not np.isnan(v):
        ax.text(xi, v + 0.02, f"{v:.3f}", color=OFFWHITE, fontsize=10, ha="center")
for xi, v in zip(x + 0.5 * w, k_true):
    ax.text(xi, v + 0.02, f"{v:.3f}", color=GOLD, fontsize=10, ha="center")
for xi, v in zip(x + 1.5 * w, k_liar):
    ax.text(xi, v + 0.02, f"{v:.3f}", color=OFFWHITE, fontsize=10, ha="center")
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylim(0.0, 1.1)
ax.set_ylabel("mean probability")
style_ax(ax, "Figure 6. C2C: same poison, two receivers")
leg = ax.legend(loc="upper right", fontsize=10.5, frameon=True, ncol=2)
leg.get_frame().set_facecolor(BG)
leg.get_frame().set_edgecolor(GOLD)
for text in leg.get_texts():
    text.set_color(GOLD)
fig.tight_layout()
fig.subplots_adjust(bottom=0.20)
fig.text(0.5, 0.03,
         "vanilla has no record: nothing to null, nothing to repair. "
         "Post-repair vanilla: full retrain only, 10 steps, no lineage.",
         color=GREY, fontsize=10.5, ha="center")
save(fig, "fig6-c2c.png")

print("done")
