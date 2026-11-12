#!/usr/bin/env python3
"""Clean vector replacement for the garbled Figure 1 (technical roadmap)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

BLUE = "#DCE9F5"
GREEN = "#DFF0E3"
ORANGE = "#FBE7D2"
PURPLE = "#EAE2F3"
GREY = "#F2F2F2"
EDGE = "#3A3A3A"


def box(x, y, w, h, title, lines, fc, title_size=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                       linewidth=0.9, edgecolor=EDGE, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold")
    if lines:
        ax.text(x + 1.0, y + h - 3.6, "\n".join(lines), ha="left", va="top",
                fontsize=7.4, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=1.0, color=EDGE,
                                 shrinkA=2, shrinkB=2))


# I. Problem
box(2, 37, 21, 20, "I. Problem Formalization", [
    "Heterogeneous swarm",
    r"$N_{\mathrm{AOA}}=1$, $N_{\mathrm{OMNI}}=2$",
    r"Unknown number $N_s$",
    r"Dual-modal observations",
    r"$z=[I,\ \alpha]^{\top}$",
], BLUE)

# II. Spatial-probabilistic decoupling
box(27, 37, 46, 20, "II. Spatial-Probabilistic Decoupling", [], GREEN, title_size=9)
box(28.2, 38.2, 21.0, 14.6, "Spatial constraint", [
    r"Dynamic Voronoi cells $V_{k,j}$",
    "Cell-local particle quota",
    "Prevents global impoverishment",
], "#EAF5EC", title_size=8)
box(50.8, 38.2, 21.0, 14.6, "Probabilistic constraint", [
    r"Joint likelihood $p(I,\alpha\mid x)$",
    "Cell-local normalization",
    "Stratified resampling",
], "#EAF5EC", title_size=8)

# III. Active sensing
box(27, 20, 46, 13.5, "III. Active Sensing (Theorem 1)", [
    r"Gaussian max-entropy upper bound:",
    r"$I(x_s;\alpha\mid p)\ \leq\ \frac{1}{2}\log\!\left(1+\sigma_\alpha^2(p)/\sigma_n^2\right)$",
    r"Maximize predicted AOA variance $\Rightarrow$ $O(N_p)$ control",
    "Information gradient + artificial potential field",
], ORANGE)

# IV. Confirmation and subtraction
box(2, 2, 46, 15, "IV. Confirmation & Sequential Subtraction", [
    r"Entropy / proximity confirmation",
    r"Subtract confirmed source: $I_{\mathrm{res}}$",
    r"Neyman--Pearson residual test ($3\sigma$)",
    r"Reset belief $\rightarrow$ next source",
], PURPLE)

# V. Output
box(54, 2, 44, 15, "V. Output", [
    r"Source map: $\{\hat{x}_{s,m},\hat{A}_{s,m}\}$",
    "Multi-robot trajectories",
    "Residual field verification",
    r"Terminate when $\mathcal{H}_0$ is accepted",
], GREY)

# Arrows
arrow(23.2, 47, 26.8, 47)          # I -> II
arrow(50, 36.8, 50, 33.8)          # II -> III
arrow(50, 19.8, 50, 17.2)          # III -> IV/V junction
arrow(48.2, 9.5, 53.8, 9.5)        # IV -> V
# Feedback loop from IV back to II (sequential reset)
ax.add_patch(FancyArrowPatch((1.2, 9.5), (1.2, 58.2), arrowstyle="-",
                             linewidth=0.8, color="#777777", linestyle="--"))
ax.add_patch(FancyArrowPatch((1.2, 58.2), (27.0, 58.2), arrowstyle="-|>",
                             mutation_scale=9, linewidth=0.8, color="#777777",
                             linestyle="--", shrinkA=0, shrinkB=2))
ax.text(14.0, 58.8, "sequential reset / next source", ha="center", va="bottom",
        fontsize=7, color="#555555", style="italic")

# Small note
ax.text(99, 0.3, "Theorem 1 is an upper-bound surrogate, not an exact equivalence.",
        ha="right", va="bottom", fontsize=6.5, color="#555555", style="italic")

fig.tight_layout(pad=0.2)
fig.savefig("Figure_1_new.pdf", bbox_inches="tight")
fig.savefig("Figure_1_new.png", dpi=300, bbox_inches="tight")
print("Wrote Figure_1_new.pdf and Figure_1_new.png")
