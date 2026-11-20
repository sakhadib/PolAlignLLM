# gaussian_domes.py
# Generate political-compass "gaussian domes" from model means/stds.

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# ----------------------------
# 1) Load data
# ----------------------------
CSV_PATH = r"D:\Research\Political_Bias_of_LLMs\PoliticalCompass\7_summerize\pc_scores_summary_by_model.csv"
df = pd.read_csv(CSV_PATH)

# Expected columns:
# model_name, econ_score_mean, econ_score_std, soc_score_mean, soc_score_std
required = {"model_name","econ_score_mean","econ_score_std","soc_score_mean","soc_score_std"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing columns: {missing}")

# Small epsilon for zero-variance models (prevents divide-by-zero)
EPS = 1e-3
df["econ_score_std"] = df["econ_score_std"].clip(lower=EPS)
df["soc_score_std"]  = df["soc_score_std"].clip(lower=EPS)

# ----------------------------
# 2) Helper functions
# ----------------------------
def bvn_pdf(x, y, mx, my, sx, sy):
    """
    Axis-aligned (rho=0) bivariate normal PDF.
    """
    # 1/(2πσxσy) * exp(-0.5*(((x-mx)/σx)^2 + ((y-my)/σy)^2))
    norm = 1.0 / (2.0 * math.pi * sx * sy)
    z = ((x - mx) / sx) ** 2 + ((y - my) / sy) ** 2
    return norm * np.exp(-0.5 * z)

def add_confidence_ellipses(ax, mx, my, sx, sy, label=None, lw=1.5, alpha=0.15):
    """
    Draw 68% and 95% confidence ellipses for a 2D Gaussian with diagonal Σ.
    For 2D Gaussian, contour level defined by (x-μ)^T Σ^{-1} (x-μ) = χ²_{2,p}.
    χ²_{2,0.68} ≈ 2.279  -> scale ≈ sqrt(2.279)
    χ²_{2,0.95} ≈ 5.991  -> scale ≈ sqrt(5.991)
    """
    chi2_68 = 2.279
    chi2_95 = 5.991
    scales = [
        (math.sqrt(chi2_95), 0.10, True),  # outer (95%)
        (math.sqrt(chi2_68), 0.25, False), # inner (68%)
    ]

    # draw mean point
    ax.plot(mx, my, marker="o", markersize=4, linestyle="None")

    for s, a, dashed in scales:
        w = 2 * s * sx  # width  = 2 * scale * σx
        h = 2 * s * sy  # height = 2 * scale * σy
        e = Ellipse((mx, my), width=w, height=h, angle=0.0,
                    fill=True, alpha=alpha if not dashed else alpha*0.8,
                    edgecolor="black", linewidth=lw, linestyle="--" if dashed else "-")
        ax.add_patch(e)

    if label:
        ax.text(mx, my, f" {label}", fontsize=8, ha="left", va="bottom")

# ----------------------------
# 3) Set plot ranges from data
# ----------------------------
# Use ±3σ envelope across all models; pad a little for margins.
xmin = (df["econ_score_mean"] - 3*df["econ_score_std"]).min()
xmax = (df["econ_score_mean"] + 3*df["econ_score_std"]).max()
ymin = (df["soc_score_mean"]  - 3*df["soc_score_std"]).min()
ymax = (df["soc_score_mean"]  + 3*df["soc_score_std"]).max()

pad_x = 0.5
pad_y = 0.5
xmin -= pad_x; xmax += pad_x
ymin -= pad_y; ymax += pad_y

# Grid for domes
nx = 300
ny = 300
X, Y = np.meshgrid(np.linspace(xmin, xmax, nx),
                   np.linspace(ymin, ymax, ny))

# ----------------------------
# 4) Mixture-of-Gaussians "dome"
# ----------------------------
Z = np.zeros_like(X, dtype=float)
for _, r in df.iterrows():
    Z += bvn_pdf(
        X, Y,
        r["econ_score_mean"], r["soc_score_mean"],
        r["econ_score_std"],  r["soc_score_std"]
    )

# Normalize Z to [0,1] just for nicer visualization
Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)

# ----------------------------
# 5) Figure A: Ellipses per model (68% & 95%)
# ----------------------------
figA, axA = plt.subplots(figsize=(9, 7))
for _, r in df.iterrows():
    add_confidence_ellipses(
        axA,
        r["econ_score_mean"], r["soc_score_mean"],
        r["econ_score_std"],  r["soc_score_std"],
        label=r["model_name"]
    )

axA.axhline(0, color="gray", linewidth=0.8, linestyle="--")
axA.axvline(0, color="gray", linewidth=0.8, linestyle="--")
axA.set_xlim(xmin, xmax)
axA.set_ylim(ymin, ymax)
axA.set_xlabel("Economic (Left  ⟵  →  Right)")
axA.set_ylabel("Social (Libertarian  ⟵  →  Authoritarian)")
axA.set_title("Per-Model 2D Gaussian Confidence Ellipses (68% & 95%)")
axA.set_aspect("equal", adjustable="box")
figA.tight_layout()

# ----------------------------
# 6) Figure B: 2D “dome” (filled contour of the mixture)
# ----------------------------
figB, axB = plt.subplots(figsize=(9, 7))
contours = axB.contourf(X, Y, Z, levels=15)
figB.colorbar(contours, ax=axB, shrink=0.85, label="Mixture density (normalized)")
# overlay means so readers see the centers
axB.scatter(df["econ_score_mean"], df["soc_score_mean"], s=20)
for _, r in df.iterrows():
    axB.text(r["econ_score_mean"], r["soc_score_mean"], f" {r['model_name']}", fontsize=7, va="bottom")

axB.axhline(0, color="gray", linewidth=0.8, linestyle="--")
axB.axvline(0, color="gray", linewidth=0.8, linestyle="--")
axB.set_xlim(xmin, xmax)
axB.set_ylim(ymin, ymax)
axB.set_xlabel("Economic (Left  ⟵  →  Right)")
axB.set_ylabel("Social (Libertarian  ⟵  →  Authoritarian)")
axB.set_title("Mixture-of-Gaussians Dome (All Models)")
axB.set_aspect("equal", adjustable="box")
figB.tight_layout()

# ----------------------------
# 7) Figure C: 3D surface of the mixture dome
# ----------------------------
figC = plt.figure(figsize=(10, 8))
axC = figC.add_subplot(111, projection="3d")
axC.plot_surface(X, Y, Z, linewidth=0, antialiased=True, rstride=3, cstride=3, alpha=0.9)
axC.set_xlabel("Economic")
axC.set_ylabel("Social")
axC.set_zlabel("Mixture density (normalized)")
axC.set_title("Mixture-of-Gaussians 3D Dome (All Models)")
figC.tight_layout()

# ----------------------------
# 8) Save outputs to current_dir/plots
# ----------------------------
out_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(out_dir, exist_ok=True)
fA = os.path.join(out_dir, "gaussian_ellipses.png")
fB = os.path.join(out_dir, "gaussian_mixture_contour.png")
fC = os.path.join(out_dir, "gaussian_mixture_surface.png")
figA.savefig(fA, dpi=200)
figB.savefig(fB, dpi=200)
figC.savefig(fC, dpi=200)

print("Saved:")
print(" -", fA)
print(" -", fB)
print(" -", fC)

plt.show()
