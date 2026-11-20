"""All figures in the paper. Run after analyze.py:  python analysis/figures.py"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text

from common import (ROOT, FIG, LOGO, PAL, MODELS, KEYS, AXES, INST_NAME, load_news)

RES = ROOT / "results24"   # questionnaire + link figures: 24 same-protocol models
RESN = ROOT / "results"    # news figure: all 26 models
R = json.load(open(RES / "results.json"))
RN = json.load(open(RESN / "results.json"))
KEYS_ALL = list(KEYS)
KEYS = [k for k in KEYS_ALL if k not in ("x_ai_grok_4", "google_gemini_2_5_pro")]
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": PAL["ink"], "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
DIV = LinearSegmentedColormap.from_list("div", [PAL["left"], "#F4F4F4", PAL["right"]])
CORR = LinearSegmentedColormap.from_list("corr", ["#A7A7A7", "#FFFFFF", "#2F3E57"])
_logo_cache = {}


def logo(key):
    f = MODELS[key][2]
    if f not in _logo_cache:
        _logo_cache[f] = plt.imread(LOGO / f"{f}.png")
    return _logo_cache[f]


def put_logo(ax, x, y, key, zoom=0.045, z=5):
    ab = AnnotationBbox(OffsetImage(logo(key), zoom=zoom), (x, y), frameon=False, zorder=z)
    ax.add_artist(ab)


def short(key):
    return MODELS[key][0]


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


pcs = pd.read_csv(RES / "model_pcs.csv", index_col=0).loc[KEYS]
PN = pd.read_csv(RESN / "news_per_model.csv", index_col=0).loc[KEYS_ALL]
load = pd.read_csv(RES / "axis_loadings.csv", index_col=0)
P = pd.read_csv(RES / "news_per_model.csv", index_col=0).loc[KEYS]

# ===================================================================== Dimensionality: three separate figures
sh = np.array(R["axis_pca"]["share"]) * 100
pa = np.array(R["axis_pca"]["parallel95"]) / len(AXES) * 100
k = np.arange(1, len(sh) + 1)
# (1) scree
fig, a1 = plt.subplots(figsize=(3.05, 1.55))
a1.bar(k, sh, color=[PAL["ink"]] + [PAL["muted"]] * (len(sh) - 1), width=0.7, zorder=2)
a1.plot(k, pa, "--", color=PAL["ink"], lw=0.8, marker="o", ms=2, mfc="white", label="random-data threshold (95th pct.)", zorder=3)
a1.text(1.5, sh[0] - 6, f"{sh[0]:.0f}%", ha="left", fontsize=7.5, fontweight="bold")
a1.text(2, sh[1] + 1.5, f"{sh[1]:.0f}%", ha="center", fontsize=6.5)
a1.set_xticks(k); a1.set_xlabel("Principal component", labelpad=1)
a1.set_ylabel("Variance (%)"); a1.set_ylim(0, 66)
a1.legend(frameon=False, loc="upper right", fontsize=6.4, handlelength=1.8)
fig.savefig(FIG / "fig_scree.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# (2) loadings
short_ax = {"pct_econ": "PCT economic (−)", "pct_soc": "PCT social (−)", "sap_right": "Sapply right (−)",
            "sap_auth": "Sapply authority", "sap_prog": "Sapply progressive", "8val_econ": "8V equality",
            "8val_dipl": "8V globalism", "8val_govt": "8V liberty", "8val_scty": "8V progress"}
ld = load.PC1.copy()
order_ax = ["pct_econ", "pct_soc", "sap_right", "sap_auth", "sap_prog", "8val_econ", "8val_dipl", "8val_govt", "8val_scty"]
vals = [(-ld[a] if a in ("pct_econ", "pct_soc", "sap_right") else ld[a]) for a in order_ax]
fig, a3 = plt.subplots(figsize=(3.05, 1.75))
yy = np.arange(len(order_ax))[::-1]
a3.barh(yy, vals, color=[PAL[a.split("_")[0]] for a in order_ax], height=0.72)
for y_, v in zip(yy, vals):
    a3.text(v + 0.006, y_, f"{v:.2f}", va="center", fontsize=6, color="#555555")
a3.set_yticks(yy); a3.set_yticklabels([short_ax[a] for a in order_ax], fontsize=6.6)
for t, a in zip(a3.get_yticklabels(), order_ax):
    t.set_color(PAL[a.split("_")[0]])
a3.axvline(0, color=PAL["ink"], lw=0.5)
a3.set_xlim(0, 0.48); a3.set_xlabel("Loading on PC1", labelpad=1)
a3.tick_params(axis="y", length=0)
fig.savefig(FIG / "fig_loadings.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# (3) model ranking
order_m = pcs.PC1.sort_values().index
fig, a2 = plt.subplots(figsize=(3.05, 3.2))
a2.axvline(0, color=PAL["muted"], lw=0.5, zorder=0)
for i, kk in enumerate(order_m):
    v = pcs.loc[kk, "PC1"]
    c = PAL["left"] if v > 0 else PAL["right"]
    a2.plot([0, v], [i, i], color=c, lw=1.4, alpha=0.55, solid_capstyle="round", zorder=1)
    put_logo(a2, v, i, kk, zoom=0.03)
    a2.text(v + (0.6 if v > 0 else -0.6), i, short(kk), fontsize=6.0, va="center",
            ha="left" if v > 0 else "right", color=PAL["ink"])
a2.set_yticks([]); a2.spines["left"].set_visible(False)
a2.set_ylim(-0.8, len(order_m) - 0.2); a2.set_xlim(-8.6, 8.4)
a2.set_xlabel("PC1 score  (← conservative / market · progressive / egalitarian →)", labelpad=1, fontsize=7)
fig.savefig(FIG / "fig_ranking.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ===================================================================== Figure 2: MTMM
T = pd.read_csv(RES / "mtmm_aligned_inputs.csv", index_col=0)
labels = {"Econ|pct": "Econ-left · PCT", "Econ|sap": "Econ-left · Sapply", "Econ|8val": "Econ-left · 8V",
          "Lib|pct": "Libertarian · PCT", "Lib|sap": "Libertarian · Sapply", "Lib|8val": "Libertarian · 8V",
          "Prog|sap": "Progressive · Sapply", "Prog|8val": "Progressive · 8V"}
rho = T.corr(method="spearman")
n = len(rho)
fig, ax = plt.subplots(figsize=(3.1, 2.9))
mask = np.triu(np.ones_like(rho, dtype=bool), 0)
for i in range(n):
    for j in range(n):
        if j >= i:
            continue
        v = rho.iloc[i, j]
        ax.add_patch(Rectangle((j, i), 1, 1, color=CORR((v + 1) / 2), lw=0))
        same_trait = rho.index[i].split("|")[0] == rho.columns[j].split("|")[0]
        ax.text(j + .5, i + .5, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                fontsize=6.3, color="white" if abs(v) > .62 else PAL["ink"],
                fontweight="bold" if same_trait else "normal")
        if same_trait:
            ax.add_patch(Rectangle((j + .04, i + .04), .92, .92, fill=False, ec=PAL["ink"], lw=1.1))
# instrument colour strips
for i, c in enumerate(rho.index):
    ax.add_patch(Rectangle((-0.32, i + .08), .22, .84, color=PAL[c.split("|")[1]], clip_on=False))
    ax.add_patch(Rectangle((i + .08, n + .1), .84, .22, color=PAL[c.split("|")[1]], clip_on=False))
ax.set_xlim(-0.35, n - 1)
ax.set_ylim(n + .35, 1)
ax.set_yticks(np.arange(1, n) + .5)
ax.set_yticklabels([labels[c] for c in rho.index[1:]], fontsize=6.4)
ax.set_xticks(np.arange(0, n - 1) + .5)
ax.set_xticklabels([labels[c] for c in rho.columns[:-1]], rotation=40, ha="right", fontsize=6.4)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
# highlight the liberty-progressive conflation block
ax.add_patch(Rectangle((5, 6), 1, 2, fill=False, ec=PAL["right"], lw=1.3, ls="--"))
ax.add_patch(Rectangle((2.6, 1.25), .45, .45, fill=False, ec=PAL["ink"], lw=1.1))
ax.text(3.2, 1.48, "same trait, other instrument\n(convergent: should be high)", fontsize=5.6, va="center")
ax.add_patch(Rectangle((2.6, 2.25), .45, .45, fill=False, ec=PAL["right"], lw=1.2, ls="--"))
ax.text(3.2, 2.48, "different traits (discriminant:\nshould be lower) -- violated", fontsize=5.6, va="center", color=PAL["right"])
sm = plt.cm.ScalarMappable(cmap=CORR, norm=plt.Normalize(-1, 1))
cb = fig.colorbar(sm, ax=ax, location="left", fraction=0.035, pad=0.06, shrink=0.6)
cb.set_label(f"Spearman ρ ({len(KEYS)} models)", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
fig.savefig(FIG / "fig_mtmm.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================================== Figure 3: news behaviour
news = load_news().dropna(subset=["pred"])
grp = {"Far Left": "Left†", "Left": "Left†", "Lean Left": "Lean Left", "Center": "Center",
       "Lean Right": "Lean Right", "Right": "Right‡", "Far Right": "Right‡"}
gval = {"Left†": -2.05, "Lean Left": -1, "Center": 0, "Lean Right": 1, "Right‡": 2.15}
news["g"] = news.source_bias.map(grp)
order = ["Left†", "Lean Left", "Center", "Lean Right", "Right‡"]
gcol = {"Left†": PAL["left"], "Lean Left": "#8FB0DA", "Center": PAL["center"], "Lean Right": "#E2A0A2",
        "Right‡": PAL["right"]}
fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.3, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1]))
for i, g in enumerate(order):
    v = news[news.g == g].pred.values
    parts = b1.violinplot([v], positions=[i], widths=0.8, showextrema=False, points=200, bw_method=0.25)
    for pc in parts["bodies"]:
        pc.set_facecolor(gcol[g]); pc.set_alpha(0.55); pc.set_edgecolor("none")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    b1.plot([i, i], [q1, q3], color=PAL["ink"], lw=2.2, solid_capstyle="butt", zorder=3)
    b1.scatter([i], [v.mean()], s=14, color="white", edgecolor=PAL["ink"], lw=0.8, zorder=4)
    b1.scatter([i], [gval[g]], marker="_", s=160, color=gcol[g], lw=2.2, zorder=4)

xs = np.arange(5)
b1.plot(xs, [gval[g] for g in order], color=PAL["muted"], lw=0.8, ls=":", zorder=1)
mix = RN["news_mixed"]
b1.set_xticks(xs)
b1.set_xticklabels([f"{l}\nn={news[news.g == g].article.nunique()}" for l, g in zip(["Left+Far L.", "Lean L.", "Center", "Lean R.", "Right+Far R."], order)], fontsize=6.4)
b1.set_ylim(-3.3, 3.1)
b1.set_ylabel("Predicted slant score (−3 … +3)")
b1.axhline(0, color=PAL["muted"], lw=0.5, zorder=0)
b1.set_title("Predictions are compressed toward the centre", loc="left")
b1.text(0.02, 0.97, f"pooled slope = {mix['slope']:.2f}  (ideal 1)\noffset at Center = {mix['intercept']:+.2f}",
        transform=b1.transAxes, fontsize=6.5, va="top")
b1.scatter([], [], marker="_", s=80, color=PAL["ink"], label="outlet label")
b1.scatter([], [], s=12, color="white", edgecolor=PAL["ink"], label="mean prediction")
b1.legend(frameon=False, loc="lower right", fontsize=6.2, handletextpad=0.3)
panel(b1, "a")

b2.axvline(0, color=PAL["muted"], lw=0.5, zorder=0)
b2.axhline(0.5, color=PAL["muted"], lw=0.5, ls=":", zorder=0)
for kk in KEYS_ALL:
    put_logo(b2, PN.loc[kk, "offset"], PN.loc[kk, "conc_ws"], kk, zoom=0.034)
for kk, dx, dy, ha in [("microsoft_phi_3_5_mini_128k_instruct", 0, -0.016, "center"), ("openai_gpt_5_nano", 0.045, 0, "left"),
                       ("openai_gpt_4_1_nano", -0.045, 0, "right"), ("openai_gpt_5", 0.04, 0.004, "left"),
                       ("anthropic_claude_4_sonnet", -0.05, 0.0, "right"), ("meta_meta_llama_3_8b_instruct", -0.045, 0, "right"),
                       ("x_ai_grok_4", -0.04, 0.008, "right")]:
    b2.text(PN.loc[kk, "offset"] + dx, PN.loc[kk, "conc_ws"] + dy, short(kk), fontsize=5.6, color="#555555", ha=ha,
            va="center" if dy == 0 else "top")
b2.set_xlabel("Directional offset at Center  (← reads content as more left)")
b2.set_ylabel("Within-story concordance")
b2.set_xlim(-1.08, 0.3)
b2.set_ylim(0.49, 0.74)
b2.text(-1.05, 0.503, "chance", fontsize=6, color="#888888")
b2.set_title("Per-model offset vs. ability to rank outlets", loc="left")
panel(b2, "b")
fig.tight_layout(w_pad=1.4)
fig.savefig(FIG / "fig_news.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================================== Figure 4: link
L = pd.read_csv(RES / "link.csv")
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.2))
for ax, yv, ttl, ylab in [(axs[0], "offset", "Questionnaire position does not predict lean", "Directional offset at Center"),
                          (axs[1], "conc_ws", "…but co-varies with discrimination", "Within-story concordance")]:
    xv = pcs.PC1.values
    yy = P[yv].values
    b = np.polyfit(xv, yy, 1)
    xx = np.linspace(xv.min() - .3, xv.max() + .3, 50)
    boots = []
    rng = np.random.default_rng(7)
    for _ in range(1000):
        idx = rng.integers(0, len(xv), len(xv))
        boots.append(np.polyval(np.polyfit(xv[idx], yy[idx], 1), xx))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    c = PAL["muted"] if yv == "offset" else PAL["8val"]
    ax.fill_between(xx, lo, hi, color=c, alpha=0.22, lw=0)
    ax.plot(xx, np.polyval(b, xx), color=c if yv != "offset" else "#9E9E9E", lw=1.1)
    for kk, xi, yi in zip(KEYS, xv, yy):
        put_logo(ax, xi, yi, kk, zoom=0.03)
    for kk in ["microsoft_phi_3_5_mini_128k_instruct", "openai_gpt_5_nano", "openai_gpt_4_1_nano"]:
        i = KEYS.index(kk)
        d = 0.075 if yv == "offset" else 0.0125
        ax.text(xv[i], yy[i] - d, short(kk), fontsize=5.6, color="#555555", va="top", ha="center")
    row = L[(L.y == yv) & (L.x == "PC1")].iloc[0]
    ax.text(0.03, 0.95 if yv == "conc_ws" else 0.08,
            f"Spearman ρ = {row.rho:+.2f}  [{row.lo:+.2f}, {row.hi:+.2f}]\n$p_{{FDR}}$ = {row.p_fdr:.3f}",
            transform=ax.transAxes, fontsize=6.6, va="top" if yv == "conc_ws" else "bottom")
    ax.set_xlabel("Ideology PC1 (→ progressive / egalitarian)")
    ax.set_ylabel(ylab)
    ax.set_title(ttl, loc="left")
axs[0].set_ylim(-1.2, 0.35)
axs[1].set_ylim(0.495, 0.74)
panel(axs[0], "a")
panel(axs[1], "b")
fig.tight_layout(w_pad=1.6)
fig.savefig(FIG / "fig_link.pdf", bbox_inches="tight")
plt.close(fig)

# ===================================================================== Figure 5 (appendix): item response map
IM = pd.read_csv(RES / "model_item_means.csv", index_col=0)
iload = pd.read_csv(RES / "item_pc1_loadings.csv", index_col=0).iloc[:, 0]
ipc = pd.read_csv(RES / "model_item_pc1.csv", index_col=0).iloc[:, 0]
# orient every item so that "agree" is the PC1-positive direction, then sort
Xo = IM * np.sign(iload.reindex(IM.columns).values)
col_order = iload.abs().sort_values(ascending=False).index
row_order = ipc.sort_values(ascending=False).index
Xo = Xo.loc[row_order, col_order]
fig, ax = plt.subplots(figsize=(6.3, 2.9))
ax.imshow(Xo.values, aspect="auto", cmap=LinearSegmentedColormap.from_list(
    "it", [PAL["right"], "#F4F4F4", PAL["left"]]), vmin=-2, vmax=2, interpolation="nearest")
ax.set_yticks(range(len(row_order)))
ax.set_yticklabels([short(k) for k in row_order], fontsize=5.6)
ax.tick_params(axis="y", pad=13, length=0)
for i, kk in enumerate(row_order):
    ab = AnnotationBbox(OffsetImage(logo(kk), zoom=0.022), (-0.013, i), xycoords=("axes fraction", "data"),
                        frameon=False, box_alignment=(0.5, .5), annotation_clip=False)
    ax.add_artist(ab)
ax.set_xticks([])
inst_of = [c.split(":")[0] for c in col_order]
for j, ins in enumerate(inst_of):
    ax.add_patch(Rectangle((j - .5, len(row_order) - .5 + 0.25), 1, 0.9, color=PAL[ins], clip_on=False, lw=0))
ax.set_xlabel("178 questionnaire items, sorted by |loading| on the item-level first component "
              "(colour strip: instrument)", labelpad=8)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Item-level answers, re-oriented so that blue = the direction of the dominant dimension "
             "(rows sorted by item-level PC1)", loc="left", fontsize=7.6)
fig.savefig(FIG / "fig_items.pdf", bbox_inches="tight")
plt.close(fig)
print("figures written")

# ===================================================================== Teaser: the familiar Political Compass picture
from common import load_scores
runs = load_scores()
pct = runs.groupby("model_name")[["pct_econ", "pct_soc"]].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(3.05, 3.05))
# standard Political Compass quadrant colours (faded)
PCT_Q = {"auth_left": "#F7B6B6", "auth_right": "#A9D2F5", "lib_left": "#C2E6BE", "lib_right": "#DCCBEE"}
for (x0, y0, c) in [(-10, 0, PCT_Q["auth_left"]), (0, 0, PCT_Q["auth_right"]),
                    (-10, -10, PCT_Q["lib_left"]), (0, -10, PCT_Q["lib_right"])]:
    ax.add_patch(Rectangle((x0, y0), 10, 10, color=c, lw=0, zorder=0))
for v in range(-10, 11):
    ax.axvline(v, color="white", lw=0.5, zorder=1)
    ax.axhline(v, color="white", lw=0.5, zorder=1)
ax.axvline(0, color=PAL["ink"], lw=1.2, zorder=2)
ax.axhline(0, color=PAL["ink"], lw=1.2, zorder=2)
for kk in KEYS_ALL:
    mx, sx = pct.loc[kk, ("pct_econ", "mean")], pct.loc[kk, ("pct_econ", "std")]
    my, sy = pct.loc[kk, ("pct_soc", "mean")], pct.loc[kk, ("pct_soc", "std")]
    ax.plot([mx - sx, mx + sx], [my, my], color=PAL["pct"], lw=0.6, alpha=0.7, zorder=3)
    ax.plot([mx, mx], [my - sy, my + sy], color=PAL["pct"], lw=0.6, alpha=0.7, zorder=3)
    if kk in ("x_ai_grok_4", "google_gemini_2_5_pro"):
        ax.scatter([mx], [my], s=95, facecolor="white", edgecolor=PAL["ink"], lw=0.6, ls=(0, (1.5, 1)), zorder=4)
    put_logo(ax, mx, my, kk, zoom=0.026, z=5)
for kk, dx, dy, ha in [("google_gemini_2_5_pro", 0.0, -0.75, "center"), ("x_ai_grok_4", 0.25, -0.85, "right"),
                       ("cohere_command_r_08_2024", -0.75, 0.25, "right"),
                       ("ibm_granite_granite_3_3_8b_instruct", -0.75, 0.05, "right")]:
    mx, my = pct.loc[kk, ("pct_econ", "mean")], pct.loc[kk, ("pct_soc", "mean")]
    ax.text(mx + dx, my + dy, short(kk) + ("$^\\dagger$" if kk in ("x_ai_grok_4", "google_gemini_2_5_pro") else ""),
            fontsize=5.8, ha=ha, va="top" if dy < 0 else "center", color=PAL["ink"], zorder=6)
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(True); s_.set_color(PAL["ink"]); s_.set_linewidth(0.6)
kw = dict(fontsize=7.2, color="#555555")
ax.text(0, 10.35, "Authoritarian", ha="center", va="bottom", **kw)
ax.text(0, -10.35, "Libertarian", ha="center", va="top", **kw)
ax.text(-9.7, 0.3, "Left", ha="left", va="bottom", **kw)
ax.text(9.7, 0.3, "Right", ha="right", va="bottom", **kw)
n_ll = int(((pct[("pct_econ", "mean")] < 0) & (pct[("pct_soc", "mean")] < 0)).sum())
ax.text(5, 5, f"{n_ll} of {len(KEYS_ALL)} models\nin the libertarian-\nleft quadrant", ha="center", va="center",
        fontsize=6.8, color=PAL["ink"], style="italic")
# magnified inset of the dense cluster, placed in the empty libertarian-right quadrant
x0, x1, y0, y1 = -6.5, -1.5, -6.6, -2.1
axin = ax.inset_axes([0.515, 0.025, 0.46, 0.44])
axin.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#E3F3E0", lw=0, zorder=0))
for v in range(-7, 0):
    axin.axvline(v, color="white", lw=0.6, zorder=1); axin.axhline(v, color="white", lw=0.6, zorder=1)
for kk in KEYS_ALL:
    mx, my = pct.loc[kk, ("pct_econ", "mean")], pct.loc[kk, ("pct_soc", "mean")]
    if x0 < mx < x1 and y0 < my < y1:
        sx, sy = pct.loc[kk, ("pct_econ", "std")], pct.loc[kk, ("pct_soc", "std")]
        axin.plot([mx - sx, mx + sx], [my, my], color=PAL["pct"], lw=0.5, alpha=0.6, zorder=2)
        axin.plot([mx, mx], [my - sy, my + sy], color=PAL["pct"], lw=0.5, alpha=0.6, zorder=2)
        put_logo(axin, mx, my, kk, zoom=0.03, z=5)
axin.set_xlim(x0, x1); axin.set_ylim(y0, y1)
axin.set_xticks([-6, -4, -2]); axin.set_yticks([-6, -4])
axin.tick_params(labelsize=5, length=1.5, pad=1)
for s_ in axin.spines.values():
    s_.set_visible(True); s_.set_color(PAL["ink"]); s_.set_linewidth(0.6)
ax.indicate_inset_zoom(axin, edgecolor=PAL["ink"], alpha=0.8, lw=0.6)
fig.savefig(FIG / "fig_teaser.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("teaser written", n_ll)
