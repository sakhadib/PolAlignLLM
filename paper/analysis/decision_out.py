"""Tables, figures and number macros for the decision-model section.  Run after decision.py.

    COHORT=24 python analysis/decision_out.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle

from common import DMODELS, FIG, LOGO, PAL, RESULTS, TAB, AXES, KEYS, load_scores

R = json.load(open(RESULTS / "decision.json"))
DM = pd.read_csv(RESULTS / "decision_models.csv", index_col=0)
IT = pd.read_csv(RESULTS / "decision_items.csv", index_col=0)
CMP = pd.read_csv(RESULTS / "decision_mtmm_compare.csv")
CS = load_scores().groupby("model_name")[AXES].mean().loc[KEYS]

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": PAL["ink"], "axes.linewidth": 0.6,
    "savefig.dpi": 300, "pdf.fonttype": 42})
DEC = "#1F7A8C"          # decision models (one colour throughout the paper)


def f2(x, d=2):
    return f"{x:.{d}f}"


def sgn(x, d=2):
    return f"{x:+.{d}f}".replace("-", "$-$")


def pc(x):
    return f"{100 * x:.0f}"


def dlogo(m):
    return f"\\model{{{DMODELS[m][2]}}}{{{DMODELS[m][0]}}}"


# ------------------------------------------------------------------ macros
rep, chat, fmt = R["replication"]["soft"], R["chat_replication"], R["format"]
N = {
    "DNModels": R["n_models"], "DNComplete": R["n_complete"], "DNDev": R["n_developers"],
    "DNRequests": f"{R['n_requests']:,}", "DLibLeft": R["libleft_hard"], "DLibLeftSoft": R["libleft_soft"],
    "DPctEconHard": sgn(R["pct_mean_hard"][0]), "DPctSocHard": sgn(R["pct_mean_hard"][1]),
    "DPctEconSoft": sgn(R["pct_mean_soft"][0]), "DPctSocSoft": sgn(R["pct_mean_soft"][1]),
    "DChatPctEcon": sgn(R["chat_pct_mean"][0]), "DChatPctSoc": sgn(R["chat_pct_mean"][1]),
    "DShrink": pc(R["shrink_median"]), "DTopP": f2(R["top_p_mean"]), "DTopPLo": f2(R["top_p_range"][0]),
    "DTopPHi": f2(R["top_p_range"][1]), "DItemR": f2(R["item_r"]), "DItemSign": pc(R["item_sign_agree"]),
    "DNChatUnan": R["n_chat_unanimous"], "DUnanAgree": pc(R["unan_sign_agree"]),
    "DConvRho": f2(R["conv_cons_rho"]), "DTopPUnan": f2(R["top_p_unan"]), "DTopPOther": f2(R["top_p_other"]),
    "DPCOne": pc(rep["pc1"]), "DPCOneHard": pc(R["replication"]["hard"]["pc1"]), "DPCTwo": pc(rep["pc2"]),
    "DViol": rep["violations"], "DViolHard": R["replication"]["hard"]["violations"],
    "DMono": f2(rep["mono"]), "DHM": f2(rep["hm"]), "DHH": f2(rep["hh"]), "DMaxMono": f2(rep["max_mono"]),
    "DLibProgEight": f2(rep["lib8_prog8"]), "DConvSig": rep["conv_sig"], "DPattern": f2(R["mtmm_pattern_r"]),
    "CMono": f2(chat["mono"]), "CHM": f2(chat["hm"]), "CHH": f2(chat["hh"]), "CViol": chat["violations"],
    "CMaxMono": f2(chat["max_mono"]), "CLibProgSap": f2(chat["lib8_prog_sap"]), "CPCOne": pc(chat["pc1"]),
    "FSameTop": pc(fmt["same_top"]), "FSameSide": pc(fmt["same_side"]), "FRho": f2(fmt["rho"]),
    "FSameTopLo": pc(fmt["same_top_range"][0]), "FSameTopHi": pc(fmt["same_top_range"][1]),
    "FRhoLo": f2(fmt["rho_range"][0]), "FSameQuad": fmt["same_quadrant"], "FNQuad": fmt["n_quadrant"],
    "FN": fmt["n"], "LunaRefNative": len(R["refusals"].get("openai/gpt-6-luna-decisions|choice", [])),
    "LunaRefNoul": len(R["refusals"].get("openai/gpt-6-luna-decisions|noul", [])),
    "LunaRefUnion": R["luna_refused_union"],
    "DShrinkMin": pc(DM.shrink.min()), "DShrinkMinModel": DMODELS[DM.shrink.idxmin()][0],
    "DSolarNeutral": pc(DM.loc[["upstage/solar-decide", "upstage/solar-decide-flash"], "p_neutral"].mean()),
    "DOtherNeutral": pc(DM.drop(index=["upstage/solar-decide", "upstage/solar-decide-flash"]).p_neutral.mean()),
}
with open(TAB / "numbers_dec.tex", "w") as f:
    for k, v in N.items():
        f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")

# ------------------------------------------------------------------ main table: two populations side by side
H, W = "\\cellcolor{hlGood}", "\\cellcolor{hlWarn}"
rows = [
    ("\\multicolumn{3}{@{}l}{\\textit{Placement}}", None, None),
    ("Libertarian-left on PCT", f"{R['chat_libleft']}/{R['n_chat']}", f"{R['libleft_hard']}/{R['n_complete']}"),
    ("Mean PCT (econ., soc.)", f"{sgn(R['chat_pct_mean'][0])}, {sgn(R['chat_pct_mean'][1])}",
     f"{sgn(R['pct_mean_hard'][0])}, {sgn(R['pct_mean_hard'][1])}"),
    ("Item means, $r$ with chat models", "--", f"{H}{f2(R['item_r'])}"),
    ("\\multicolumn{3}{@{}l}{\\textit{Structure (\\S\\ref{sec:dimensionality}--\\ref{sec:mtmm})}}", None, None),
    ("PC1 share / comps.\\ $>$ chance", f"{pc(chat['pc1'])}\\% / {chat['n_retained']}", f"{pc(rep['pc1'])}\\% / {rep['n_retained']}"),
    ("Mean $\\rho$: same trait", f2(chat["mono"]), f2(rep["mono"])),
    ("Mean $\\rho$: diff.\\ trait, same instr.", f"{W}{f2(chat['hm'])}", f"{W}{f2(rep['hm'])}"),
    ("Mean $\\rho$: diff.\\ trait, diff.\\ instr.", f2(chat["hh"]), f2(rep["hh"])),
    ("Discriminant violations (of 21)", f"{W}{chat['violations']}", f"{W}{rep['violations']}"),
    ("\\multicolumn{3}{@{}l}{\\textit{Only measurable with probabilities}}", None, None),
    ("Mean top-option probability", "--", f2(R["top_p_mean"])),
    ("PCT distance kept by expected score", "--", f"{pc(R['shrink_median'])}\\%"),
    ("Same top answer, choice vs.\\ yes/no", "--", f"{W}{pc(fmt['same_top'])}\\%"),
    ("Same side (agree/disagree)", "--", f"{pc(fmt['same_side'])}\\%"),
    ("Same PCT quadrant", "--", f"{H}{fmt['same_quadrant']}/{fmt['n_quadrant']}"),
]
lines = []
for a, b, c in rows:
    lines.append(f"{a} \\\\" if b is None else f"\\quad {a} & {b} & {c} \\\\")
open(TAB / "tab_dec_compare.tex", "w").write("\n".join(lines))

# ------------------------------------------------------------------ appendix: per-model table
lines = []
order = DM.sort_values("hard_pct_soc", na_position="last").index
for m in order:
    r = DM.loc[m]
    if pd.isna(r.get("hard_pct_econ")):
        pos = "\\multicolumn{4}{c}{\\textit{%d items unanswered}}" % (178 - int(r.n_items))
    else:
        pos = " & ".join([sgn(r.hard_pct_econ), sgn(r.hard_pct_soc), sgn(r.soft_pct_econ), sgn(r.soft_pct_soc)])
    tp = f2(r.top_p)
    tp = (H if r.top_p >= 0.6 else "") + tp
    pn = pc(r.p_neutral)
    pn = (W if r.p_neutral >= 0.5 else "") + pn
    st = "--" if pd.isna(r.get("fmt_same_top")) else pc(r.fmt_same_top)
    rho = "--" if pd.isna(r.get("fmt_rho")) else f2(r.fmt_rho)
    if not pd.isna(r.get("fmt_rho")) and r.fmt_rho < 0.6:
        rho = W + rho
    name = dlogo(m) + ("$^{\\ddagger}$" if r.fmt == "noul" else "")
    lines.append(f"{name} & {DMODELS[m][1]} & {pos} & {tp} & {pn} & {st} & {rho} \\\\")
open(TAB / "tab_dec_models.tex", "w").write("\n".join(lines))

# ------------------------------------------------------------------ appendix: MTMM replication, pair by pair
LAB = {"Econ": "Econ", "Lib": "Lib", "Prog": "Prog"}
INS = {"pct": "\\instPCT{PCT}", "sap": "\\instSAP{Sap}", "8val": "\\instEV{8V}"}


def nm(x):
    t, i = x.split("|")
    return f"{LAB[t]}$\\cdot${INS[i]}"


REL = {"monotrait-heteromethod": "same trait", "heterotrait-monomethod": "diff.\\ trait, same instr.",
       "heterotrait-heteromethod": "diff.\\ trait, diff.\\ instr."}
lines = []
for rel in ["monotrait-heteromethod", "heterotrait-monomethod", "heterotrait-heteromethod"]:
    lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{REL[rel]}}}}} \\\\")
    for r in CMP[CMP.relation == rel].itertuples():
        cells = []
        for v, best in [(r.chat, chat["max_mono"]), (r.dec_soft, rep["max_mono"]), (r.dec_hard, R["replication"]["hard"]["max_mono"])]:
            s = sgn(v)
            if rel != "monotrait-heteromethod" and v >= best:
                s = "\\cellcolor{hlRight}" + s
            elif rel == "monotrait-heteromethod" and v < 0.3:
                s = W + s
            cells.append(s)
        lines.append(f"\\quad {nm(r.a)} -- {nm(r.b)} & " + " & ".join(cells) + " \\\\")
open(TAB / "tab_mtmm_rep.tex", "w").write("\n".join(lines))

# ------------------------------------------------------------------ main figure: PCT plane, chat vs decision
logos = {}


def put_logo(ax, x, y, f, zoom):
    if f not in logos:
        logos[f] = plt.imread(LOGO / f"{f}.png")
    ax.add_artist(AnnotationBbox(OffsetImage(logos[f], zoom=zoom), (x, y), frameon=False, zorder=6))


C = DM.dropna(subset=["hard_pct_econ"])
fig, ax = plt.subplots(figsize=(3.1, 2.75))
ax.add_patch(Rectangle((-10, -10), 10, 10, color="#E3F3E0", lw=0, zorder=0))
ax.axvline(0, color=PAL["ink"], lw=0.8, zorder=1)
ax.axhline(0, color=PAL["ink"], lw=0.8, zorder=1)
ax.scatter(CS.pct_econ, CS.pct_soc, s=16, color=PAL["muted"], edgecolor="white", lw=0.4, zorder=2,
           label=f"chat LLMs (n={len(CS)}), answers")
for m, r in C.iterrows():
    ax.annotate("", xy=(r.soft_pct_econ, r.soft_pct_soc), xytext=(r.hard_pct_econ, r.hard_pct_soc),
                arrowprops=dict(arrowstyle="-|>", color=DEC, lw=0.5, alpha=0.55, mutation_scale=5), zorder=3)
ax.scatter(C.hard_pct_econ, C.hard_pct_soc, s=14, facecolor="white", edgecolor=DEC, lw=0.7, zorder=4,
           label="decision models, most likely answer")
ax.scatter(C.soft_pct_econ, C.soft_pct_soc, s=14, color=DEC, edgecolor="white", lw=0.3, zorder=5,
           label="decision models, expected score")
for m in ["google/gemini-decision-1", "openai/gpt-6-luna-decisions", "baseten/joey-1", "upstage/solar-decide-flash"]:
    if m in C.index:
        put_logo(ax, C.loc[m, "soft_pct_econ"], C.loc[m, "soft_pct_soc"], DMODELS[m][2], 0.022)
ax.set_xlim(-8.5, 2.2)
ax.set_ylim(-8.8, 2.2)
ax.set_aspect("equal")
ax.set_xlabel("PCT economic (← left   right →)")
ax.set_ylabel("PCT social (← libertarian   auth. →)")

ax.text(-8.3, 1.9, f"mean: chat ({R['chat_pct_mean'][0]:+.1f}, {R['chat_pct_mean'][1]:+.1f})\n"
        f"decision answer ({R['pct_mean_hard'][0]:+.1f}, {R['pct_mean_hard'][1]:+.1f})\n"
        f"decision expected ({R['pct_mean_soft'][0]:+.1f}, {R['pct_mean_soft'][1]:+.1f})",
        fontsize=5.8, va="top", color=PAL["ink"])
fig.tight_layout()
fig.savefig(FIG / "fig_decision.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------------------ appendix figure: same consensus, item by item
COL = {"pct": PAL["pct"], "sap": PAL["sap"], "8val": PAL["8val"]}
NAME = {"pct": "Political Compass", "sap": "SapplyValues", "8val": "8Values"}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5))
ax = axs[0]
for t in ["pct", "sap", "8val"]:
    s = IT[IT.index.str.startswith(t + ":")]
    ax.scatter(s.chat_mean, s.dec_mean, s=9, color=COL[t], alpha=0.8, lw=0, label=NAME[t])
u = IT[IT.chat_unanimous]
ax.scatter(u.chat_mean, u.dec_mean, s=26, facecolor="none", edgecolor=PAL["ink"], lw=0.5, label="all chat models agree")
ax.plot([-2, 2], [-2, 2], color=PAL["muted"], lw=0.6, ls=":")
ax.axhline(0, color=PAL["muted"], lw=0.4)
ax.axvline(0, color=PAL["muted"], lw=0.4)
ax.set_xlabel("Chat LLMs: mean answer (−2 … +2)")
ax.set_ylabel("Decision models: mean expected answer")
ax.set_title(f"Same consensus, item by item (r = {R['item_r']:.2f})", loc="left")
ax.legend(frameon=False, fontsize=5.8, loc="upper left", handletextpad=0.2)
ax.text(-0.02, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
ax = axs[1]
for t in ["pct", "sap", "8val"]:
    s = IT[IT.index.str.startswith(t + ":")]
    ax.scatter(s.chat_mean.abs(), s.dec_top_p, s=9, color=COL[t], alpha=0.8, lw=0)
b = np.polyfit(IT.chat_mean.abs(), IT.dec_top_p, 1)
xx = np.linspace(0, IT.chat_mean.abs().max(), 20)
ax.plot(xx, np.polyval(b, xx), color=PAL["ink"], lw=0.8)
ax.set_xlabel("Chat LLMs: strength of agreement |mean answer|")
ax.set_ylabel("Decision models: mean top probability")
ax.set_title(f"More conviction where chat models agree (ρ = {R['conv_cons_rho']:.2f})", loc="left")
ax.text(-0.02, 1.04, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
fig.tight_layout(w_pad=1.6)
fig.savefig(FIG / "fig_dec_items.pdf", bbox_inches="tight")
plt.close(fig)
print("decision outputs written", len(N), "macros")
