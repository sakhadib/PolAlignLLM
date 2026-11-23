"""News slant rating by decision models: same metrics as the chat models (analyze.py, section 6) and the
ideology -> behaviour link within the decision-model population.

    COHORT=24 python analysis/decision_news.py      (after analyze.py and decision.py)
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

from common import AXES, DATA, DMODELS, RESULTS, TAB

GT = {"Far Left": -3, "Left": -2, "Lean Left": -1, "Center": 0, "Lean Right": 1, "Right": 2, "Far Right": 3}
EXTRA_ROW = 242            # one article not in the chat models' news set; dropped so both use the same 1,062
rng = np.random.default_rng(2027)
D = DATA / "decision_news"


def within_story_concordance(df):
    num = den = 0.0
    for _, g in df.groupby("story_id"):
        y, x = g.ground_truth.values, g.pred.values
        if len(g) < 2:
            continue
        dy = np.sign(y[:, None] - y[None, :])
        dx = np.sign(x[:, None] - x[None, :])
        m = np.triu(dy != 0, 1)
        den += m.sum()
        num += ((dx == dy) & m).sum() + 0.5 * ((dx == 0) & m).sum()
    return num / den


def macro_f1_3(df, t=0.5):
    y = np.sign(df.ground_truth)
    p = np.where(df.pred <= -t, -1, np.where(df.pred >= t, 1, 0))
    return f1_score(y, p, average="macro")


per, refused = [], {}
for f in sorted(D.glob("*.csv")):
    mid = (D / (f.stem + ".model.txt")).read_text().strip()
    p = pd.read_csv(f)
    p = p[p.row != EXTRA_ROW]
    rf = D / (f.stem + ".refused.txt")
    if rf.exists():
        rows = [int(x) for x in rf.read_text().split() if int(x) != EXTRA_ROW]
        refused[mid] = p[p.row.isin(rows)].source_bias.value_counts().to_dict()
    g = p.dropna(subset=["expected"]).assign(ground_truth=lambda d: d.source_bias.map(GT), pred=lambda d: d.expected)
    lr = g[g.ground_truth != 0]
    fit = np.polyfit(g.ground_truth, g.pred, 1)
    per.append(dict(model=mid, n=len(g), auc_lr=roc_auc_score(lr.ground_truth > 0, lr.pred),
                    conc_ws=within_story_concordance(g), macro_f1=macro_f1_3(g), slope=fit[0], offset=fit[1],
                    center_mean=g[g.ground_truth == 0].pred.mean(), top_p=g.top_p.mean()))
P = pd.DataFrame(per).set_index("model")

# questionnaire PC1 of the decision models (expected axis scores), oriented like the chat models' PC1
DM = pd.read_csv(RESULTS / "decision_models.csv", index_col=0)
X = DM[[f"soft_{a}" for a in AXES]].dropna()
Z = (X - X.mean()) / X.std(ddof=1)
u, s, _ = np.linalg.svd(Z.values, full_matrices=False)
pc1 = pd.Series(u[:, 0] * s[0], index=X.index)
CM = pd.read_csv(RESULTS / "model_axis_means.csv", index_col=0)
CP = pd.read_csv(RESULTS / "model_pcs.csv", index_col=0)
chat_sign = np.sign(stats.spearmanr(CP.loc[CM.index, "PC1"], CM.sap_prog)[0])
if np.sign(stats.spearmanr(pc1, X.soft_sap_prog)[0]) != chat_sign:
    pc1 = -pc1
J = P.join(pc1.rename("PC1"), how="inner")


def boot(x, y, B=3000):
    n = len(x)
    v = [stats.spearmanr(x[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.nanpercentile(v, [2.5, 97.5])


link = {}
for y in ["offset", "center_mean", "slope", "conc_ws", "auc_lr", "macro_f1"]:
    r, pv = stats.spearmanr(J.PC1, J[y])
    lo, hi = boot(J.PC1.values, J[y].values)
    link[y] = dict(rho=float(r), p=float(pv), lo=float(lo), hi=float(hi))
P.join(pc1.rename("PC1")).to_csv(RESULTS / "decision_news_per_model.csv")
R = dict(n=len(P), n_link=len(J), link=link, refused=refused,
         med={k: float(P[k].median()) for k in ["auc_lr", "conc_ws", "macro_f1", "slope", "offset", "top_p"]},
         rng={k: [float(P[k].min()), float(P[k].max())] for k in ["auc_lr", "conc_ws", "slope", "offset"]},
         n_neg_offset=int((P.offset < 0).sum()))
json.dump(R, open(RESULTS / "decision_news.json", "w"), indent=1)


# ------------------------------------------------------------------ macros and table rows
def sg(x):
    return f"{x:+.2f}".replace("-", "$-$")


g = next(iter(refused)) if refused else None
gem = refused.get("google/gemini-decision-1", {})
N = {"DNNews": R["n"], "DNNewsAUC": f"{R['med']['auc_lr']:.2f}", "DNNewsConc": f"{R['med']['conc_ws']:.2f}",
     "DNNewsSlope": f"{R['med']['slope']:.2f}", "DNNewsOffset": sg(R["med"]["offset"]),
     "DNNewsNeg": R["n_neg_offset"], "DNNewsTopP": f"{R['med']['top_p']:.2f}",
     "DNNewsAUCLo": f"{R['rng']['auc_lr'][0]:.2f}", "DNNewsAUCHi": f"{R['rng']['auc_lr'][1]:.2f}",
     "DRhoConc": sg(link["conc_ws"]["rho"]), "DRhoConcLo": sg(link["conc_ws"]["lo"]), "DRhoConcHi": sg(link["conc_ws"]["hi"]),
     "DPConc": f"{link['conc_ws']['p']:.3f}" if link["conc_ws"]["p"] >= 0.001 else "<0.001",
     "DRhoOff": sg(link["offset"]["rho"]), "DRhoOffLo": sg(link["offset"]["lo"]), "DRhoOffHi": sg(link["offset"]["hi"]),
     "DRhoSlope": sg(link["slope"]["rho"]), "DRhoAUC": sg(link["auc_lr"]["rho"]),
     "GemRefused": sum(gem.values()), "GemRefFarRight": gem.get("Far Right", 0),
     "GemRefRight": gem.get("Right", 0), "GemRefLeftAll": gem.get("Left", 0) + gem.get("Far Left", 0),
     "GemRefMid": gem.get("Lean Left", 0) + gem.get("Lean Right", 0) + gem.get("Center", 0)}
with open(TAB / "numbers_decnews.tex", "w") as f:
    for k, v in N.items():
        f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")


def cell(y, good):
    r = link[y]
    t = sg(r["rho"]) + ("$^{\\ast}$" if r["p"] < .05 else "")
    if good and r["p"] < .05:
        return "\\cellcolor{hlGood}\\textbf{" + t + "}"
    if y == "offset":
        return "\\cellcolor{hlGray}\\textbf{" + t + "}"
    return t


row = (f"\\midrule\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{\\textcolor{{cDec}}{{Decision models}} ($n={len(J)}$)}}}} \\\\\n"
       f"Ideology PC1 & {cell('offset', False)} & {cell('center_mean', False)} & {cell('slope', True)} & "
       f"{cell('conc_ws', True)} & {cell('auc_lr', True)} \\\\")
open(TAB / "tab_link_dec.tex", "w").write(row)
print(json.dumps({k: v for k, v in R.items() if k != "refused"}, indent=1))
print("refused", refused)

# ------------------------------------------------------------------ appendix table: per-model news metrics
TT = P.join(pc1.rename("PC1")).sort_values("conc_ws", ascending=False)
best = {k: TT[k].max() for k in ["auc_lr", "conc_ws"]}
lines = []
for m, r in TT.iterrows():
    nm = f"\\model{{{DMODELS[m][2]}}}{{{DMODELS[m][0]}}}"
    au = f"{r.auc_lr:.2f}"
    co = f"{r.conc_ws:.2f}"
    if r.conc_ws >= 0.70:
        co = "\\cellcolor{hlGood}" + co
    if r.auc_lr >= 0.78:
        au = "\\cellcolor{hlGood}" + au
    off = sg(r.offset)
    if r.offset > 0:
        off = "\\cellcolor{hlRight}" + off
    n = f"{int(r.n)}" + ("" if r.n >= 1062 else "$^{\\S}$")
    lines.append(f"{nm} & {n} & {au} & {co} & {r.slope:.2f} & {off} & {r.top_p:.2f} & {sg(r.PC1)} \\\\")
open(TAB / "tab_dec_news.tex", "w").write("\n".join(lines))
