"""All statistics reported in the paper. Writes results/results.json and CSV intermediates.

Run:  python analysis/analyze.py
"""
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from common import (ROOT, KEYS, MODELS, AXES, load_scores, load_items, load_news, GOLD_ORDER, DATA, RESULTS)

OUT = RESULTS
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2027)
R = {}


def boot_ci(f, n, B=5000):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(f(idx))
        except Exception:
            pass
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


# =========================================================== 1. reliability
runs = load_scores()
R["n_models"] = int(runs.model_name.nunique())
R["n_runs"] = int(runs.groupby("model_name").size().min())
rel = {}
for a in AXES:
    groups = [g[a].values for _, g in runs.groupby("model_name")]
    F, p = stats.f_oneway(*groups)
    grand = runs[a].mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = ((runs[a] - grand) ** 2).sum()
    k = len(groups[0])
    msb = ssb / (len(groups) - 1)
    msw = (sst - ssb) / (len(runs) - len(groups))
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    rel[a] = dict(eta2=float(ssb / sst), F=float(F), p=float(p), icc1=float(icc1))
R["reliability_axes"] = rel

items = {i: load_items(i) for i in ["pct", "sap", "8val"]}
agree = {}
for inst, d in items.items():
    # share of runs that give the model's modal answer to an item
    def modal_share(s):
        return s.value_counts().iloc[0] / len(s)
    ms = d.dropna().groupby(["model", "item"])["value"].apply(modal_share)
    agree[inst] = dict(mean=float(ms.mean()),
                       per_model_min=float(ms.groupby("model").mean().min()),
                       per_model_max=float(ms.groupby("model").mean().max()))
R["item_agreement"] = agree

# =========================================================== 2. landscape
M = runs.groupby("model_name")[AXES].mean().loc[KEYS]
M.to_csv(OUT / "model_axis_means.csv")
quad = np.where(M.pct_econ < 0, "Left", "Right")
quad = [("Lib" if s < 0 else "Auth") + "-" + q for s, q in zip(M.pct_soc, quad)]
R["pct_quadrants"] = pd.Series(quad).value_counts().to_dict()
R["sap_auth_range"] = [float(M.sap_auth.min()), float(M.sap_auth.max())]

# =========================================================== 3. dimensionality
Z = (M - M.mean()) / M.std(ddof=1)
CORR = np.corrcoef(Z.values.T)
ev, vec = np.linalg.eigh(CORR)
order = np.argsort(ev)[::-1]
ev, vec = ev[order], vec[:, order]
# orient PC1 so that "more progressive" is positive
if vec[list(AXES).index("8val_scty"), 0] < 0:
    vec[:, 0] *= -1
if vec[list(AXES).index("8val_econ"), 1] < 0:
    vec[:, 1] *= -1
share = ev / ev.sum()
scores = Z.values @ vec[:, :2]
pcs = pd.DataFrame(scores, index=M.index, columns=["PC1", "PC2"])
pcs.to_csv(OUT / "model_pcs.csv")
load = pd.DataFrame(vec[:, :2], index=AXES, columns=["PC1", "PC2"])
load.to_csv(OUT / "axis_loadings.csv")


def parallel(n, p, B=2000):
    e = []
    for _ in range(B):
        X = rng.standard_normal((n, p))
        e.append(np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1])
    return np.percentile(np.array(e), 95, axis=0)


pa = parallel(len(M), len(AXES))
R["axis_pca"] = dict(share=share.tolist(), eig=ev.tolist(), parallel95=pa.tolist(),
                     n_retained=int(np.sum(ev > pa)),
                     loadings=load.round(3).to_dict())


def pc1_share(idx):
    X = Z.values[idx]
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    e = np.linalg.eigvalsh(np.corrcoef(X.T))
    return e.max() / e.sum()


R["axis_pca"]["pc1_share_ci"] = boot_ci(pc1_share, len(M))
# robustness: PC1 share without Grok 4 (the only economic-right model)
Zg = Z.drop(index="x_ai_grok_4", errors="ignore")
eg = np.linalg.eigvalsh(np.corrcoef(((Zg - Zg.mean()) / Zg.std()).values.T))
R["axis_pca"]["pc1_share_no_grok"] = float(eg.max() / eg.sum())

# item-level PCA: models x items (mean over runs), items with variance only
IM = pd.concat([d.groupby(["model", "item"])["value"].mean().unstack() for d in items.values()], axis=1).loc[KEYS]
IM = IM.loc[:, IM.std() > 1e-9]
IZ = (IM - IM.mean()) / IM.std(ddof=1)
u, s, vt = np.linalg.svd(IZ.values, full_matrices=False)
ish = s ** 2 / (s ** 2).sum()
# parallel analysis for item level: column-wise permutation of real data
perm = []
for _ in range(500):
    X = np.column_stack([rng.permutation(IZ.values[:, j]) for j in range(IZ.shape[1])])
    _, sp, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    perm.append((sp ** 2 / (sp ** 2).sum()))
perm95 = np.percentile(np.array(perm), 95, axis=0)
item_pc1 = u[:, 0] * s[0]
if np.corrcoef(item_pc1, pcs.PC1)[0, 1] < 0:
    item_pc1 = -item_pc1
R["item_pca"] = dict(n_items=int(IM.shape[1]), share=ish[:5].tolist(), perm95=perm95[:5].tolist(),
                     n_retained=int(np.sum(ish > perm95)),
                     r_with_axis_pc1=float(np.corrcoef(item_pc1, pcs.PC1)[0, 1]))
pd.Series(item_pc1, index=IM.index, name="item_PC1").to_csv(OUT / "model_item_pc1.csv")
# per-item loading on item PC1 (for the item heat-map figure)
iload = pd.Series(vt[0] * (1 if np.corrcoef(u[:, 0], item_pc1)[0, 1] > 0 else -1), index=IM.columns)
iload.to_csv(OUT / "item_pc1_loadings.csv")
IM.to_csv(OUT / "model_item_means.csv")

# share of each instrument's items whose PC1 loading is "large" (|r|>.5 with item PC1)
item_r = IM.apply(lambda c: np.corrcoef(c, item_pc1)[0, 1])
R["item_pca"]["frac_items_abs_r_gt_.5"] = {
    inst: float((item_r[[c for c in item_r.index if c.startswith(inst + ":")]].abs() > .5).mean())
    for inst in ["pct", "sap", "8val"]}
R["item_pca"]["frac_items_abs_r_gt_.5_all"] = float((item_r.abs() > .5).mean())

# directional consensus: items where all 26 models' mean answer has the same sign (excluding exact 0)
cons = {}
for inst, d in items.items():
    mm = d.groupby(["model", "item"])["value"].mean().unstack()
    sg = np.sign(mm)
    cons[inst] = dict(n_items=int(mm.shape[1]),
                      n_unanimous=int(((sg > 0).all() | (sg < 0).all()).sum()))
R["consensus"] = cons

# =========================================================== 4. MTMM (trait-aligned)
# higher = more of trait: economic-left, libertarian (anti-authority), progressive
T = pd.DataFrame({
    "Econ|pct": -M.pct_econ, "Econ|sap": -M.sap_right, "Econ|8val": M["8val_econ"],
    "Lib|pct": -M.pct_soc, "Lib|sap": -M.sap_auth, "Lib|8val": M["8val_govt"],
    "Prog|sap": M.sap_prog, "Prog|8val": M["8val_scty"],
})
T.to_csv(OUT / "mtmm_aligned_inputs.csv")
rows = []
for a, b in itertools.combinations(T.columns, 2):
    ta, ma = a.split("|")
    tb, mb = b.split("|")
    rel_t = ("monotrait-heteromethod" if ta == tb else
             "heterotrait-monomethod" if ma == mb else "heterotrait-heteromethod")
    r, p = stats.pearsonr(T[a], T[b])
    rho, prho = stats.spearmanr(T[a], T[b])
    ci = boot_ci(lambda idx: stats.spearmanr(T[a].values[idx], T[b].values[idx])[0], len(T), B=3000)
    rows.append(dict(a=a, b=b, relation=rel_t, r=r, p=p, rho=rho, p_rho=prho, rho_lo=ci[0], rho_hi=ci[1]))
mt = pd.DataFrame(rows)
mt["p_rho_fdr"] = multipletests(mt.p_rho, method="fdr_bh")[1]
mt.to_csv(OUT / "mtmm_pairs.csv", index=False)
R["mtmm"] = {f"{r.a}~{r.b}": dict(relation=r.relation, r=round(r.r, 3), rho=round(r.rho, 3),
                                  ci=[round(r.rho_lo, 2), round(r.rho_hi, 2)], p_fdr=float(r.p_rho_fdr))
             for r in mt.itertuples()}
R["mtmm_summary"] = mt.groupby("relation").rho.agg(["mean", "median", "count"]).round(3).to_dict()

# =========================================================== 5. prompt paraphrase robustness (7 other models)
pv = pd.read_csv(DATA / "prompt_variants_scores_wide.csv")
pr = {}
for a in ["pct_econ", "pct_soc", "sap_right", "sap_auth", "sap_prog", "8val_econ", "8val_dipl", "8val_govt", "8val_scty"]:
    fit = smf.ols(f"Q('{a}') ~ C(model_name) + C(prompt_varient)", data=pv).fit()
    from statsmodels.stats.anova import anova_lm
    an = anova_lm(fit, typ=2)
    sst = an.sum_sq.sum()
    pr[a] = dict(eta_model=float(an.sum_sq.iloc[0] / sst), eta_prompt=float(an.sum_sq.iloc[1] / sst),
                 p_prompt=float(an["PR(>F)"].iloc[1]))
R["prompt_robustness"] = pr
R["prompt_robustness_n_models"] = int(pv.model_name.nunique())
R["prompt_robustness_n_variants"] = int(pv.prompt_varient.nunique())

# neutral-option rates in the 7-model run (response style)
neut = {}
for inst in ["8val", "sap"]:
    d = pd.read_csv(DATA / f"prompt_variants_{inst}_items.csv")
    ms = d.columns[4:]
    neut[inst] = {m: float((d[m].astype(str).str.strip().str.upper() == "N").mean()) for m in ms}
R["prompt_variant_neutral_rate"] = neut
# neutral rates in the main cohort
R["main_neutral_rate"] = {inst: float((items[inst].value == 0).mean()) for inst in ["sap", "8val"]}
R["main_neutral_rate_max"] = {inst: float(items[inst].assign(n=items[inst].value == 0).groupby("model").n.mean().max())
                              for inst in ["sap", "8val"]}

# =========================================================== 6. news classification
news = load_news()
R["news_n_articles"] = int(news.article.nunique())
R["news_n_stories"] = int(news.story_id.nunique())
R["news_label_counts"] = news.drop_duplicates("article").source_bias.value_counts().reindex(GOLD_ORDER).to_dict()
R["news_n_pred_total"] = int(news.pred.notna().sum())
R["news_n_pred_possible"] = int(len(news))
R["news_valid_rate_min"] = float(news.groupby("model").pred.apply(lambda s: s.notna().mean()).min())
nv = news.dropna(subset=["pred"]).copy()
R["news_frac_exact_zero"] = float((nv.pred == 0).mean())

side = np.sign(nv.ground_truth)


def within_story_concordance(df):
    """Pairs of articles in the same story with different gold labels: P(pred ordered like gold)."""
    num = den = 0.0
    for _, g in df.groupby("story_id"):
        y = g.ground_truth.values
        x = g.pred.values
        if len(g) < 2:
            continue
        dy = np.sign(y[:, None] - y[None, :])
        dx = np.sign(x[:, None] - x[None, :])
        m = np.triu(dy != 0, 1)
        den += m.sum()
        num += ((dx == dy) & m).sum() + 0.5 * ((dx == 0) & m).sum()
    return num / den, den


def macro_f1_3(df, t=0.5):
    y = np.sign(df.ground_truth)
    p = np.where(df.pred <= -t, -1, np.where(df.pred >= t, 1, 0))
    return f1_score(y, p, average="macro")


per = []
for m, g in nv.groupby("model"):
    lr = g[g.ground_truth != 0]
    auc = roc_auc_score(lr.ground_truth > 0, lr.pred)
    cws, npairs = within_story_concordance(g)
    fit = np.polyfit(g.ground_truth, g.pred, 1)
    per.append(dict(model=m, n=len(g), auc_lr=auc, conc_ws=cws, n_pairs=npairs,
                    rho=stats.spearmanr(g.pred, g.ground_truth)[0],
                    macro_f1=macro_f1_3(g), slope=fit[0], offset=fit[1],
                    center_mean=g[g.ground_truth == 0].pred.mean(),
                    mae=(g.pred - g.ground_truth).abs().mean(),
                    acc_rounded=(g.pred.round().clip(-3, 3) == g.ground_truth).mean()))
P = pd.DataFrame(per).set_index("model").loc[KEYS]
P.to_csv(OUT / "news_per_model.csv")
R["news_per_model_summary"] = P.describe().round(3).to_dict()

# pooled per-category stats (article-level bootstrap via cluster on stories)
pool = nv.groupby("source_bias").pred.agg(["mean", "std", "count"]).reindex(GOLD_ORDER)
R["news_pooled_mean_by_label"] = pool["mean"].round(3).to_dict()
# pooled mixed model: pred ~ gold with random intercepts for story and model
nv["gold"] = nv.ground_truth
mm = smf.mixedlm("pred ~ gold", nv, groups=nv["model"], re_formula="~gold",
                 vc_formula={"story": "0 + C(story_id)"}).fit(method="lbfgs")
R["news_mixed"] = dict(intercept=float(mm.params["Intercept"]), slope=float(mm.params["gold"]),
                       intercept_ci=mm.conf_int().loc["Intercept"].tolist(),
                       slope_ci=mm.conf_int().loc["gold"].tolist())
# chance level & majority baseline for macro-F1
R["news_macro_f1_majority"] = float(f1_score(np.sign(nv.ground_truth), np.zeros(len(nv)), average="macro"))
# within-story pooled concordance
R["news_conc_ws_pooled"] = float(within_story_concordance(nv.assign(story_id=nv.story_id + "|" + nv.model))[0])

# =========================================================== 7. ideology -> behaviour
J = P.join(pcs)
J["item_PC1"] = pd.read_csv(OUT / "model_item_pc1.csv", index_col=0)["item_PC1"]
J = J.join(M)
link = []
for y in ["offset", "center_mean", "slope", "conc_ws", "auc_lr", "macro_f1"]:
    for x in ["PC1", "PC2", "pct_econ", "pct_soc", "sap_prog", "8val_econ", "8val_scty"]:
        rho, p = stats.spearmanr(J[x], J[y])
        ci = boot_ci(lambda idx: stats.spearmanr(J[x].values[idx], J[y].values[idx])[0], len(J), B=3000)
        link.append(dict(y=y, x=x, rho=rho, p=p, lo=ci[0], hi=ci[1]))
L = pd.DataFrame(link)
L["p_fdr"] = multipletests(L.p, method="fdr_bh")[1]
L.to_csv(OUT / "link.csv", index=False)
R["link"] = {f"{r.y}~{r.x}": dict(rho=round(r.rho, 3), ci=[round(r.lo, 2), round(r.hi, 2)],
                                  p=float(r.p), p_fdr=float(r.p_fdr)) for r in L.itertuples()}
# sensitivity: drop the 3 smallest "nano/mini-class" outliers? -> report leave-one-provider-out for conc~PC1
lopo = {}
for prov in sorted(set(v[1] for v in MODELS.values())):
    keep = [k for k in J.index if MODELS[k][1] != prov]
    if 2 <= len(J) - len(keep) or 0 < len(J) - len(keep):
        lopo[prov] = float(stats.spearmanr(J.loc[keep, "PC1"], J.loc[keep, "conc_ws"])[0])
R["link_conc_pc1_leave_one_provider_out"] = lopo
# minimum detectable |rho| at 80% power, alpha .05, n=26 (Fisher z approximation)
n = len(J)
z = (stats.norm.ppf(.975) + stats.norm.ppf(.8)) / np.sqrt(n - 3)
R["mdes_rho_80"] = float(np.tanh(z))
# open vs closed (model level)
acc = np.array([MODELS[k][3] for k in J.index])
u_ = stats.mannwhitneyu(J.PC1[acc == "closed"], J.PC1[acc == "open"])
R["open_closed_pc1"] = dict(closed_median=float(J.PC1[acc == "closed"].median()),
                            open_median=float(J.PC1[acc == "open"].median()),
                            n_closed=int((acc == "closed").sum()), n_open=int((acc == "open").sum()),
                            U=float(u_.statistic), p=float(u_.pvalue))
J.to_csv(OUT / "model_table.csv")

with open(OUT / "results.json", "w") as f:
    json.dump(R, f, indent=1, default=float)
print(json.dumps(R, indent=1, default=float)[:200000])
