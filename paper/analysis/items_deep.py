"""Instrument-level analyses using the instruments' own scoring keys (data/instruments/).

1. exact re-implementation of each scoring function, validated against the recorded scores
2. reference respondents (all-agree, all-disagree, uniform random, ...)
3. acquiescence: per-model agreement tendency, and scores after removing it
4. item decomposition: lean of every (item, axis) pair, consensus vs contested items
5. exploratory: absolutist wording vs agreement

Run (24 same-protocol models):  COHORT=24 python analysis/items_deep.py
"""
import json, re
import numpy as np
import pandas as pd
from scipy import stats

import keys as K
from common import DATA, KEYS, RESULTS, load_scores, load_items, read_items_wide

rng = np.random.default_rng(11)
OUT = RESULTS
D = {}
qs, econv, socv, e0, s0 = K.pct_keys()
et, ee = K.ev_keys()
st, se = K.sap_keys()


def raw(inst):
    d = read_items_wide(inst)
    cols = [c for c in d.columns if c.startswith("pred_")]
    meta = [re.match(r"pred_\[(\d+)\]_(.+)$", c) for c in cols]
    keep = [i for i, m in enumerate(meta) if m.group(2) in KEYS]
    cols = [cols[i] for i in keep]
    mods = np.array([meta[i].group(2) for i in keep])
    runs = np.array([int(meta[i].group(1)) for i in keep])
    A = d[cols].apply(lambda s: s.astype(str).str.strip().str.lower())
    return d, A, mods, runs


# ------------------------------------------------------------- 1. validation of re-implemented scoring
recorded = load_scores().set_index(["model_name", "run"])
diffs = []
for inst in ["pct", "8val", "sap"]:
    d, A, mods, runs = raw(inst)
    for j, (m, r) in enumerate(zip(mods, runs)):
        ans = A.iloc[:, j].tolist()
        if inst == "pct":
            e, s = K.score_pct(ans, econv, socv, e0, s0)
            got = {"pct_econ": e, "pct_soc": s}
        elif inst == "8val":
            o = K.score_ev(ans, ee)
            got = {f"8val_{k}": v for k, v in o.items()}
        else:
            o = K.score_sap(ans, se)
            got = {f"sap_{k}": v for k, v in o.items()}
        for ax, v in got.items():
            diffs.append(dict(axis=ax, scale=100 if inst == "8val" else 20, diff=abs(v - recorded.loc[(m, r), ax])))
Dv = pd.DataFrame(diffs)
D["rescore_max_abs_diff_pct_scale"] = float((Dv["diff"] / Dv["scale"] * 100).max())  # in % of axis range
D["rescore_n"] = int(len(Dv))

# ------------------------------------------------------------- 2. reference respondents
def all_scores(p, e, s):
    pe, ps = K.score_pct(p, econv, socv, e0, s0)
    o = K.score_ev(e, ee)
    q = K.score_sap(s, se)
    return {"pct_econ": pe, "pct_soc": ps, "8val_econ": o["econ"], "8val_dipl": o["dipl"], "8val_govt": o["govt"],
            "8val_scty": o["scty"], "sap_right": q["right"], "sap_auth": q["auth"], "sap_prog": q["prog"]}


ref = {lab: all_scores([c] * 62, [c] * 70, [c] * 46) for lab, c in
       [("Always strongly agree", "sa"), ("Always agree", "a"), ("Always disagree", "d"), ("Always strongly disagree", "sd")]}
sims = pd.DataFrame([all_scores(list(rng.choice(["sd", "d", "a", "sa"], 62)),
                                list(rng.choice(["sd", "d", "n", "a", "sa"], 70)),
                                list(rng.choice(["sd", "d", "n", "a", "sa"], 46))) for _ in range(10000)])
ref["Uniform random (mean)"] = sims.mean().to_dict()
pd.DataFrame(ref).T.to_csv(OUT / "reference_respondents.csv")
D["random_libleft_share"] = float(((sims.pct_econ < 0) & (sims.pct_soc < 0)).mean())
D["random_pct_mean"] = [float(sims.pct_econ.mean()), float(sims.pct_soc.mean())]
D["allagree_pct"] = [ref["Always agree"]["pct_econ"], ref["Always agree"]["pct_soc"]]
D["allagree_8v_govt"] = ref["Always agree"]["8val_govt"]

# ------------------------------------------------------------- 3. acquiescence
items = {i: load_items(i) for i in ["pct", "sap", "8val"]}
X = {}
for i, d in items.items():
    x = d.groupby(["model", "item"]).value.mean().unstack().loc[KEYS]
    X[i] = x[sorted(x.columns, key=lambda c: int(c.split(":q")[1]))]
acq = pd.concat(X.values(), axis=1).mean(axis=1)
pcs = pd.read_csv(OUT / "model_pcs.csv", index_col=0).loc[KEYS]
D["acq_rho_pc1"] = float(stats.spearmanr(acq, pcs.PC1)[0])
D["acq_rho_sap_8v"] = float(stats.spearmanr(X["sap"].mean(axis=1), X["8val"].mean(axis=1))[0])
# linear keys (+ = economic-left / libertarian / progressive / globalist)
W = {"pct_econ": ("pct", -(econv[:, 3] - econv[:, 0])), "pct_soc": ("pct", -(socv[:, 3] - socv[:, 0])),
     "sap_right": ("sap", -se["right"].values), "sap_auth": ("sap", -se["auth"].values), "sap_prog": ("sap", se["prog"].values),
     "8val_econ": ("8val", ee["econ"].values), "8val_dipl": ("8val", ee["dipl"].values),
     "8val_govt": ("8val", ee["govt"].values), "8val_scty": ("8val", ee["scty"].values)}
D["key_balance"] = {ax: float(w[w > 0].sum() / np.abs(w).sum()) for ax, (ins, w) in W.items()}


def lin(center):
    out = {}
    for ax, (ins, w) in W.items():
        x = X[ins].values
        if center:
            x = x - x.mean(axis=1, keepdims=True)
        out[ax] = (x * w).sum(axis=1) / np.abs(w).sum()
    return pd.DataFrame(out, index=KEYS)


def pc1share(Df):
    Z = (Df - Df.mean()) / Df.std()
    e = np.linalg.eigvalsh(np.corrcoef(Z.values.T))
    return float(e.max() / e.sum())


lr, lc = lin(False), lin(True)
D["acqcorr_pc1_share_raw"] = pc1share(lr)
D["acqcorr_pc1_share"] = pc1share(lc)
C = lc.corr(method="spearman")
D["acqcorr_rho_lib8v_progsap"] = float(C.loc["8val_govt", "sap_prog"])
D["acqcorr_rho_libsap_lib8v"] = float(C.loc["sap_auth", "8val_govt"])

# ------------------------------------------------------------- 4. item decomposition
rows = []
d, A, mods, _ = raw("pct")
for ax, tab in [("pct_soc", socv), ("pct_econ", econv)]:
    for i in range(62):
        row = tab[i]
        if np.abs(row).sum() == 0:
            continue
        v = row[A.iloc[i].map(K.PCT_IDX).values]
        half = (row.max() - row.min()) / 2
        lean = -(v - row.mean()) / half                     # + = toward left / libertarian
        rows.append(dict(inst="pct", axis=ax, item=f"pct:q{i+1}", text=d.question_text[i], w=half,
                         lean=lean.mean(), mm=pd.Series(lean).groupby(mods).mean()))
for inst, eff, poles in [("8val", ee, {"econ": 1, "dipl": 1, "govt": 1, "scty": 1}),
                         ("sap", se, {"right": -1, "auth": -1, "prog": 1})]:
    d, A, mods, _ = raw(inst)
    for i in range(len(d)):
        m = A.iloc[i].map(K.MULT).values
        for ax, pole in poles.items():
            w = eff[ax].values[i]
            if w == 0:
                continue
            lean = pole * np.sign(w) * m
            rows.append(dict(inst=inst, axis=f"{inst}_{ax}", item=f"{inst}:q{i+1}", text=d.question_text[i], w=abs(w),
                             lean=lean.mean(), mm=pd.Series(lean).groupby(mods).mean()))
IL = pd.DataFrame(rows)
IL["consensus"] = IL.mm.apply(lambda s: max((s > 0).mean(), (s < 0).mean()))
IL["unanimous"] = IL.mm.apply(lambda s: bool((s > 0).all() or (s < 0).all()))
IL = IL.drop(columns="mm")
ABS = (r"\b(always|never|all|only|no one|nobody|every|everyone|any|must|entirely|completely|ultimately|"
       r"superior|cannot|should not be allowed|no business|abolished|banned)\b")
IL["absolutist"] = IL.text.str.lower().str.contains(ABS, regex=True)
IL.to_csv(OUT / "item_lean.csv", index=False)

D["n_pairs"] = int(len(IL))
D["n_unan_pairs"] = int(IL.unanimous.sum())
D["n_unan_items"] = int(IL[IL.unanimous].item.nunique())
wl = lambda x: float(np.average(x.lean, weights=x.w))
D["lean_unan"] = wl(IL[IL.unanimous])
D["lean_cont"] = wl(IL[~IL.unanimous])
D["frac_pos_cont"] = float((IL[~IL.unanimous].lean > 0).mean())
D["unan_neg_pairs"] = int((IL[IL.unanimous].lean < 0).sum())
share = IL.assign(c=IL.lean * IL.w).groupby(["axis", "unanimous"]).c.sum().unstack()
share["share_unan"] = share[True] / (share[True] + share[False])
share["frac_items_unan"] = IL.groupby("axis").unanimous.mean()
share.to_csv(OUT / "item_share_by_axis.csv")
D["share_unan"] = share["share_unan"].round(3).to_dict()
D["frac_items_unan"] = share["frac_items_unan"].round(3).to_dict()
D["rho_consensus_lean"] = float(stats.spearmanr(IL.consensus, IL.lean)[0])

# ------------------------------------------------------------- 5. exploratory: wording
txt = {}
for inst in ["pct", "sap", "8val"]:
    d = read_items_wide(inst)
    txt.update({f"{inst}:{q}": t for q, t in zip(d.question_id, d.question_text)})
ag = pd.concat([d.groupby("item").value.mean() for d in items.values()])
W_ = pd.DataFrame({"agree": ag, "abs": [bool(re.search(ABS, txt[i].lower())) for i in ag.index]})
D["agree_abs"] = float(W_[W_["abs"]].agree.mean())
D["agree_nonabs"] = float(W_[~W_["abs"]].agree.mean())
D["n_abs_items"] = int(W_["abs"].sum())
D["p_abs"] = float(stats.mannwhitneyu(W_[W_["abs"]].agree, W_[~W_["abs"]].agree).pvalue)
# keying of absolutist items: does agreeing push right/auth/conservative?
keyrows = []
for i, t in enumerate(pd.read_csv(DATA / "pct_item_responses.csv", encoding="utf-8-sig").question_text):
    s = -(econv[i, 3] - econv[i, 0]) - (socv[i, 3] - socv[i, 0])
    keyrows.append((bool(re.search(ABS, t.lower())), np.sign(s)))
for i, t in enumerate(pd.read_csv(DATA / "8val_item_responses.csv", encoding="utf-8-sig").question_text):
    keyrows.append((bool(re.search(ABS, t.lower())), np.sign(ee.iloc[i].sum())))
for i, t in enumerate(pd.read_csv(DATA / "sap_item_responses.csv", encoding="utf-8-sig").question_text):
    r = se.iloc[i]
    keyrows.append((bool(re.search(ABS, t.lower())), np.sign(-r["right"] - r["auth"] + r["prog"])))
KR = pd.DataFrame(keyrows, columns=["abs", "key"])
KR = KR[KR.key != 0]
D["abs_keyed_right"] = float((KR[KR["abs"]].key < 0).mean())
D["nonabs_keyed_right"] = float((KR[~KR["abs"]].key < 0).mean())
D["p_abs_key"] = float(stats.fisher_exact(pd.crosstab(KR["abs"], KR.key).values)[1])

json.dump(D, open(OUT / "items_deep.json", "w"), indent=1)
print(json.dumps(D, indent=1))
