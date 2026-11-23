"""Decision models: questionnaire placement from probabilities, conviction, format robustness, refusals,
and a replication of the dimensional (PCA) and construct-validity (MTMM) analyses in a second model population.

Run (after analyze.py with COHORT=24):  COHORT=24 python analysis/decision.py
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import scoring_inst as S
from common import (AXES, DMODELS, KEYS, RESULTS, load_decision, load_items)

OUT = RESULTS
rng = np.random.default_rng(2027)
D, _ = load_decision()
EV = {"sd": -2, "d": -1, "n": 0, "a": 1, "sa": 2}
CODES = {"pct": ["sd", "d", "a", "sa"], "8val": ["sd", "d", "n", "a", "sa"], "sap": ["sd", "d", "n", "a", "sa"]}
QIDS = {t: [f"{t}:q{i}" for i in range(1, S.N_ITEMS[t] + 1)] for t in CODES}
ALL_ITEMS = [i for t in ["pct", "sap", "8val"] for i in QIDS[t]]
R = {}


def probs(df, item):
    t = item.split(":")[0]
    r = df.loc[item]
    return {c: float(r[f"p_{c}"]) for c in CODES[t]}


def axes_from(df, soft):
    """Nine paper axes from one model's answers (df indexed by item)."""
    out = {}
    for t in ["pct", "sap", "8val"]:
        P = [probs(df, i) for i in QIDS[t]]
        sc = S.SOFT[t](P) if soft else S.SCORERS[t]([max(p, key=p.get) for p in P])
        out.update({f"{t}:{k}": v for k, v in sc.items()})
    return {"pct_econ": out["pct:econ_score"], "pct_soc": out["pct:soc_score"], "sap_right": out["sap:right_score"],
            "sap_auth": out["sap:auth_score"], "sap_prog": out["sap:prog_score"], "8val_econ": out["8val:Economic_Equality"],
            "8val_dipl": out["8val:Diplomatic_Globe"], "8val_govt": out["8val:Govt_Liberty"],
            "8val_scty": out["8val:Societal_Progress"]}


def expected(df):
    return pd.Series({i: sum(probs(df, i)[c] * EV[c] for c in CODES[i.split(':')[0]]) for i in df.index})


# ------------------------------------------------------------------ primary answers per model
# primary format = native choice; models that accept only yes/no questions (Span-01) use the yes/no encoding
prim = {}
for m in DMODELS:
    for fmt in ("choice", "noul"):
        x = D[(D.model == m) & (D.fmt == fmt)].set_index("item")
        if len(x):
            prim[m] = (fmt, x)
            break
complete = [m for m, (f, x) in prim.items() if len(x) == len(ALL_ITEMS)]
R["n_models"] = len(DMODELS)
R["n_complete"] = len(complete)
R["n_developers"] = len({DMODELS[m][1] for m in DMODELS})
R["noul_only"] = [m for m, (f, _) in prim.items() if f == "noul"]
R["n_requests"] = int(len(D))

rows = []
for m in DMODELS:
    fmt, x = prim[m]
    row = {"model": m, "fmt": fmt, "n_items": len(x)}
    p = x[[c for c in x.columns if c.startswith("p_")]]
    top = p.max(axis=1)
    k = p.notna().sum(axis=1)
    ent = -(p.clip(lower=1e-12) * np.log(p.clip(lower=1e-12))).sum(axis=1) / np.log(k)
    row.update(top_p=float(top.mean()), entropy=float(ent.mean()),
               p_neutral=float(x.loc[[i for i in x.index if not i.startswith("pct")], "p_n"].mean()))
    if m in complete:
        h, s = axes_from(x, False), axes_from(x, True)
        row.update({f"hard_{a}": v for a, v in h.items()})
        row.update({f"soft_{a}": v for a, v in s.items()})
    rows.append(row)
DM = pd.DataFrame(rows).set_index("model")

# ------------------------------------------------------------------ placement
C = DM.loc[complete]
for kind in ("hard", "soft"):
    R[f"libleft_{kind}"] = int(((C[f"{kind}_pct_econ"] < 0) & (C[f"{kind}_pct_soc"] < 0)).sum())
    R[f"pct_mean_{kind}"] = [float(C[f"{kind}_pct_econ"].mean()), float(C[f"{kind}_pct_soc"].mean())]
rad_h = np.hypot(C.hard_pct_econ, C.hard_pct_soc)
rad_s = np.hypot(C.soft_pct_econ, C.soft_pct_soc)
DM.loc[complete, "shrink"] = rad_s / rad_h
R["shrink_median"] = float((rad_s / rad_h).median())
R["top_p_mean"] = float(DM.top_p.mean())
R["top_p_range"] = [float(DM.top_p.min()), float(DM.top_p.max())]
R["entropy_mean"] = float(DM.entropy.mean())

# chat models (same-protocol cohort) for comparison
items_chat = pd.concat([load_items(i) for i in ["pct", "sap", "8val"]])
CM = items_chat.groupby(["model", "item"]).value.mean().unstack().loc[KEYS]
from common import load_scores
CS = load_scores().groupby("model_name")[AXES].mean().loc[KEYS]
R["chat_pct_mean"] = [float(CS.pct_econ.mean()), float(CS.pct_soc.mean())]
R["chat_libleft"] = int(((CS.pct_econ < 0) & (CS.pct_soc < 0)).sum())
R["n_chat"] = len(CS)

# ------------------------------------------------------------------ item level: same consensus?
EXP = pd.DataFrame({m: expected(prim[m][1]) for m in complete}).T[ALL_ITEMS]
TOP = pd.DataFrame({m: prim[m][1][[c for c in prim[m][1].columns if c.startswith("p_")]].max(axis=1)
                    for m in complete}).T[ALL_ITEMS]
chat_mean = CM[ALL_ITEMS].mean()
dec_mean = EXP.mean()
# unanimity as defined in \S7 / items_deep.py: every chat model pushes a scored axis the same way
IL = pd.read_csv(OUT / "item_lean.csv")
unan_items = set(IL[IL.unanimous].item)
IT = pd.DataFrame({"chat_mean": chat_mean, "dec_mean": dec_mean, "dec_top_p": TOP.mean()})
IT["chat_unanimous"] = [i in unan_items for i in IT.index]
IT.to_csv(OUT / "decision_items.csv")
R["item_r"] = float(stats.pearsonr(IT.chat_mean, IT.dec_mean)[0])
R["item_rho"] = float(stats.spearmanr(IT.chat_mean, IT.dec_mean)[0])
nz = IT[(IT.chat_mean != 0) & (IT.dec_mean != 0)]
R["item_sign_agree"] = float((np.sign(nz.chat_mean) == np.sign(nz.dec_mean)).mean())
U = IT[IT.chat_unanimous]
R["n_chat_unanimous"] = int(len(U))
R["unan_sign_agree"] = float((np.sign(U.chat_mean) == np.sign(U.dec_mean)).mean())
# do decision models also agree among themselves on these items?
dec_unan = ((np.sign(EXP) > 0).all() | (np.sign(EXP) < 0).all())
R["dec_unanimous"] = int(dec_unan.sum())
R["unan_both"] = int((dec_unan & IT.chat_unanimous).sum())
# conviction follows consensus: decision models are most sure where chat models agree most
R["conv_cons_rho"] = float(stats.spearmanr(IT.chat_mean.abs(), IT.dec_top_p)[0])
R["conv_cons_p"] = float(stats.spearmanr(IT.chat_mean.abs(), IT.dec_top_p)[1])
R["top_p_unan"] = float(IT[IT.chat_unanimous].dec_top_p.mean())
R["top_p_other"] = float(IT[~IT.chat_unanimous].dec_top_p.mean())

# ------------------------------------------------------------------ replication: PCA and MTMM
def pca_share(M):
    Z = (M - M.mean()) / M.std(ddof=1)
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.values.T)))[::-1]
    pa = []
    for _ in range(2000):
        X = rng.standard_normal(M.shape)
        pa.append(np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1])
    pa = np.percentile(pa, 95, axis=0)
    return float(ev[0] / ev.sum()), int(np.sum(ev > pa)), float(ev[1] / ev.sum())


def mtmm(M):
    T = pd.DataFrame({"Econ|pct": -M.pct_econ, "Econ|sap": -M.sap_right, "Econ|8val": M["8val_econ"],
                      "Lib|pct": -M.pct_soc, "Lib|sap": -M.sap_auth, "Lib|8val": M["8val_govt"],
                      "Prog|sap": M.sap_prog, "Prog|8val": M["8val_scty"]})
    rows = []
    for a, b in itertools.combinations(T.columns, 2):
        rel = ("monotrait-heteromethod" if a.split("|")[0] == b.split("|")[0] else
               "heterotrait-monomethod" if a.split("|")[1] == b.split("|")[1] else "heterotrait-heteromethod")
        rho, p = stats.spearmanr(T[a], T[b])
        rows.append(dict(a=a, b=b, relation=rel, rho=rho, p=p))
    mt = pd.DataFrame(rows)
    mt["p_fdr"] = multipletests(mt.p, method="fdr_bh")[1]
    best = {s: mt[((mt.a == s) | (mt.b == s)) & (mt.relation == "monotrait-heteromethod")].rho.max() for s in T.columns}
    viol = int(sum(1 for r in mt.itertuples() if r.a.split("|")[0] != r.b.split("|")[0]
                   and r.rho >= min(best[r.a], best[r.b])))
    g = mt.groupby("relation").rho.mean()
    lp = mt[(mt.a == "Lib|8val") & (mt.b == "Prog|sap") | (mt.a == "Lib|sap") & (mt.b == "Prog|sap")]
    return mt, dict(mono=float(g["monotrait-heteromethod"]), hh=float(g["heterotrait-heteromethod"]),
                    hm=float(g["heterotrait-monomethod"]), violations=viol,
                    conv_sig=int(((mt.relation == "monotrait-heteromethod") & (mt.p_fdr < .05)).sum()),
                    max_mono=float(mt[mt.relation == "monotrait-heteromethod"].rho.max()))


rep = {}
for kind in ("soft", "hard"):
    M = C[[f"{kind}_{a}" for a in AXES]].rename(columns=lambda c: c[len(kind) + 1:])
    share, nret, share2 = pca_share(M)
    mt, summ = mtmm(M)
    mt.to_csv(OUT / f"decision_mtmm_{kind}.csv", index=False)
    rep[kind] = dict(pc1=share, pc2=share2, n_retained=nret, **summ,
                     lib8_prog_sap=float(mt[(mt.a == "Lib|8val") & (mt.b == "Prog|sap")].rho.iloc[0]),
                     lib8_prog8=float(mt[(mt.a == "Lib|8val") & (mt.b == "Prog|8val")].rho.iloc[0]))
R["replication"] = rep
# chat MTMM for the side-by-side table
chat_mt, chat_summ = mtmm(CS)
cs, cn, cs2 = pca_share(CS)
R["chat_replication"] = dict(pc1=cs, pc2=cs2, n_retained=cn, **chat_summ,
                             lib8_prog_sap=float(chat_mt[(chat_mt.a == "Lib|8val") & (chat_mt.b == "Prog|sap")].rho.iloc[0]),
                             lib8_prog8=float(chat_mt[(chat_mt.a == "Lib|8val") & (chat_mt.b == "Prog|8val")].rho.iloc[0]))
dm = pd.read_csv(OUT / "decision_mtmm_soft.csv")
R["mtmm_pattern_r"] = float(stats.spearmanr(chat_mt.rho.values, dm.rho.values)[0])
pd.DataFrame({"a": chat_mt.a, "b": chat_mt.b, "relation": chat_mt.relation, "chat": chat_mt.rho,
              "dec_soft": dm.rho, "dec_hard": pd.read_csv(OUT / "decision_mtmm_hard.csv").rho}).to_csv(
    OUT / "decision_mtmm_compare.csv", index=False)

# ------------------------------------------------------------------ answer-format robustness (choice vs yes/no)
fr = []
for m in DMODELS:
    a = D[(D.model == m) & (D.fmt == "choice")].set_index("item")
    b = D[(D.model == m) & (D.fmt == "noul")].set_index("item")
    both = a.index.intersection(b.index)
    if not len(both):
        continue
    ea, eb = expected(a.loc[both]), expected(b.loc[both])
    ta = [max(probs(a, i), key=probs(a, i).get) for i in both]
    tb = [max(probs(b, i), key=probs(b, i).get) for i in both]
    row = dict(model=m, n=len(both), same_top=float(np.mean([x == y for x, y in zip(ta, tb)])),
               same_side=float(np.mean(np.sign(ea) == np.sign(eb))), rho=float(stats.spearmanr(ea, eb)[0]))
    if len(a) == len(b) == len(ALL_ITEMS):
        hb = axes_from(b, False)
        row["same_quadrant"] = bool((np.sign(hb["pct_econ"]), np.sign(hb["pct_soc"])) ==
                                    (np.sign(DM.loc[m, "hard_pct_econ"]), np.sign(DM.loc[m, "hard_pct_soc"])))
        row["noul_pct_econ"], row["noul_pct_soc"] = hb["pct_econ"], hb["pct_soc"]
    fr.append(row)
FR = pd.DataFrame(fr).set_index("model")
FR.to_csv(OUT / "decision_format.csv")
DM = DM.join(FR.add_prefix("fmt_"))
R["format"] = dict(n=len(FR), same_top=float(FR.same_top.median()), same_side=float(FR.same_side.median()),
                   rho=float(FR.rho.median()), same_top_range=[float(FR.same_top.min()), float(FR.same_top.max())],
                   rho_range=[float(FR.rho.min()), float(FR.rho.max())],
                   same_quadrant=int(FR.same_quadrant.sum()), n_quadrant=int(FR.same_quadrant.notna().sum()),
                   unstable=[m for m in FR.index if FR.loc[m, "rho"] < 0.6])

# ------------------------------------------------------------------ refusals (requests that never returned an answer)
ref = {}
for m in DMODELS:
    for fmt in ("choice", "noul"):
        x = D[(D.model == m) & (D.fmt == fmt)]
        if len(x) and len(x) < len(ALL_ITEMS):
            ref[f"{m}|{fmt}"] = sorted(set(ALL_ITEMS) - set(x["item"]))
R["refusals"] = ref
luna = "openai/gpt-6-luna-decisions"
R["luna_refused_union"] = len(set().union(*[set(v) for k, v in ref.items() if k.startswith(luna)]))
R["luna_refused_intersection"] = len(set.intersection(*[set(v) for k, v in ref.items() if k.startswith(luna)]))
R["luna_answered_by_one"] = len(ALL_ITEMS) - R["luna_refused_intersection"]
DM.to_csv(OUT / "decision_models.csv")
json.dump(R, open(OUT / "decision.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ("refusals",)}, indent=1, default=float))
