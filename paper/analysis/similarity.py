"""Key-free similarity analysis of answer vectors (no scoring keys).

Each run is a vector of answers (SA=+2, A=+1, N=0, D=-1, SD=-2). Model vectors = mean over runs.
- raw vs item-centred cosine similarity (consensus vs deviations)
- within- vs between-model similarity of runs (key-free reliability)
- representational similarity analysis (RSA): do instruments agree on which models are alike?
- developer lineage: are same-developer models more alike? (permutation test; leave-one-developer-out;
  controlling for ideological distance |dPC1|)
- CSLS hubness, reported as a check
Run: COHORT=24 python analysis/similarity.py
"""
import json, re
import numpy as np
import pandas as pd
from scipy import stats
from common import KEYS, MODELS, DATA, RESULTS

OUT = RESULTS
rng = np.random.default_rng(5)
LIK = {"sd": -2, "d": -1, "n": 0, "a": 1, "sa": 2}


def runvecs(inst):
    d = pd.read_csv(DATA / f"{inst}_item_responses.csv", encoding="utf-8-sig")
    cols = [c for c in d.columns if c.startswith("pred_")]
    V = d[cols].apply(lambda s: s.astype(str).str.strip().str.lower().map(LIK)).T
    m = [re.match(r"pred_\[(\d+)\]_(.+)$", c) for c in cols]
    V.index = pd.MultiIndex.from_tuples([(x.group(2), int(x.group(1))) for x in m])
    return V.loc[KEYS]


def cos(A):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return A @ A.T


n = len(KEYS)
off = ~np.eye(n, dtype=bool)
iu = np.triu_indices(n, 1)
R = {"inst": {}}
S = {}
runs_all = []
for inst in ["pct", "sap", "8val"]:
    V = runvecs(inst)
    runs_all.append(V)
    M = V.groupby(level=0).mean().loc[KEYS].values
    Sraw, Sc = cos(M), cos(M - M.mean(0))
    Rv = V.values - V.values.mean(0)
    Sr = cos(Rv)
    lab = np.array([a for a, _ in V.index])
    same = (lab[:, None] == lab[None, :]) & ~np.eye(len(lab), dtype=bool)
    diff = lab[:, None] != lab[None, :]
    S[inst] = Sc
    R["inst"][inst] = dict(raw_between=float(Sraw[off].mean()), centred_between=float(Sc[off].mean()),
                           runs_within=float(Sr[same].mean()), runs_between=float(Sr[diff].mean()))
# all items together
Vall = pd.concat(runs_all, axis=1)
Mall = Vall.groupby(level=0).mean().loc[KEYS].values
S["all"] = cos(Mall - Mall.mean(0))
R["all_raw_between"] = float(cos(Mall)[off].mean())
pd.DataFrame(S["all"], index=KEYS, columns=KEYS).to_csv(OUT / "similarity_all.csv")
pairs = pd.DataFrame([dict(a=KEYS[i], b=KEYS[j], all=S["all"][i, j], pct=S["pct"][i, j], sap=S["sap"][i, j], ev=S["8val"][i, j])
                      for i, j in zip(*np.triu_indices(len(KEYS), 1))])
pairs.to_csv(OUT / "similarity_pairs.csv", index=False)

# RSA with a permutation (Mantel) test
def mantel(A, B, n_perm=5000):
    obs = stats.spearmanr(A[iu], B[iu])[0]
    null = []
    for _ in range(n_perm):
        p = rng.permutation(n)
        null.append(stats.spearmanr(A[iu], B[p][:, p][iu])[0])
    return float(obs), float(np.mean(np.array(null) >= obs))


R["rsa"] = {}
for a, b in [("pct", "sap"), ("pct", "8val"), ("sap", "8val")]:
    r, p = mantel(S[a], S[b])
    R["rsa"][f"{a}~{b}"] = dict(rho=r, p=p)

# developer lineage
dev = np.array([MODELS[k][1] for k in KEYS])
pcs = pd.read_csv(OUT / "model_pcs.csv", index_col=0).loc[KEYS].PC1.values


def excess(Sm, d, keep=None):
    m = np.ones(n, bool) if keep is None else keep
    sd = (d[:, None] == d[None, :]) & off & m[:, None] & m[None, :]
    nd = (d[:, None] != d[None, :]) & off & m[:, None] & m[None, :]
    return Sm[sd].mean() - Sm[nd].mean()


def perm_test(Sm, keep=None, n_perm=10000):
    obs = excess(Sm, dev, keep)
    idx = np.where(keep)[0] if keep is not None else np.arange(n)
    null = []
    for _ in range(n_perm):
        d2 = dev.copy()
        d2[idx] = rng.permutation(dev[idx])
        null.append(excess(Sm, d2, keep))
    return float(obs), float(np.mean(np.array(null) >= obs))


lin = {}
for k in ["all", "pct", "sap", "8val"]:
    lin[k] = dict(zip(["excess", "p"], perm_test(S[k])))
R["lineage"] = lin
# leave-one-developer-out (developers with >= 2 models)
multi = [d for d in pd.Series(dev).value_counts().index if (dev == d).sum() >= 2]
R["lineage_lodo"] = {}
for d in multi:
    keep = dev != d
    if len(set(dev[keep][pd.Series(dev[keep]).duplicated(keep=False).values])) == 0:
        continue
    e, p = perm_test(S["all"], keep, 5000)
    R["lineage_lodo"][d] = dict(excess=e, p=p)
# control for ideological distance: residualise pairwise similarity on |dPC1|
dpc = np.abs(pcs[:, None] - pcs[None, :])
x, y = dpc[iu], S["all"][iu]
b = np.polyfit(x, y, 1)
res = np.zeros((n, n)); res[iu] = y - np.polyval(b, x); res = res + res.T
R["sim_vs_dpc1_rho"] = float(stats.spearmanr(x, y)[0])
R["lineage_ctrl"] = dict(zip(["excess", "p"], perm_test(res)))
R["n_same_dev_pairs"] = int(((dev[:, None] == dev[None, :]) & off).sum() // 2)
R["n_openai_pairs"] = int(((dev[:, None] == "OpenAI") & (dev[None, :] == "OpenAI") & off).sum() // 2)
R["developers_multi"] = {d: int((dev == d).sum()) for d in multi}

# CSLS hubness (k=5): mean similarity to 5 nearest neighbours, per instrument
hub = {}
for inst in ["pct", "sap", "8val"]:
    s = np.sort(np.where(off, S[inst], -np.inf), axis=1)[:, ::-1][:, :5].mean(1)
    hub[inst] = s
H = pd.DataFrame(hub, index=KEYS)
hc = H.corr(method="spearman").values[np.triu_indices(3, 1)]
R["hubness_cross_inst_rho"] = [float(v) for v in hc]
R["hubness_abs_pc1_rho"] = float(stats.spearmanr(H.mean(1), np.abs(pcs))[0])

json.dump(R, open(OUT / "similarity.json", "w"), indent=1)
print(json.dumps(R, indent=1))
