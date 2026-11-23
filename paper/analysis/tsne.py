"""t-SNE as a non-linear check on the PCA result (the same-protocol models with 10 runs each).

Two feature sets per run: (A) the nine axis scores, standardised; (B) raw answers to all 178 items
(-2..+2, no scoring keys). For each: 2-D t-SNE for the figure, plus stability over seeds/perplexities of
(i) same-model neighbour purity, (ii) agreement between the 1-D t-SNE ordering of model centroids and PC1.
Run: COHORT=24 python analysis/tsne.py
"""
import json, re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from common import MULTI as KEYS, RESULTS, AXES, load_scores, DATA, read_items_wide

OUT = RESULTS
runs = load_scores().set_index(["model_name", "run"]).loc[KEYS]
XA = runs[AXES]
XA = (XA - XA.mean()) / XA.std()

# raw item answers per run
LIK = {"sd": -2, "d": -1, "n": 0, "a": 1, "sa": 2}
parts = []
for inst in ["pct", "sap", "8val"]:
    d = read_items_wide(inst)
    cols = [c for c in d.columns if c.startswith("pred_")]
    v = d[cols].apply(lambda s: s.astype(str).str.strip().str.lower().map(LIK)).T
    v.columns = [f"{inst}:{q}" for q in d.question_id]
    idx = [re.match(r"pred_\[(\d+)\]_(.+)$", c) for c in cols]
    v.index = pd.MultiIndex.from_tuples([(m.group(2), int(m.group(1))) for m in idx], names=["model_name", "run"])
    parts.append(v)
XB = pd.concat(parts, axis=1).loc[XA.index]
XB = XB.loc[:, XB.std() > 0]
XB = (XB - XB.mean()) / XB.std()

pcs = pd.read_csv(OUT / "model_pcs.csv", index_col=0).loc[KEYS]
labels = np.array([m for m, _ in XA.index])
R = {}


def purity(E, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    _, ind = nn.kneighbors(E)
    return float(np.mean([(labels[ind[i, 1:]] == labels[i]).mean() for i in range(len(E))]))


for name, X in [("axes", XA.values), ("items", XB.values)]:
    res = []
    for perp in [10, 30]:
        for seed in range(5):
            E2 = TSNE(2, perplexity=perp, random_state=seed, init="pca", learning_rate="auto").fit_transform(X)
            E1 = TSNE(1, perplexity=perp, random_state=seed, init="pca", learning_rate="auto").fit_transform(X)[:, 0]
            cen = pd.Series(E1, index=labels).groupby(level=0).mean().loc[KEYS]
            rho = abs(stats.spearmanr(cen, pcs.PC1)[0])
            res.append(dict(perp=perp, seed=seed, purity2d=purity(E2), rho1d=rho))
            if perp == 30 and seed == 0:
                pd.DataFrame(E2, index=XA.index, columns=["t1", "t2"]).to_csv(OUT / f"tsne2d_{name}.csv")
    df = pd.DataFrame(res)
    # chance purity: share of the other 239 points from the same model = 9/239
    R[name] = dict(purity_mean=float(df.purity2d.mean()), purity_min=float(df.purity2d.min()),
                   rho1d_mean=float(df.rho1d.mean()), rho1d_min=float(df.rho1d.min()),
                   rho1d_max=float(df.rho1d.max()), n_dims=int(X.shape[1]))
R["chance_purity"] = 9 / 239
R["n_points"] = int(len(XA))
# raw-feature kNN purity (no embedding) as a reference
R["axes"]["purity_raw"] = purity(XA.values)
R["items"]["purity_raw"] = purity(XB.values)
json.dump(R, open(OUT / "tsne.json", "w"), indent=1)
print(json.dumps(R, indent=1))
