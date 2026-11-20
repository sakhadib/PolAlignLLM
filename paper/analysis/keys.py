"""Item keys of the three instruments, parsed from their public source code (data/instruments/),
and exact re-implementations of their scoring functions."""
import re, json
import numpy as np
import pandas as pd
from common import DATA

INS = DATA / "instruments"


def _norm(t):
    t = re.sub(r"<.*?>", "", str(t)).replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def pct_keys():
    js = (INS / "pct_script.js").read_text()
    js = re.sub(r"//.*", "", js)
    tabs = {}
    for name in ["econv", "socv"]:
        m = re.search(name + r"\s*=\s*\[(.*?)\n\]", js, re.S)
        tabs[name] = np.array([[float(x) for x in r.split(",")] for r in re.findall(r"\[([-\d\.,\s]+)\]", m.group(1))])
    html = (INS / "pct_index.html").read_text(encoding="utf-8")
    qs = [_norm(q) for q in re.findall(r"<legend[^>]*>(.*?)</legend>", html, re.S)]
    e0 = float(re.search(r"var e0 = ([-\d\.]+)", js).group(1))
    s0 = float(re.search(r"var s0 = ([-\d\.]+)", js).group(1))
    return qs, tabs["econv"], tabs["socv"], e0, s0


def _js_array(path):
    t = (INS / path).read_text(encoding="utf-8")
    t = t[t.index("["): t.rindex("]") + 1]
    t = re.sub(r",\s*([\]}])", r"\1", t)
    return json.loads(t)


def ev_keys():
    q = _js_array("8val_questions.js")
    return [_norm(x["question"]) for x in q], pd.DataFrame([x["effect"] for x in q])


def sap_keys():
    q = _js_array("sap_questions.js")
    eff = pd.DataFrame([x["effects"] for x in q]).fillna(0)
    return [_norm(x["question"]) for x in q], eff


# answer codes -> instrument-specific values
PCT_IDX = {"sd": 0, "d": 1, "a": 2, "sa": 3}
MULT = {"sa": 1.0, "a": 0.5, "n": 0.0, "d": -0.5, "sd": -1.0}


def score_pct(ans, econv, socv, e0, s0):
    """ans: list of codes in instrument order."""
    se = sum(econv[i][PCT_IDX[a]] for i, a in enumerate(ans))
    ss = sum(socv[i][PCT_IDX[a]] for i, a in enumerate(ans))
    return se / 8.0 + e0, ss / 19.5 + s0


def score_ev(ans, eff):
    m = np.array([MULT[a] for a in ans])
    out = {}
    for ax in ["econ", "dipl", "govt", "scty"]:
        w = eff[ax].values
        mx = np.abs(w).sum()
        out[ax] = 100 * (mx + (m * w).sum()) / (2 * mx)
    return out


def score_sap(ans, eff):
    m = np.array([MULT[a] for a in ans])
    out = {}
    for ax in eff.columns:
        w = eff[ax].values
        out[ax] = (m * w).sum() * 10 / np.abs(w).sum()
    return out


def order_map(data_texts, inst_texts):
    """index into instrument order for each data item (by normalised text)."""
    pos = {t: i for i, t in enumerate(inst_texts)}
    out = []
    for t in data_texts:
        t = _norm(t)
        if t in pos:
            out.append(pos[t]); continue
        # fallback: best prefix match
        best = max(range(len(inst_texts)), key=lambda i: len(os.path.commonprefix([t, inst_texts[i]])))
        out.append(best)
    return out


import os
