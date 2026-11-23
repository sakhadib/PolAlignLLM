"""Scoring of the three instruments, read from their public source code (data/instruments/).
Matches the recorded paper scores to rounding (max diff 0.01; 0.1 on 8Values)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

INS = Path(__file__).resolve().parent / "data" / "instruments"
PCT_IDX = {"sd": 0, "d": 1, "a": 2, "sa": 3}
MULT = {"sa": 1.0, "a": 0.5, "n": 0.0, "d": -0.5, "sd": -1.0}


def _js_array(name):
    t = (INS / name).read_text(encoding="utf-8")
    t = t[t.index("["): t.rindex("]") + 1]
    return json.loads(re.sub(r",\s*([\]}])", r"\1", t))


def _pct_tables():
    js = re.sub(r"//.*", "", (INS / "pct_script.js").read_text())
    tabs = {}
    for name in ["econv", "socv"]:
        m = re.search(name + r"\s*=\s*\[(.*?)\n\]", js, re.S)
        tabs[name] = np.array([[float(x) for x in r.split(",")]
                               for r in re.findall(r"\[([-\d\.,\s]+)\]", m.group(1))])
    e0 = float(re.search(r"var e0 = ([-\d\.]+)", js).group(1))
    s0 = float(re.search(r"var s0 = ([-\d\.]+)", js).group(1))
    return tabs["econv"], tabs["socv"], e0, s0


ECONV, SOCV, E0, S0 = _pct_tables()
EV_EFF = pd.DataFrame([q["effect"] for q in _js_array("8val_questions.js")])
SAP_EFF = pd.DataFrame([q["effects"] for q in _js_array("sap_questions.js")]).fillna(0)


def score_pct(ans):
    se = sum(ECONV[i][PCT_IDX[a]] for i, a in enumerate(ans))
    ss = sum(SOCV[i][PCT_IDX[a]] for i, a in enumerate(ans))
    return {"econ_score": round(se / 8.0 + E0, 2), "soc_score": round(ss / 19.5 + S0, 2)}


def score_8val(ans):
    m = np.array([MULT[a] for a in ans])
    out = {}
    for ax, (hi, lo) in {"econ": ("Economic_Equality", "Economic_Markets"), "dipl": ("Diplomatic_Globe", "Diplomatic_Nation"),
                         "govt": ("Govt_Liberty", "Govt_Authority"), "scty": ("Societal_Progress", "Societal_Tradition")}.items():
        w = EV_EFF[ax].values
        mx = np.abs(w).sum()
        v = round(100 * (mx + (m * w).sum()) / (2 * mx), 1)
        out[hi], out[lo] = v, round(100 - v, 1)
    return out


def score_sap(ans):
    m = np.array([MULT[a] for a in ans])
    return {f"{ax}_score": round(float((m * SAP_EFF[ax].values).sum() * 10 / np.abs(SAP_EFF[ax].values).sum()), 2)
            for ax in ["right", "auth", "prog"]}


SCORERS = {"pct": score_pct, "8val": score_8val, "sap": score_sap}
N_ITEMS = {"pct": len(ECONV), "8val": len(EV_EFF), "sap": len(SAP_EFF)}


# ---------------------------------------------------------------- probability-weighted ("soft") scores
# Each instrument's score is linear in the per-item answer, so the expected score under a model's answer
# distribution is the same formula applied to the expected per-item contribution (no rounding inside).
def soft_pct(probs):
    """probs: list (one per item) of dicts code -> probability over sd/d/a/sa."""
    se = sum(sum(p.get(c, 0) * ECONV[i][PCT_IDX[c]] for c in PCT_IDX) for i, p in enumerate(probs))
    ss = sum(sum(p.get(c, 0) * SOCV[i][PCT_IDX[c]] for c in PCT_IDX) for i, p in enumerate(probs))
    return {"econ_score": round(se / 8.0 + E0, 3), "soc_score": round(ss / 19.5 + S0, 3)}


def _exp_mult(probs):
    return np.array([sum(p.get(c, 0) * MULT[c] for c in MULT) for p in probs])


def soft_8val(probs):
    m = _exp_mult(probs)
    out = {}
    for ax, (hi, lo) in {"econ": ("Economic_Equality", "Economic_Markets"), "dipl": ("Diplomatic_Globe", "Diplomatic_Nation"),
                         "govt": ("Govt_Liberty", "Govt_Authority"), "scty": ("Societal_Progress", "Societal_Tradition")}.items():
        w = EV_EFF[ax].values
        mx = np.abs(w).sum()
        v = 100 * (mx + (m * w).sum()) / (2 * mx)
        out[hi], out[lo] = round(v, 2), round(100 - v, 2)
    return out


def soft_sap(probs):
    m = _exp_mult(probs)
    return {f"{ax}_score": round(float((m * SAP_EFF[ax].values).sum() * 10 / np.abs(SAP_EFF[ax].values).sum()), 3)
            for ax in ["right", "auth", "prog"]}


SOFT = {"pct": soft_pct, "8val": soft_8val, "sap": soft_sap}
