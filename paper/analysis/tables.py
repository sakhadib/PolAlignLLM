"""LaTeX tables and number macros generated from results/ (no hand-copied numbers).
Run after analyze.py:  python analysis/tables.py"""
import json
import pandas as pd
from common import ROOT, TAB, MODELS, KEYS, GOLD_ORDER

RES = ROOT / "results24"          # questionnaire + link analyses: 24 same-protocol models
RESN = ROOT / "results"            # news task: all 26 models (one protocol)
R = json.load(open(RES / "results.json"))
RN = json.load(open(RESN / "results.json"))
KEYS24 = [k for k in KEYS if k not in ("x_ai_grok_4", "google_gemini_2_5_pro")]
TAB.mkdir(exist_ok=True)


def f2(x):
    return f"{x:.2f}"


def sgn(x, d=2):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$")


def mlogo(k):
    return f"\\model{{{MODELS[k][2]}}}{{{MODELS[k][0]}}}"


# ------------------------------------------------------------------ number macros
L = pd.read_csv(RES / "link.csv")
mt = pd.read_csv(RES / "mtmm_pairs.csv")
P = pd.read_csv(RESN / "news_per_model.csv", index_col=0).loc[KEYS]
rel = R["reliability_axes"]


def mtr(a, b):
    r = mt[((mt.a == a) & (mt.b == b)) | ((mt.a == b) & (mt.b == a))].iloc[0]
    return r


def lk(y, x):
    return L[(L.y == y) & (L.x == x)].iloc[0]


N = {}
N["NModels"] = R["n_models"]
N["NModelsAll"] = RN["n_models"]
N["NRuns"] = R["n_runs"]
N["EtaMin"] = f2(min(v["eta2"] for v in rel.values()))
N["EtaMax"] = f2(max(v["eta2"] for v in rel.values()))
N["EtaMinOther"] = f2(min(v["eta2"] for k, v in rel.items() if k != "sap_auth"))
N["EtaSapAuth"] = f2(rel["sap_auth"]["eta2"])
N["AgreeMean"] = f"{100*sum(v['mean'] for v in R['item_agreement'].values())/3:.0f}"
N["AgreeMinModel"] = f"{100*min(v['per_model_min'] for v in R['item_agreement'].values()):.0f}"
N["LibLeftCount"] = R["pct_quadrants"].get("Lib-Left", 0)
N["PCOneShare"] = f"{100*R['axis_pca']['share'][0]:.0f}"
N["PCTwoShare"] = f"{100*R['axis_pca']['share'][1]:.0f}"
N["PCOneLo"] = f"{100*R['axis_pca']['pc1_share_ci'][0]:.0f}"
N["PCOneHi"] = f"{100*R['axis_pca']['pc1_share_ci'][1]:.0f}"
N["PCOneNoGrok"] = f"{100*R['axis_pca']['pc1_share_no_grok']:.0f}"
N["ItemPCOne"] = f"{100*R['item_pca']['share'][0]:.0f}"
N["ItemPCTwo"] = f"{100*R['item_pca']['share'][1]:.0f}"
N["ItemPerm"] = f"{100*R['item_pca']['perm95'][0]:.1f}"
N["ItemRatio"] = f"{R['item_pca']['share'][0]/R['item_pca']['perm95'][0]:.1f}"
N["ItemRAxis"] = f2(R["item_pca"]["r_with_axis_pc1"])
N["ItemFracHalf"] = f"{100*R['item_pca']['frac_items_abs_r_gt_.5_all']:.0f}"
N["NItems"] = R["item_pca"]["n_items"]
N["ConsUnan"] = sum(v["n_unanimous"] for v in R["consensus"].values())
N["RhoLibProgSap"] = f2(mtr("Lib|8val", "Prog|sap").rho)
N["RhoLibProgEight"] = f2(mtr("Lib|8val", "Prog|8val").rho)
N["RhoProgConv"] = f2(mtr("Prog|sap", "Prog|8val").rho)
N["RhoLibPctEight"] = f2(mtr("Lib|pct", "Lib|8val").rho)
N["RhoLibPctSap"] = sgn(mtr("Lib|pct", "Lib|sap").rho)
N["RhoLibSapEight"] = sgn(mtr("Lib|sap", "Lib|8val").rho)
N["RhoEconPctSap"] = f2(mtr("Econ|pct", "Econ|sap").rho)
N["RhoEconPctEight"] = f2(mtr("Econ|pct", "Econ|8val").rho)
N["RhoEconSapEight"] = f2(mtr("Econ|sap", "Econ|8val").rho)
N["RhoLibPctProgEight"] = f2(mtr("Lib|pct", "Prog|8val").rho)
N["RhoLibPctProgSap"] = f2(mtr("Lib|pct", "Prog|sap").rho)
ms = R["mtmm_summary"]["mean"]
N["MeanMono"] = f2(ms["monotrait-heteromethod"])
N["MeanHH"] = f2(ms["heterotrait-heteromethod"])
N["MeanHM"] = f2(ms["heterotrait-monomethod"])
N["NArticles"] = f"{RN['news_n_articles']:,}"
N["NStories"] = RN["news_n_stories"]
N["NPreds"] = f"{RN['news_n_pred_total']:,}"
N["NPredsPossible"] = f"{RN['news_n_pred_possible']:,}"
N["ValidMin"] = f"{100*RN['news_valid_rate_min']:.0f}"
N["FracZero"] = f"{100*RN['news_frac_exact_zero']:.1f}"
mx = RN["news_mixed"]
N["MixSlope"] = f2(mx["slope"])
N["MixSlopeLo"] = f2(mx["slope_ci"][0])
N["MixSlopeHi"] = f2(mx["slope_ci"][1])
N["MixInt"] = sgn(mx["intercept"])
N["MixIntLo"] = sgn(mx["intercept_ci"][0])
N["MixIntHi"] = sgn(mx["intercept_ci"][1])
N["ConcPooled"] = f2(RN["news_conc_ws_pooled"])
N["ConcMin"] = f2(P.conc_ws.min())
N["ConcMax"] = f2(P.conc_ws.max())
N["AUCMin"] = f2(P.auc_lr.min())
N["AUCMax"] = f2(P.auc_lr.max())
N["AUCMed"] = f2(P.auc_lr.median())
N["FOneMin"] = f2(P.macro_f1.min())
N["FOneMax"] = f2(P.macro_f1.max())
N["FOneMaj"] = f2(RN["news_macro_f1_majority"])
N["OffsetMin"] = sgn(P.offset.min())
N["OffsetMax"] = sgn(P.offset.max())
N["NegOffsetCount"] = int((P.offset < 0).sum())
N["SlopeMin"] = f2(P.slope.min())
N["SlopeMax"] = f2(P.slope.max())
lab = RN["news_label_counts"]
N["NFarLeft"], N["NFarRight"] = lab["Far Left"], lab["Far Right"]
for y, x, nm in [("offset", "PC1", "OffPC"), ("conc_ws", "PC1", "ConcPC"), ("center_mean", "PC1", "CenPC"),
                 ("slope", "PC1", "SlopePC"), ("auc_lr", "PC1", "AucPC"), ("conc_ws", "sap_prog", "ConcProg"),
                 ("conc_ws", "pct_soc", "ConcPctSoc")]:
    r = lk(y, x)
    N[f"Rho{nm}"] = sgn(r.rho)
    N[f"Rho{nm}Lo"] = sgn(r.lo)
    N[f"Rho{nm}Hi"] = sgn(r.hi)
    N[f"P{nm}"] = f"{r.p_fdr:.3f}"
N["MaxAbsOffsetRho"] = f2(L[L.y.isin(["offset", "center_mean", "slope"])].rho.abs().max())
N["MDES"] = f2(R["mdes_rho_80"])
lopo = R["link_conc_pc1_leave_one_provider_out"]
N["LopoMin"] = f2(min(lopo.values()))
N["LopoMax"] = f2(max(lopo.values()))
oc = R["open_closed_pc1"]
N["OCp"] = f"{oc['p']:.3f}"
N["NClosed"], N["NOpen"] = oc["n_closed"], oc["n_open"]
pr = R["prompt_robustness"]
N["PromptEtaMax"] = f2(max(v["eta_prompt"] for k, v in pr.items() if k != "sap_auth"))
N["PromptEtaSapAuth"] = f2(pr["sap_auth"]["eta_prompt"])
N["ModelEtaSapAuth"] = f2(pr["sap_auth"]["eta_model"])
N["ModelEtaMinOther"] = f2(min(v["eta_model"] for k, v in pr.items() if k != "sap_auth"))
N["NeutralSap"] = f"{100*R['main_neutral_rate']['sap']:.1f}"
N["NeutralEight"] = f"{100*R['main_neutral_rate']['8val']:.1f}"

# sensitivity: questionnaire analyses on all 26 models (incl. the two differently-prompted ones)
L26 = pd.read_csv(RESN / "link.csv")
mt26 = pd.read_csv(RESN / "mtmm_pairs.csv")
N["SPCOne"] = f"{100*RN['axis_pca']['share'][0]:.0f}"
N["SPCTwo"] = f"{100*RN['axis_pca']['share'][1]:.0f}"
r_ = L26[(L26.y == "offset") & (L26.x == "PC1")].iloc[0]; N["SRhoOff"] = sgn(r_.rho)
r_ = L26[(L26.y == "conc_ws") & (L26.x == "PC1")].iloc[0]; N["SRhoConc"] = sgn(r_.rho); N["SPConc"] = f"{r_.p_fdr:.3f}"
N["SMeanMono"] = f2(RN["mtmm_summary"]["mean"]["monotrait-heteromethod"])
N["SMeanHH"] = f2(RN["mtmm_summary"]["mean"]["heterotrait-heteromethod"])
N["SRhoLibProgSap"] = f2(mt26[(mt26.a == "Lib|8val") & (mt26.b == "Prog|sap")].iloc[0].rho)
N["ItemNRetained"] = R["item_pca"]["n_retained"]
N["AxisNRetained"] = R["axis_pca"]["n_retained"]

with open(TAB / "numbers.tex", "w") as f:
    for k, v in N.items():
        f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")

# ------------------------------------------------------------------ Table: reliability (main)
AX = [("pct", "pct_econ", "Economic"), ("pct", "pct_soc", "Social"),
      ("sap", "sap_right", "Right"), ("sap", "sap_auth", "Authority"), ("sap", "sap_prog", "Progressive"),
      ("8val", "8val_econ", "Equality"), ("8val", "8val_dipl", "Globalism"), ("8val", "8val_govt", "Liberty"),
      ("8val", "8val_scty", "Progress")]
INST = {"pct": "\\instPCT{PCT}", "sap": "\\instSAP{Sapply}", "8val": "\\instEV{8Values}"}
lines = []
prev = None
for inst, a, nm in AX:
    first = inst != prev
    if first and prev is not None:
        lines.append("\\midrule")
    cnt = sum(1 for x in AX if x[0] == inst)
    lab_ = f"\\multirow{{{cnt}}}{{*}}{{{INST[inst]}}}" if first else ""
    agr = f"\\multirow{{{cnt}}}{{*}}{{{100*R['item_agreement'][inst]['mean']:.1f}}}" if first else ""
    p = pr[a]
    c = [nm, f"{rel[a]['eta2']:.2f}", f"{rel[a]['icc1']:.2f}", f"{p['eta_model']:.2f}", f"{p['eta_prompt']:.3f}"]
    if a == "sap_auth":
        c = [f"\\cellcolor{{hlWarn}}{x}" for x in c]
    lines.append(f"{lab_} & " + " & ".join(c) + f" & {agr} \\\\")
    prev = inst
with open(TAB / "tab_reliability.tex", "w") as f:
    f.write("\n".join(lines))

# ------------------------------------------------------------------ Table: link (main)
names = {"PC1": "Ideology PC1", "PC2": "Ideology PC2", "sap_prog": "Sapply progressive", "pct_soc": "PCT social",
         "pct_econ": "PCT economic", "8val_scty": "8V progress", "8val_econ": "8V equality"}
outs = [("offset", "Offset"), ("center_mean", "Center mean"), ("slope", "Slope"), ("conc_ws", "Within-story conc."),
        ("auc_lr", "AUC (L vs.\\ R)")]
lines = []
for x in ["PC1", "PC2", "pct_econ", "pct_soc", "sap_prog", "8val_scty"]:
    cells = []
    for y, _ in outs:
        r = lk(y, x)
        star = "$^{\\ast}$" if r.p_fdr < .05 else ""
        txt = f"{sgn(r.rho)}{star}"
        if r.p_fdr < .05:
            txt = f"\\cellcolor{{hlGood}}\\textbf{{{txt}}}"
        elif x == "PC1" and y == "offset":
            txt = f"\\cellcolor{{hlGray}}\\textbf{{{txt}}}"
        cells.append(txt)
    lines.append(f"{names[x]} & " + " & ".join(cells) + " \\\\")
with open(TAB / "tab_link.tex", "w") as f:
    f.write("\n".join(lines))

# ------------------------------------------------------------------ Appendix: model table with logos
pcs = pd.read_csv(RES / "model_pcs.csv", index_col=0)
M = pd.read_csv(RESN / "model_axis_means.csv", index_col=0)
lines = []
for k in list(pcs.loc[KEYS24].sort_values("PC1", ascending=False).index) + ["google_gemini_2_5_pro", "x_ai_grok_4"]:
    _, prov, _, acc = MODELS[k]
    m = M.loc[k]
    pc = sgn(pcs.loc[k, "PC1"]) if k in pcs.index else "--"
    if k == "google_gemini_2_5_pro":
        lines.append("\\midrule")
    nm = mlogo(k) + ("$^{\\dagger}$" if k not in pcs.index else "")
    lines.append(f"{nm} & {prov} & {acc} & {pc} & {sgn(m.pct_econ)} & {sgn(m.pct_soc)} & "
                 f"{sgn(m.sap_right)} & {sgn(m.sap_auth)} & {sgn(m.sap_prog)} & {m['8val_econ']:.0f} & "
                 f"{m['8val_dipl']:.0f} & {m['8val_govt']:.0f} & {m['8val_scty']:.0f} \\\\")
with open(TAB / "tab_models.tex", "w") as f:
    f.write("\n".join(lines))

# ------------------------------------------------------------------ Appendix: per-model news metrics with logos
lines = []
for k in P.sort_values("conc_ws", ascending=False).index:
    p = P.loc[k]
    lines.append(f"{mlogo(k)} & {int(p.n)} & {p.auc_lr:.3f} & {p.conc_ws:.3f} & {p.macro_f1:.3f} & {p.rho:.2f} & "
                 f"{p.slope:.2f} & {sgn(p.offset)} & {p.mae:.2f} \\\\")
with open(TAB / "tab_news_models.tex", "w") as f:
    f.write("\n".join(lines))

# ------------------------------------------------------------------ Appendix: full MTMM pairs
lab2 = {"Econ": "Econ", "Lib": "Lib", "Prog": "Prog", "pct": "PCT", "sap": "Sap", "8val": "8V"}
rel_short = {"monotrait-heteromethod": "MT-HM", "heterotrait-heteromethod": "HT-HM", "heterotrait-monomethod": "HT-MM"}
lines = []
for r in mt.sort_values(["relation", "rho"], ascending=[False, False]).itertuples():
    a = "--".join(lab2[x] for x in r.a.split("|"))
    b = "--".join(lab2[x] for x in r.b.split("|"))
    lines.append(f"{a} & {b} & {rel_short[r.relation]} & {sgn(r.rho)} & [{sgn(r.rho_lo)}, {sgn(r.rho_hi)}] & {r.p_rho_fdr:.3f} \\\\")
with open(TAB / "tab_mtmm_full.tex", "w") as f:
    f.write("\n".join(lines))
print(f"wrote {len(N)} number macros and tables")

# ------------------------------------------------------------------ instrument-key analyses (items_deep.py)
DQ = json.load(open(RES / "items_deep.json"))
N2 = {}
N2["RescoreMaxDiff"] = f"{DQ['rescore_max_abs_diff_pct_scale']:.2f}"
N2["RescoreN"] = f"{DQ['rescore_n']:,}"
N2["RandLibLeft"] = f"{100*DQ['random_libleft_share']:.0f}"
N2["AllAgreeEcon"] = sgn(DQ["allagree_pct"][0])
N2["AllAgreeSoc"] = sgn(DQ["allagree_pct"][1])
N2["AllAgreeLib"] = f"{DQ['allagree_8v_govt']:.1f}"
N2["KeyBalPctSoc"] = f"{100*DQ['key_balance']['pct_soc']:.0f}"
N2["KeyBalEvGovt"] = f"{100*DQ['key_balance']['8val_govt']:.0f}"
N2["AcqRhoPC"] = sgn(DQ["acq_rho_pc1"])
N2["AcqRhoInst"] = f2(DQ["acq_rho_sap_8v"])
N2["AcqCorrPCOne"] = f"{100*DQ['acqcorr_pc1_share']:.0f}"
N2["AcqCorrLibProg"] = f2(DQ["acqcorr_rho_lib8v_progsap"])
N2["NPairs"] = DQ["n_pairs"]
N2["NUnanPairs"] = DQ["n_unan_pairs"]
N2["NUnanItems"] = DQ["n_unan_items"]
N2["UnanNeg"] = DQ["unan_neg_pairs"]
N2["UnanPos"] = DQ["n_unan_pairs"] - DQ["unan_neg_pairs"]
N2["LeanUnan"] = f2(DQ["lean_unan"])
N2["LeanCont"] = f2(DQ["lean_cont"])
N2["FracPosCont"] = f"{100*DQ['frac_pos_cont']:.0f}"
su, fu = DQ["share_unan"], DQ["frac_items_unan"]
N2["ShareUnanPctSoc"] = f"{100*su['pct_soc']:.0f}"
N2["FracUnanPctSoc"] = f"{100*fu['pct_soc']:.0f}"
N2["ShareUnanPctEcon"] = f"{100*su['pct_econ']:.0f}"
N2["FracUnanPctEcon"] = f"{100*fu['pct_econ']:.0f}"
N2["ShareUnanEvScty"] = f"{100*su['8val_scty']:.0f}"
N2["NAxesOverShare"] = sum(1 for a in su if 0 <= su[a] <= 1 and su[a] > fu[a])
N2["AgreeAbs"] = sgn(DQ["agree_abs"])
N2["AgreeNonAbs"] = sgn(DQ["agree_nonabs"])
N2["NAbsItems"] = DQ["n_abs_items"]
N2["PAbs"] = f"{DQ['p_abs']:.3f}"
N2["AbsKeyedRight"] = f"{100*DQ['abs_keyed_right']:.0f}"
N2["NonAbsKeyedRight"] = f"{100*DQ['nonabs_keyed_right']:.0f}"
N2["PAbsKey"] = f"{DQ['p_abs_key']:.2f}"
with open(TAB / "numbers.tex", "a") as f:
    for k, v in N2.items():
        f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")

# reference respondents table
RR = pd.read_csv(RES / "reference_respondents.csv", index_col=0)
lines = []
for lab, r in RR.iterrows():
    lines.append(f"{lab} & {sgn(r.pct_econ)} & {sgn(r.pct_soc)} & {sgn(r.sap_right)} & {sgn(r.sap_auth)} & "
                 f"{sgn(r.sap_prog)} & {r['8val_econ']:.1f} & {r['8val_dipl']:.1f} & {r['8val_govt']:.1f} & {r['8val_scty']:.1f} \\\\")
with open(TAB / "tab_reference.tex", "w") as f:
    f.write("\n".join(lines))

# consensus (unanimous) items table: one row per item, its strongest-weight axis
IL = pd.read_csv(RES / "item_lean.csv")
U = IL[IL.unanimous].sort_values("w", ascending=False).drop_duplicates("item").sort_values("lean", ascending=False)
inst_cmd = {"pct": "\\instPCT{PCT}", "sap": "\\instSAP{Sap}", "8val": "\\instEV{8V}"}
def tex_esc(t):
    t = t.replace("’", "'").replace("“", "``").replace("”", "''")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_")]:
        t = t.replace(a, b)
    return t
lines = [f"{inst_cmd[r.inst]} & {tex_esc(r.text)} & {sgn(r.lean)} \\\\" for r in U.itertuples()]
with open(TAB / "tab_unanimous.tex", "w") as f:
    f.write("\n".join(lines))
print("item tables written", len(U))

# =================================================================== APPENDIX v2: highlighted tables
def hl(color, txt):
    return f"\\cellcolor{{{color}}}{txt}"

# ---------- A. models table (24 + 2), with highlights
M26 = pd.read_csv(RESN / "model_axis_means.csv", index_col=0)
pcs24 = pd.read_csv(RES / "model_pcs.csv", index_col=0)
order = list(pcs24.loc[KEYS24].sort_values("PC1", ascending=False).index)
top3, bot3 = order[:3], order[-3:]
lines = []
for k in order + ["google_gemini_2_5_pro", "x_ai_grok_4"]:
    _, prov, _, acc = MODELS[k]
    m = M26.loc[k]
    if k == "google_gemini_2_5_pro":
        lines.append("\\midrule")
    nm = mlogo(k) + ("$^{\\dagger}$" if k not in pcs24.index else "")
    pc = sgn(pcs24.loc[k, "PC1"]) if k in pcs24.index else "--"
    if k in top3: pc = hl("hlLeft", pc)
    if k in bot3: pc = hl("hlRight", pc)
    pe = sgn(m.pct_econ); ps = sgn(m.pct_soc); sr = sgn(m.sap_right); sa = sgn(m.sap_auth)
    if m.pct_econ > 0: pe = hl("hlWarn", pe)
    if m.pct_soc > 0: ps = hl("hlWarn", ps)
    if m.sap_right > 0: sr = hl("hlWarn", sr)
    sa = hl("hlAuth", sa) if m.sap_auth > 0 else sa
    g = f"{m['8val_govt']:.0f}"
    if m["8val_govt"] < 50: g = hl("hlAuth", g)
    lines.append(f"{nm} & {prov} & {acc} & {pc} & {pe} & {ps} & {sr} & {sa} & {sgn(m.sap_prog)} & "
                 f"{m['8val_econ']:.0f} & {m['8val_dipl']:.0f} & {g} & {m['8val_scty']:.0f} \\\\")
with open(TAB / "tab_models.tex", "w") as f:
    f.write("\n".join(lines))
M24 = M26.loc[KEYS24]
N3 = {"SapAuthPos": int((M24.sap_auth > 0).sum()), "PctSocNeg": int((M24.pct_soc < 0).sum()),
      "EvGovtBelow": int((M24["8val_govt"] < 50).sum())}

# ---------- H. per-model news table with best / worst highlights
cols = [("auc_lr", "{:.3f}", True), ("conc_ws", "{:.3f}", True), ("macro_f1", "{:.3f}", True),
        ("rho", "{:.2f}", True), ("slope", "{:.2f}", True)]
lines = []
for k in P.sort_values("conc_ws", ascending=False).index:
    p = P.loc[k]
    cells = []
    for c, fmt, hi in cols:
        t = fmt.format(p[c])
        if p[c] == P[c].max(): t = hl("hlGood", t)
        if p[c] == P[c].min(): t = hl("hlWarn", t)
        cells.append(t)
    off = sgn(p.offset)
    if p.offset == P.offset.min(): off = hl("hlLeft", off)
    if p.offset > 0: off = hl("hlRight", off)
    lines.append(f"{mlogo(k)} & {int(p.n)} & " + " & ".join(cells) + f" & {off} & {p.mae:.2f} \\\\")
with open(TAB / "tab_news_models.tex", "w") as f:
    f.write("\n".join(lines))
N3["NPosOffset"] = int((P.offset > 0).sum())
N3["SlopeBelowHalf"] = int((P.slope < 0.5).sum())
N3["BestConcModel"] = MODELS[P.conc_ws.idxmax()][0]
N3["WorstConcModel"] = MODELS[P.conc_ws.idxmin()][0]

# ---------- F1. reference respondents with highlights
lines = []
for lab, r in RR.iterrows():
    c = [sgn(r.pct_econ), sgn(r.pct_soc), sgn(r.sap_right), sgn(r.sap_auth), sgn(r.sap_prog),
         f"{r['8val_econ']:.1f}", f"{r['8val_dipl']:.1f}", f"{r['8val_govt']:.1f}", f"{r['8val_scty']:.1f}"]
    if lab in ("Always agree", "Always strongly agree"):
        c[1] = hl("hlAuth", c[1]); c[7] = hl("hlAuth", c[7])
    if lab.startswith("Uniform"):
        c = [hl("hlGray", x) for x in c]
        lab = hl("hlGray", lab)
    lines.append(f"{lab} & " + " & ".join(c) + " \\\\")
with open(TAB / "tab_reference.tex", "w") as f:
    f.write("\n".join(lines))

# ---------- F2. consensus items, highlighting the authority/nation-leaning consensus
lines = []
for r in U.itertuples():
    lean = sgn(r.lean)
    row = f"{inst_cmd[r.inst]} & {tex_esc(r.text)} & {lean} \\\\"
    if r.lean < 0:
        row = f"\\rowcolor{{hlAuth}}{row}"
    lines.append(row)
with open(TAB / "tab_unanimous.tex", "w") as f:
    f.write("\n".join(lines))
N3["UnanItemsNeg"] = int((U.lean < 0).sum())

# ---------- E. MTMM as a matrix: rho / CI / p_FDR stacked in each cell
order_s = ["Econ|pct", "Econ|sap", "Econ|8val", "Lib|pct", "Lib|sap", "Lib|8val", "Prog|sap", "Prog|8val"]
lab_s = {"Econ": "Econ.\\ left", "Lib": "Libertarian", "Prog": "Progressive"}
icmd = {"pct": "\\instPCT{PCT}", "sap": "\\instSAP{Sapply}", "8val": "\\instEV{8V}"}
get = {}
for r in mt.itertuples():
    get[(r.a, r.b)] = r; get[(r.b, r.a)] = r
conv_best = {}
for s_ in order_s:
    vals = [get[(s_, t)].rho for t in order_s if t != s_ and t.split("|")[0] == s_.split("|")[0]]
    conv_best[s_] = max(vals)
def cell(a, b):
    r = get[(a, b)]
    body = (f"\\makecell{{\\textbf{{{sgn(r.rho)}}}\\\\[-1pt]{{\\tiny[{sgn(r.rho_lo)}, {sgn(r.rho_hi)}]}}"
            f"\\\\[-1pt]{{\\tiny $p$\\,=\\,{r.p_rho_fdr:.3f}}}}}")
    same = a.split("|")[0] == b.split("|")[0]
    if same:
        return hl("hlGood" if r.p_rho_fdr < .05 else "hlWarn", body)
    if r.rho >= min(conv_best[a], conv_best[b]):
        return hl("hlRight", body)
    return body
head = " & ".join(f"\\makecell{{{lab_s[s_.split('|')[0]]}\\\\{icmd[s_.split('|')[1]]}}}" for s_ in order_s[:-1])
lines = [f" & {head} \\\\", "\\midrule"]
for i, a in enumerate(order_s[1:], start=1):
    cells = [cell(a, b) for b in order_s[:i]] + [""] * (len(order_s) - 1 - i)
    lab = f"\\makecell[l]{{{lab_s[a.split('|')[0]]}\\\\{icmd[a.split('|')[1]]}}}"
    lines.append(f"{lab} & " + " & ".join(cells) + " \\\\[2pt]")
with open(TAB / "tab_mtmm_matrix.tex", "w") as f:
    f.write("\n".join(lines))
N3["NViolations"] = sum(1 for r in mt.itertuples() if r.a.split("|")[0] != r.b.split("|")[0]
                        and r.rho >= min(conv_best[r.a], conv_best[r.b]))
N3["NConvSig"] = int(((mt.relation == "monotrait-heteromethod") & (mt.p_rho_fdr < .05)).sum())
N3["NConv"] = int((mt.relation == "monotrait-heteromethod").sum())

# ---------- G. sensitivity table 24 vs 26
def lk2(Lx, y, x):
    return Lx[(Lx.y == y) & (Lx.x == x)].iloc[0].rho
rows_s = [("PC1 share of variance (\\%)", f"{100*R['axis_pca']['share'][0]:.0f}", f"{100*RN['axis_pca']['share'][0]:.0f}"),
          ("PC2 share of variance (\\%)", f"{100*R['axis_pca']['share'][1]:.0f}", hl("hlWarn", f"{100*RN['axis_pca']['share'][1]:.0f}")),
          ("Components above random threshold", str(R["axis_pca"]["n_retained"]), hl("hlWarn", str(RN["axis_pca"]["n_retained"]))),
          ("$\\rho$(\\ev liberty, \\sap progressive)", f2(mtr("Lib|8val", "Prog|sap").rho), f2(mt26[(mt26.a == "Lib|8val") & (mt26.b == "Prog|sap")].iloc[0].rho)),
          ("Mean same-trait $\\rho$ (convergent)", f2(R["mtmm_summary"]["mean"]["monotrait-heteromethod"]), hl("hlWarn", f2(RN["mtmm_summary"]["mean"]["monotrait-heteromethod"]))),
          ("Mean different-trait $\\rho$ (across instruments)", f2(R["mtmm_summary"]["mean"]["heterotrait-heteromethod"]), hl("hlWarn", f2(RN["mtmm_summary"]["mean"]["heterotrait-heteromethod"]))),
          ("$\\rho$(PC1, directional offset)", sgn(lk2(L, "offset", "PC1")), sgn(lk2(L26, "offset", "PC1"))),
          ("$\\rho$(PC1, within-story concordance)", sgn(lk2(L, "conc_ws", "PC1")), sgn(lk2(L26, "conc_ws", "PC1")))]
with open(TAB / "tab_sensitivity.tex", "w") as f:
    f.write("\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows_s))
with open(TAB / "numbers.tex", "a") as f:
    for k, v in N3.items():
        f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
print("appendix v2 tables", N3)

# =================================================================== Appendix: full item tables (colour-coded)
import sys as _sys
_sys.path.insert(0, str(ROOT / "analysis"))
import keys as KK
from common import load_items
qs_, econv_, socv_, e0_, s0_ = KK.pct_keys()
et_, ee_ = KK.ev_keys()
st_, se_ = KK.sap_keys()
itm = {i: load_items(i) for i in ["pct", "sap", "8val"]}
DATAP = ROOT / "data"


def item_stats(inst):
    d = itm[inst]
    mm = d.groupby(["model", "item"]).value.mean().unstack().loc[KEYS24]
    mean = mm.mean()
    cons = mm.apply(lambda c: max((c > 0).mean(), (c < 0).mean()))
    return mean, cons


def fmt_ans(v):
    return "0.0" if abs(v) < 0.05 else sgn(v, 1)


def item_rows(inst, keyfun, catfun):
    mean, cons = item_stats(inst)
    texts = pd.read_csv(DATAP / f"{inst}_item_responses.csv", encoding="utf-8-sig").question_text.tolist()
    out = []
    for i, t in enumerate(texts):
        k = f"{inst}:q{i+1}"
        c = cons[k]
        star = "$\\bigstar$" if c >= 0.999 else ""
        out.append(f"\\rowcolor{{{catfun(i)}}} {i+1} & {tex_esc(t)} & {keyfun(i)} & {fmt_ans(mean[k])} & {100*c:.0f}\\%{star} \\\\")
    return out


def pct_key(i):
    de = econv_[i, 3] - econv_[i, 0]
    ds = socv_[i, 3] - socv_[i, 0]
    if de == 0 and ds == 0:
        return "unscored"
    if de != 0:
        return "Left" if de < 0 else "Right"
    return "Lib." if ds < 0 else "Auth."


def pct_cat(i):
    n = i + 1
    return ("catD" if n < 8 else "catA" if n < 22 else "catB" if n < 40 else "catC" if n < 52 else "catE" if n < 57 else "catF")


def sap_key(i):
    r = se_.iloc[i]
    if r["right"] != 0: return "Right" if r["right"] > 0 else "Left"
    if r["auth"] != 0: return "Auth." if r["auth"] > 0 else "Lib."
    return "Prog." if r["prog"] > 0 else "Cons."


def sap_cat(i):
    r = se_.iloc[i]
    return "catA" if r["right"] != 0 else "catB" if r["auth"] != 0 else "catC"


EVLAB = {"econ": ("Equal.", "Mkt."), "dipl": ("Glob.", "Nat."), "govt": ("Lib.", "Auth."), "scty": ("Prog.", "Trad.")}


def ev_key(i):
    r = ee_.iloc[i]
    parts = [EVLAB[a][0 if r[a] > 0 else 1] for a in ["econ", "dipl", "govt", "scty"] if r[a] != 0]
    return ", ".join(parts)


def ev_cat(i):
    r = ee_.iloc[i].abs()
    return {"econ": "catA", "dipl": "catD", "govt": "catB", "scty": "catC"}[r.idxmax()]


for inst, kf, cf in [("pct", pct_key, pct_cat), ("sap", sap_key, sap_cat), ("8val", ev_key, ev_cat)]:
    with open(TAB / f"tab_items_{inst}.tex", "w") as f:
        f.write("\n".join(item_rows(inst, kf, cf)))
N4 = {"PctUnscored": sum(1 for i in range(62) if pct_key(i) == "unscored")}
with open(TAB / "numbers.tex", "a") as f:
    for k, v in N4.items():
        f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
print("item tables done")
_mean_all = pd.concat([item_stats(i)[0] for i in ["pct", "sap", "8val"]])
N5 = {"ItemsAgreeShare": f"{100*(_mean_all > 0).mean():.0f}",
      "ItemsStrongDisagree": int((_mean_all <= -1.0).sum()),
      "ItemsStrongAgree": int((_mean_all >= 1.0).sum())}
with open(TAB / "numbers.tex", "a") as f:
    for k, v in N5.items():
        f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
print(N5)

# ---------- full-width chunks of the item tables
CHUNKS = {"pct": [(0, 31), (31, 62)], "sap": [(0, 46)], "8val": [(0, 35), (35, 70)]}
for inst, kf, cf in [("pct", pct_key, pct_cat), ("sap", sap_key, sap_cat), ("8val", ev_key, ev_cat)]:
    rows_ = item_rows(inst, kf, cf)
    for j, (a_, b_) in enumerate(CHUNKS[inst], start=1):
        with open(TAB / f"tab_items_{inst}_{j}.tex", "w") as f:
            f.write("\n".join(rows_[a_:b_]))
print("item chunks done")
