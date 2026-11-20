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
    lines.append(f"{lab_} & {nm} & {rel[a]['eta2']:.2f} & {rel[a]['icc1']:.2f} & {p['eta_model']:.2f} & {p['eta_prompt']:.3f} & {agr} \\\\")
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
            txt = f"\\textbf{{{txt}}}"
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
