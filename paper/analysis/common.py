"""Shared metadata, palette and loaders for the MultiDems paper analyses."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
RESULTS = ROOT / ("results" if __import__("os").environ.get("COHORT", "26") == "26" else "results24")
TAB = ROOT / "tables"
LOGO = ROOT / "logos"

# ---------------------------------------------------------------- palette
# One palette for the whole paper (figures and LaTeX use the same hex codes).
PAL = {
    "pct": "#6C4AB6",      # Political Compass  (violet)
    "sap": "#E09F3E",      # SapplyValues       (amber)
    "8val": "#2A9D8F",     # 8Values            (teal)
    "left": "#3B6FB6",     # left-leaning label (blue)
    "right": "#C44E52",    # right-leaning label (red)
    "center": "#8C8C8C",   # center label       (grey)
    "ink": "#2B2B2B",
    "muted": "#BDBDBD",
}
INST_NAME = {"pct": "Political Compass", "sap": "SapplyValues", "8val": "8Values"}

# ---------------------------------------------------------------- models
# key -> (display name, provider, logo file, access)
MODELS = {
    "openai_gpt_5": ("GPT-5", "OpenAI", "openai", "closed"),
    "openai_gpt_5_mini": ("GPT-5 mini", "OpenAI", "openai", "closed"),
    "openai_gpt_5_nano": ("GPT-5 nano", "OpenAI", "openai", "closed"),
    "openai_gpt_4_1": ("GPT-4.1", "OpenAI", "openai", "closed"),
    "openai_gpt_4_1_mini": ("GPT-4.1 mini", "OpenAI", "openai", "closed"),
    "openai_gpt_4_1_nano": ("GPT-4.1 nano", "OpenAI", "openai", "closed"),
    "openai_gpt_4o": ("GPT-4o", "OpenAI", "openai", "closed"),
    "openai_gpt_oss_120b": ("gpt-oss-120b", "OpenAI", "openai", "open"),
    "anthropic_claude_4_sonnet": ("Claude Sonnet 4", "Anthropic", "claude", "closed"),
    "google_gemini_2_5_pro": ("Gemini 2.5 Pro", "Google", "gemini", "closed"),
    "google_gemini_2_5_flash": ("Gemini 2.5 Flash", "Google", "gemini", "closed"),
    "google_gemma_3_27b_it": ("Gemma 3 27B", "Google", "gemma", "open"),
    "x_ai_grok_4": ("Grok 4", "xAI", "grok", "closed"),
    "mistralai_mistral_medium_3": ("Mistral Medium 3", "Mistral AI", "mistral", "closed"),
    "meta_meta_llama_3_8b_instruct": ("Llama 3 8B", "Meta", "meta", "open"),
    "meta_meta_llama_3_70b_instruct": ("Llama 3 70B", "Meta", "meta", "open"),
    "meta_llama_4_maverick_instruct": ("Llama 4 Maverick", "Meta", "meta", "open"),
    "meta_llama_llama_4_scout": ("Llama 4 Scout", "Meta", "meta", "open"),
    "deepseek_ai_deepseek_v3": ("DeepSeek-V3", "DeepSeek", "deepseek", "open"),
    "qwen_qwen3_235b_a22b_instruct_2507": ("Qwen3 235B", "Alibaba", "qwen", "open"),
    "moonshotai_kimi_k2_instruct": ("Kimi K2", "Moonshot AI", "kimi", "open"),
    "minimax_minimax_01": ("MiniMax-01", "MiniMax", "minimax", "open"),
    "meituan_longcat_flash_chat": ("LongCat-Flash", "Meituan", "longcat", "open"),
    "cohere_command_r_08_2024": ("Command R", "Cohere", "cohere", "open"),
    "ibm_granite_granite_3_3_8b_instruct": ("Granite 3.3 8B", "IBM", "ibm", "open"),
    "microsoft_phi_3_5_mini_128k_instruct": ("Phi-3.5 mini", "Microsoft", "microsoft", "open"),
    # added Nov 2026, same questionnaire prompt and settings, answered once (single run)
    "x_ai_grok_4_3": ("Grok 4.3", "xAI", "grok", "closed"),
    "nvidia_nemotron_3_5_lightning": ("Nemotron 3.5 Lightning", "NVIDIA", "nvidia", "open"),
    "amazon_nova_micro_v1": ("Nova Micro", "Amazon", "nova", "closed"),
    "gryphe_mythomax_l2_13b": ("MythoMax L2 13B", "Gryphe", "gryphe", "open"),
}
SINGLE_RUN = ["x_ai_grok_4_3", "nvidia_nemotron_3_5_lightning", "amazon_nova_micro_v1", "gryphe_mythomax_l2_13b"]
import os
# Grok 4 and Gemini 2.5 Pro answered the questionnaires under a different elicitation protocol
# ("your own beliefs", 10 ratings per call, T=1.0). Set COHORT=24 to exclude them.
DIFF_PROTOCOL = ["x_ai_grok_4", "google_gemini_2_5_pro"]
COHORT = os.environ.get("COHORT", "26")
KEYS = [k for k in MODELS if COHORT == "26" or k not in DIFF_PROTOCOL]
MULTI = [k for k in KEYS if k not in SINGLE_RUN]          # models with 10 runs (reliability, run-level t-SNE)
SINGLE = DATA / "single_run"

GOLD_ORDER = ["Far Left", "Left", "Lean Left", "Center", "Lean Right", "Right", "Far Right"]
GOLD_VAL = dict(zip(GOLD_ORDER, range(-3, 4)))


def _model_from_pred(col: str) -> tuple[str, int]:
    m = re.match(r"pred_\[(\d+)\]_(.+)$", col)
    return m.group(2), int(m.group(1))


def load_scores() -> pd.DataFrame:
    """Per-run axis scores, one row per (model, run). Restricted to the 26 models."""
    pct = pd.read_csv(DATA / "pct_scores_runs.csv").rename(
        columns={"econ_score": "pct_econ", "soc_score": "pct_soc"})
    sap = pd.read_csv(DATA / "sap_scores_runs.csv").rename(
        columns={"right_score": "sap_right", "auth_score": "sap_auth", "prog_score": "sap_prog"})
    ev = pd.read_csv(DATA / "8val_scores_runs.csv")
    ev["model_name"] = ev["responder"].str.replace(r"^pred_\[\d+\]_", "", regex=True)
    ev = ev.rename(columns={"Economic_Equality": "8val_econ", "Diplomatic_Globe": "8val_dipl",
                            "Govt_Liberty": "8val_govt", "Societal_Progress": "8val_scty"})
    ev = ev[["model_name", "run", "8val_econ", "8val_dipl", "8val_govt", "8val_scty"]]
    df = pct.merge(sap, on=["model_name", "run"]).merge(ev, on=["model_name", "run"])
    extra = []
    for k in SINGLE_RUN:
        j = __import__("json").load(open(SINGLE / f"{k}_scores.json"))
        extra.append({"model_name": k, "run": 1, "pct_econ": j["pct:econ_score"], "pct_soc": j["pct:soc_score"],
                      "sap_right": j["sap:right_score"], "sap_auth": j["sap:auth_score"], "sap_prog": j["sap:prog_score"],
                      "8val_econ": j["8val:Economic_Equality"], "8val_dipl": j["8val:Diplomatic_Globe"],
                      "8val_govt": j["8val:Govt_Liberty"], "8val_scty": j["8val:Societal_Progress"]})
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df[df.model_name.isin(KEYS)].reset_index(drop=True)


AXES = ["pct_econ", "pct_soc", "sap_right", "sap_auth", "sap_prog",
        "8val_econ", "8val_dipl", "8val_govt", "8val_scty"]
AXIS_LABEL = {
    "pct_econ": "PCT Economic (right +)", "pct_soc": "PCT Social (auth. +)",
    "sap_right": "Sapply Right (+)", "sap_auth": "Sapply Authority (+)", "sap_prog": "Sapply Progressive (+)",
    "8val_econ": "8V Equality (+)", "8val_dipl": "8V Globalism (+)",
    "8val_govt": "8V Liberty (+)", "8val_scty": "8V Progress (+)",
}
AXIS_INST = {a: a.split("_")[0] for a in AXES}

LIKERT = {"sd": -2, "d": -1, "n": 0, "a": 1, "sa": 2}


def read_items_wide(inst: str) -> pd.DataFrame:
    """Wide item answers (question_id, question_text, pred_[run]_model ...), including single-run models."""
    d = pd.read_csv(DATA / f"{inst}_item_responses.csv", encoding="utf-8-sig")
    for k in SINGLE_RUN:
        d = d.merge(pd.read_csv(SINGLE / f"{k}_{inst}_items.csv"), on="question_id", how="left")
    return d


def load_items(inst: str) -> pd.DataFrame:
    """Long item responses: model, run, item, value in {-2..2}."""
    d = read_items_wide(inst)
    cols = [c for c in d.columns if c.startswith("pred_")]
    long = d.melt(id_vars=["question_id"], value_vars=cols, var_name="col", value_name="resp")
    mr = long["col"].map(_model_from_pred)
    long["model"] = [x[0] for x in mr]
    long["run"] = [x[1] for x in mr]
    long["value"] = long["resp"].astype(str).str.strip().str.lower().map(LIKERT)
    long["item"] = inst + ":" + long["question_id"].astype(str)
    return long[long.model.isin(KEYS)][["model", "run", "item", "value"]]


def load_news() -> pd.DataFrame:
    d = pd.read_csv(DATA / "news_predictions.csv")
    pc = [c for c in d.columns if c.startswith("pred_")]
    d["article"] = np.arange(len(d))
    long = d.melt(id_vars=["article", "story_id", "source_bias", "ground_truth"], value_vars=pc,
                  var_name="col", value_name="pred")
    long["model"] = long["col"].str[5:]
    return long[long.model.isin(KEYS)].drop(columns="col")


# ---------------------------------------------------------------- decision models (OpenRouter Decisions API)
# OpenRouter id -> (display name, developer, logo file). One request per statement; answers are probabilities.
DMODELS = {
    "baseten/elephant-1": ("Elephant-1", "Baseten", "baseten"),
    "baseten/groundhog-1": ("Groundhog-1", "Baseten", "baseten"),
    "baseten/joey-1": ("Joey-1", "Baseten", "baseten"),
    "baseten/mammoth-1": ("Mammoth-1", "Baseten", "baseten"),
    "cloudflare/clef": ("Clef", "Cloudflare", "cloudflare"),
    "cloudflare/clef-flash": ("Clef Flash", "Cloudflare", "cloudflare"),
    "google/gemini-decision-1": ("Gemini Decision-1", "Google", "gemini"),
    "heygen/heygen-decision-1-turbo": ("HeyGen Decision-1 Turbo", "HeyGen", "heygen"),
    "jaredpalmer/kev-4b": ("Kev-4B", "Jared Palmer", "kev"),
    "liquid/d1.1": ("D1.1", "Liquid AI", "liquid"),
    "liquid/d1.2": ("D1.2", "Liquid AI", "liquid"),
    "liquid/d1-ultra": ("D1 Ultra", "Liquid AI", "liquid"),
    "liquid/d1.2-ultra": ("D1.2 Ultra", "Liquid AI", "liquid"),
    "microsoft/microsoft-decision-1.1": ("Microsoft Decision-1.1", "Microsoft", "microsoft"),
    "morph/reflex": ("Reflex", "Morph", "morph"),
    "nace-ai/drex-max": ("DREX Max", "Nace.AI", "nace"),
    "nace-ai/drex-max-finance": ("DREX Max Finance", "Nace.AI", "nace"),
    "nace-ai/drex-v1.5": ("DREX v1.5", "Nace.AI", "nace"),
    "openai/gpt-6-luna-decisions": ("GPT-6 Luna Decisions", "OpenAI", "openai"),
    "perplexity/pplx-decider-flash-v1.2": ("Decider Flash 1.2", "Perplexity", "perplexity"),
    "perplexity/pplx-decider-v1.2": ("Decider 1.2", "Perplexity", "perplexity"),
    "respan/span-01": ("Span-01", "Respan", "respan"),
    "respan/span-02": ("Span-02", "Respan", "respan"),
    "scaledown/classify-1": ("Classify-1", "ScaleDown", "scaledown"),
    "togethercomputer/tev1-4b-experimental": ("TEV1 4B (exp.)", "Together AI", "together"),
    "typesafe/jev-1.13": ("Jev-1.13", "TypeSafe", "typesafe"),
    "typesafe/jev-1.14": ("Jev-1.14", "TypeSafe", "typesafe"),
    "upstage/solar-decide": ("Solar Decide", "Upstage", "upstage"),
    "upstage/solar-decide-flash": ("Solar Decide Flash", "Upstage", "upstage"),
}
DECISION = DATA / "decision"


def load_decision():
    """Long table: model, fmt ('choice' | 'noul'), item ('pct:q1'), p_sd..p_sa (NaN if not offered); refusals listed."""
    import json, glob
    rows, refused = [], []
    for f in sorted(DECISION.glob("*.jsonl")):
        mid = (DECISION / (f.stem + ".model.txt")).read_text().strip()
        fmt = "noul" if f.stem.endswith("__noul") else "choice"
        last = {}
        for line in open(f, encoding="utf-8"):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            last[(x["test"], x["qid"])] = x
        for (t, q), x in last.items():
            if x.get("probs"):
                rows.append({"model": mid, "fmt": fmt, "item": f"{t}:{q}", **{f"p_{c}": v for c, v in x["probs"].items()}})
            else:
                refused.append({"model": mid, "fmt": fmt, "item": f"{t}:{q}"})
    return pd.DataFrame(rows), pd.DataFrame(refused)
