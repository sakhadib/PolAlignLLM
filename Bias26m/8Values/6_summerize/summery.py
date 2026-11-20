"""
summery.py
Computes mean and standard deviation of all 8 political values per model.

Input CSV:
  D:\\Research\\Political_Bias_of_LLMs\\8Values\\5_tocsv\\8values_scores_runs.csv
Expected columns:
  responder, run, Economic_Equality, Economic_Markets, Diplomatic_Globe, Diplomatic_Nation, 
  Govt_Liberty, Govt_Authority, Societal_Progress, Societal_Tradition

Output:
  8values_scores_summary_by_model.csv (saved beside this script)

Run:
  python summery.py
"""

from pathlib import Path
import logging
import pandas as pd
import re

# ========= CONFIG =========
INPUT_CSV = r"D:\Research\Political_Bias_of_LLMs\8Values\5_tocsv\8values_scores_runs.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "8values_scores_summary_by_model.csv"

# Pattern: "pred_[n]_MODELNAME"
RESP_RE = re.compile(r"^\s*pred_\[(\d+)\]_(.+?)\s*$", re.IGNORECASE)

def parse_responder(responder: str):
    """
    Extract model name from responder.
    Expected: pred_[n]_some_model_name
    Fallback: model_name=responder (trimmed) if pattern does not match.
    """
    if not isinstance(responder, str):
        return str(responder)

    m = RESP_RE.match(responder)
    if m:
        model_name = m.group(2).strip()
        return model_name
    else:
        # Fallback handling if pattern differs
        return responder.strip()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("8values-summary")

    src = Path(INPUT_CSV)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    log.info("Loading: %s", src)
    df = pd.read_csv(src, encoding="utf-8-sig")

    # Extract model names from responder column
    df['model_name'] = df['responder'].apply(parse_responder)

    # Define the 8 value columns
    value_columns = [
        "Economic_Equality", "Economic_Markets", "Diplomatic_Globe", "Diplomatic_Nation",
        "Govt_Liberty", "Govt_Authority", "Societal_Progress", "Societal_Tradition"
    ]

    # Ensure numeric types; non-numeric coerced to NaN
    for col in value_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without a model_name
    df = df.dropna(subset=["model_name"])

    # Create aggregation dictionary for all 8 values
    agg_functions = {}
    for col in value_columns:
        agg_functions[col] = ['mean', 'std']

    # Group and aggregate (std is sample std with ddof=1 by default)
    summary = (
        df.groupby("model_name", dropna=False)
          .agg(agg_functions)
          .reset_index()
    )

    # Flatten column names
    summary.columns = ['model_name'] + [f"{col}_{stat}" for col in value_columns for stat in ['mean', 'std']]
    
    # Sort by model name
    summary = summary.sort_values("model_name").reset_index(drop=True)

    # Optional: round for readability (3 decimal places to match input precision)
    summary = summary.round(3)

    log.info("Writing: %s", OUTPUT_CSV)
    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Done. %d models summarized.", len(summary))

    # Quick preview
    log.info("Preview:\n%s", summary.head().to_string(index=False))

if __name__ == "__main__":
    main()
