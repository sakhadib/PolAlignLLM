"""
summarize_model_scores.py
Computes mean and standard deviation of econ/soc scores per model.

Input CSV:
  D:\\Research\\BiasStudyPaper\\6_filter_to_csv\\pc_scores_runs.csv
Expected columns:
  model_name, run, econ_score, soc_score

Output:
  pc_scores_summary_by_model.csv (saved beside this script)

Run:
  python summarize_model_scores.py
"""

from pathlib import Path
import logging
import pandas as pd

# ========= CONFIG =========
INPUT_CSV = r"D:\Research\Political_Bias_of_LLMs\PoliticalCompass\6_filter_to_csv\pc_scores_runs.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "pc_scores_summary_by_model.csv"

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("summary")

    src = Path(INPUT_CSV)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    log.info("Loading: %s", src)
    df = pd.read_csv(src, encoding="utf-8-sig")

    # Ensure numeric types; non-numeric coerced to NaN
    for col in ["econ_score", "soc_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without a model_name
    df = df.dropna(subset=["model_name"])

    # Group and aggregate (std is sample std with ddof=1 by default)
    summary = (
        df.groupby("model_name", dropna=False)
          .agg(
              econ_score_mean=("econ_score", "mean"),
              econ_score_std =("econ_score", "std"),
              soc_score_mean =("soc_score", "mean"),
              soc_score_std  =("soc_score", "std"),
          )
          .reset_index()
          .sort_values("model_name")
          .reset_index(drop=True)
    )

    # Optional: round for readability
    summary = summary.round(4)

    log.info("Writing: %s", OUTPUT_CSV)
    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Done. %d models summarized.", len(summary))

    # Quick preview
    log.info("Preview:\n%s", summary.head().to_string(index=False))

if __name__ == "__main__":
    main()
