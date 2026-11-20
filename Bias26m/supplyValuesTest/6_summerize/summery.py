"""
summery.py
Computes mean and standard deviation of right/auth/prog scores per model.

Input CSV:
  D:\\Research\\Political_Bias_of_LLMs\\supplyValuesTest\\5_transform_to_csv\\sapply_scores_runs.csv
Expected columns:
  model_name, run, right_score, auth_score, prog_score

Output:
  sapply_scores_summary_by_model.csv (saved beside this script)

Run:
  python summery.py
"""

from pathlib import Path
import logging
import pandas as pd

# ========= CONFIG =========
INPUT_CSV = r"D:\Research\Political_Bias_of_LLMs\supplyValuesTest\5_transform_to_csv\sapply_scores_runs.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "sapply_scores_summary_by_model.csv"

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
    for col in ["right_score", "auth_score", "prog_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without a model_name
    df = df.dropna(subset=["model_name"])

    # Group and aggregate (std is sample std with ddof=1 by default)
    summary = (
        df.groupby("model_name", dropna=False)
          .agg(
              right_score_mean=("right_score", "mean"),
              right_score_std =("right_score", "std"),
              auth_score_mean =("auth_score", "mean"),
              auth_score_std  =("auth_score", "std"),
              prog_score_mean =("prog_score", "mean"),
              prog_score_std  =("prog_score", "std"),
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
