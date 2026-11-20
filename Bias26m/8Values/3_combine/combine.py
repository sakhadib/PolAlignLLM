import os
import glob
from pathlib import Path
import pandas as pd
import re

# ===== Configure these =====
INPUT_DIR = r"D:\Research\Political_Bias_of_LLMs\8Values\2_8ValuesTest\predictions"
OUTPUT_CSV = "_merged_predictions.csv"
KEYS = ["question_id", "question_text"]  # columns to keep once

def natural_sort_key(text):
    """
    Generate a key for natural sorting that handles numbers within text properly.
    For example: q1, q2, q10, q11 instead of q1, q10, q11, q2
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', str(text))]

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {INPUT_DIR}")

    # Read the first file as the base
    df_merged = pd.read_csv(files[0], encoding="utf-8-sig")
    if not set(KEYS).issubset(df_merged.columns):
        raise ValueError(f"{files[0]} is missing required columns: {KEYS}")
    # Ensure key columns come first
    df_merged = df_merged[KEYS + [c for c in df_merged.columns if c not in KEYS]]
    # De-duplicate rows by keys if any
    df_merged = df_merged.drop_duplicates(subset=KEYS, keep="first")

    # Iteratively merge the rest
    for f in files[1:]:
        df = pd.read_csv(f, encoding="utf-8-sig")
        if not set(KEYS).issubset(df.columns):
            raise ValueError(f"{f} is missing required columns: {KEYS}")

        # De-duplicate within the current file on keys
        df = df.drop_duplicates(subset=KEYS, keep="first")

        # Rename any colliding non-key columns to keep uniqueness
        stem = Path(f).stem
        rename_map = {}
        for col in df.columns:
            if col in KEYS:
                continue
            if col in df_merged.columns:
                rename_map[col] = f"{stem}__{col}"
        if rename_map:
            df = df.rename(columns=rename_map)

        # Outer merge on keys to keep all questions (if any differ)
        df_merged = df_merged.merge(df, on=KEYS, how="outer", validate="one_to_one")

    # Sort by question_id using natural sorting
    df_merged['_sort_key'] = df_merged['question_id'].apply(natural_sort_key)
    df_merged = df_merged.sort_values('_sort_key')
    df_merged = df_merged.drop(columns=['_sort_key'])
    df_merged = df_merged.reset_index(drop=True)

    # Write result
    df_merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Merged {len(files)} files into: {OUTPUT_CSV}")
    print("Final shape:", df_merged.shape)

if __name__ == "__main__":
    main()
