#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check.py

Check for missing cell values in the merged predictions CSV file.
Reports any missing values found in the console.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_missing_values(csv_file="_merged_predictions.csv"):
    """
    Check for missing values in the CSV file and report them.
    
    Args:
        csv_file (str): Path to the CSV file to check
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"❌ ERROR: File not found: {csv_file}")
        return
    
    print(f"🔍 Checking for missing values in: {csv_file}")
    print("=" * 60)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"📊 File loaded successfully. Shape: {df.shape}")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        print()
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        return
    
    # Check for missing values
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    
    if total_missing == 0:
        print("✅ NO MISSING VALUES FOUND!")
        print("   All cells contain data.")
        return
    
    print(f"⚠️  MISSING VALUES DETECTED: {total_missing} total missing cells")
    print()
    
    # Report missing values by column
    print("📋 Missing values by column:")
    print("-" * 40)
    for col, missing_count in missing_summary.items():
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            print(f"   {col}: {missing_count} missing ({percentage:.1f}%)")
    
    print()
    
    # Find rows with any missing values
    rows_with_missing = df[df.isnull().any(axis=1)]
    
    if len(rows_with_missing) > 0:
        print(f"🔍 Rows with missing values: {len(rows_with_missing)} out of {len(df)}")
        print("-" * 50)
        
        # Show details for each row with missing values
        for idx, row in rows_with_missing.iterrows():
            question_id = row.get('question_id', f'Row {idx}')
            missing_cols = [col for col in df.columns if pd.isnull(row[col])]
            print(f"   {question_id}: Missing in {len(missing_cols)} columns")
            if len(missing_cols) <= 10:  # Show column names if not too many
                print(f"      Missing columns: {', '.join(missing_cols)}")
            else:
                print(f"      Missing columns: {', '.join(missing_cols[:10])}... (+{len(missing_cols)-10} more)")
    
    print()
    
    # Summary statistics
    total_cells = df.shape[0] * df.shape[1]
    filled_cells = total_cells - total_missing
    completeness = (filled_cells / total_cells) * 100
    
    print("📈 SUMMARY:")
    print(f"   Total cells: {total_cells:,}")
    print(f"   Filled cells: {filled_cells:,}")
    print(f"   Missing cells: {total_missing:,}")
    print(f"   Data completeness: {completeness:.2f}%")


def main():
    """Main function to run the missing value check."""
    check_missing_values()


if __name__ == "__main__":
    main()
