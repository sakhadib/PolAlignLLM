#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
separator.py

Convert question.json to questions.csv format for political bias testing.
Reads JSON with question data and creates CSV with question_id and question_text columns.
"""

import json
import csv
import os
from pathlib import Path


def convert_json_to_csv(json_file_path, csv_file_path):
    """
    Convert question.json to questions.csv format.
    
    Args:
        json_file_path (str): Path to the input JSON file
        csv_file_path (str): Path to the output CSV file
    """
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Create CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['question_id', 'question_text'])
        
        # Write questions with q1, q2, q3... format
        for item in questions_data:
            question_id = f"q{item['id'] + 1}"  # Convert 0-based to 1-based indexing
            question_text = item['question']
            writer.writerow([question_id, question_text])
    
    print(f"Successfully converted {len(questions_data)} questions from {json_file_path} to {csv_file_path}")


def main():
    """Main function to execute the conversion."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Define file paths
    json_file = script_dir / "question.json"
    csv_file = script_dir / "questions.csv"
    
    # Check if input file exists
    if not json_file.exists():
        print(f"Error: {json_file} not found!")
        return
    
    # Convert JSON to CSV
    convert_json_to_csv(json_file, csv_file)
    
    # Show preview of the created file
    print("\nPreview of created CSV file:")
    print("-" * 50)
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(f"{row[0]:<8} | {row[1]}")
            if i >= 5:  # Show first 6 rows (header + 5 questions)
                print("...")
                break


if __name__ == "__main__":
    main()
