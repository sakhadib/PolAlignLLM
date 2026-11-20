import csv

def clean_csv(input_file, output_file):
    # Read the existing CSV
    with open(input_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    
    # Select only the required columns
    cleaned_rows = [
        {
            "question_id": row.get("question_id", "").strip(),
            "question_text": row.get("question_text", "").strip()
        }
        for row in reader
    ]
    
    # Write cleaned CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question_text"])
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f"Saved cleaned file: {output_file} ({len(cleaned_rows)} rows)")

if __name__ == "__main__":
    clean_csv("questions_wide_git.csv", "questions_wide_git_clean.csv")
    clean_csv("questions_wide_site.csv", "questions_wide_site_clean.csv")
