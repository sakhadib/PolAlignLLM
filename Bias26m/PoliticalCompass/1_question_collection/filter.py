import csv

INPUT_CSV = "questions_wide_site.csv"
OUTPUT_CSV = "questions_wide_site_updated.csv"

def update_question_ids():
    # Read the existing CSV
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    
    # Update the question_id column sequentially
    for idx, row in enumerate(reader, start=1):
        row["question_id"] = f"q{idx}"
    
    # Write updated data to a new CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reader[0].keys())
        writer.writeheader()
        writer.writerows(reader)

    print(f"Updated question IDs saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    update_question_ids()
