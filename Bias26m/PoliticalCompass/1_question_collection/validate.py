import csv

GIT_FILE = "questions_wide_git_clean.csv"
SITE_FILE = "questions_wide_site_clean.csv"

def compare_questions():
    # Read Git CSV into a dict
    with open(GIT_FILE, "r", encoding="utf-8") as f:
        git_data = {row["question_id"]: row["question_text"].strip() for row in csv.DictReader(f)}

    # Read Site CSV into a dict
    with open(SITE_FILE, "r", encoding="utf-8") as f:
        site_data = {row["question_id"]: row["question_text"].strip() for row in csv.DictReader(f)}

    mismatches = []

    # Compare questions by question_id
    all_ids = sorted(set(git_data.keys()) | set(site_data.keys()))

    for qid in all_ids:
        git_text = git_data.get(qid, "").strip()
        site_text = site_data.get(qid, "").strip()

        if git_text != site_text:
            mismatches.append((qid, git_text, site_text))

    # Print result
    if mismatches:
        print(f"Found {len(mismatches)} mismatches:\n")
        for qid, git_text, site_text in mismatches:
            print(f"{qid}:\n  Git : {git_text}\n  Site: {site_text}\n")
    else:
        print("✅ All questions match perfectly!")

if __name__ == "__main__":
    compare_questions()
