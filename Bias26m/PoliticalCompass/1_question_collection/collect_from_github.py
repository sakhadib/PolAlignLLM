import csv
import sys
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

def scrape_questions_wide(url: str, out_csv: str = "questions_wide_git.csv") -> None:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    fieldsets = soup.find_all("fieldset", class_="form-group")

    records = []
    for fs in fieldsets:
        legend = fs.find("legend")
        if not legend:
            continue
        question_text = legend.get_text(strip=True)

        first_input = fs.find("input", {"type": "radio"})
        qid = first_input["name"].strip() if (first_input and first_input.has_attr("name")) else f"q_{len(records)}"

        # Collect options into a dict keyed by value (sd/d/a/sa)
        options = defaultdict(str)
        for opt in fs.select(".form-check.form-check-inline"):
            inp = opt.find("input", {"type": "radio"})
            lab = opt.find("label")
            if not (inp and lab):
                continue
            val = inp.get("value", "").strip().lower()
            options[val] = lab.get_text(strip=True)

        records.append({
            "question_id": qid,
            "question_text": question_text,
            "sd_label": options.get("sd", ""),
            "d_label": options.get("d", ""),
            "a_label": options.get("a", ""),
            "sa_label": options.get("sa", "")
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question_text", "sd_label", "d_label", "a_label", "sa_label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} questions to {out_csv}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://politicalcompass.github.io/"
    scrape_questions_wide(url)
