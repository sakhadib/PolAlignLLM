import csv
import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.politicalcompass.org/test/en?page={}"
OUT_CSV = "questions_wide_site.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def normalize_choice_label(text: str) -> str:
    """
    Map label text to canonical keys: sd, d, a, sa.
    Falls back to empty string if unknown.
    """
    t = re.sub(r"\s+", " ", text or "").strip().lower()
    # Common variants
    mapping = {
        "strongly disagree": "sd",
        "disagree": "d",
        "agree": "a",
        "strongly agree": "sa",
    }
    return mapping.get(t, "")

def extract_questions_from_html(html: str):
    """
    Returns a list of dicts:
    {
      question_id, question_text, sd_label, d_label, a_label, sa_label
    }
    Works for both the old and new fieldset structures.
    """
    soup = BeautifulSoup(html, "html.parser")

    # New structure often uses <fieldset class="b1 pa2 mb1">,
    # old structure used <fieldset class="form-group">
    fieldsets = soup.find_all("fieldset")
    records = []

    for fs in fieldsets:
        # Find the question text: prefer legend, else skip
        legend = fs.find("legend")
        if not legend:
            continue
        question_text = legend.get_text(" ", strip=True)

        # Collect radio inputs and associated labels contained in <label> tags
        # New structure nests input + visible label text within <label><span>...</span></label>
        opt_pairs = []
        labels = fs.find_all("label")
        for lab in labels:
            inp = lab.find("input", {"type": "radio"})
            if inp is None:
                # Some structures place input outside; try sibling search as fallback
                maybe_inp = lab.find_next("input", {"type": "radio"})
                if maybe_inp and lab.find_parent("fieldset") == maybe_inp.find_parent("fieldset"):
                    inp = maybe_inp
            if inp is None:
                continue

            # Visible option text is the label's visible text (excluding the input)
            label_text = lab.get_text(" ", strip=True)
            # Remove the id-like prefix if the label text mistakenly includes it
            # (rare, but we keep it clean)
            # Typically not needed, but harmless:
            label_text = re.sub(r"^\s*[A-Za-z0-9_-]+\s*", "", label_text).strip()

            opt_pairs.append((inp, label_text))

        if not opt_pairs:
            # Fallback for very old structure: search by .form-check groups
            for opt in fs.select(".form-check, .form-check-inline"):
                inp = opt.find("input", {"type": "radio"})
                lab = opt.find("label")
                if inp and lab:
                    opt_pairs.append((inp, lab.get_text(" ", strip=True)))

        if not opt_pairs:
            continue

        # Derive question_id from the first input's name if available
        qid = None
        if opt_pairs[0][0].has_attr("name"):
            qid = opt_pairs[0][0]["name"].strip()
        else:
            # Try any input in the fieldset
            any_inp = fs.find("input", {"type": "radio"})
            if any_inp and any_inp.has_attr("name"):
                qid = any_inp["name"].strip()
            else:
                # Fallback to a hash of question text
                qid = "q_" + str(abs(hash(question_text)) % 10**8)

        # Build mapping to canonical columns.
        # Primary strategy: use visible label text (robust across value changes).
        by_key = {"sd": "", "d": "", "a": "", "sa": ""}

        for inp, lbl in opt_pairs:
            key = normalize_choice_label(lbl)
            if key:
                by_key[key] = lbl

        # If any are still blank, fallback to order-based assumption (0..3 => sd,d,a,sa)
        if not all(by_key.values()):
            ordered_lbls = [lbl for _, lbl in opt_pairs]
            order_map = ["sd", "d", "a", "sa"]
            for i, lbl in enumerate(ordered_lbls[:4]):
                if not by_key[order_map[i]]:
                    by_key[order_map[i]] = lbl

        records.append({
            "question_id": qid,
            "question_text": question_text,
            "sd_label": by_key["sd"],
            "d_label": by_key["d"],
            "a_label": by_key["a"],
            "sa_label": by_key["sa"],
        })

    return records

def main():
    all_records = []
    for k in range(1, 7):  # pages 1..6 inclusive
        url = BASE_URL.format(k)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        page_records = extract_questions_from_html(resp.text)
        all_records.extend(page_records)
        print(f"Parsed page {k}: {len(page_records)} questions")

    # Deduplicate by (question_id, question_text) just in case
    seen = set()
    unique_records = []
    for r in all_records:
        key = (r["question_id"], r["question_text"])
        if key not in seen:
            seen.add(key)
            unique_records.append(r)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "question_text", "sd_label", "d_label", "a_label", "sa_label"]
        )
        writer.writeheader()
        writer.writerows(unique_records)

    print(f"Saved {len(unique_records)} questions to {OUT_CSV}")

if __name__ == "__main__":
    main()
