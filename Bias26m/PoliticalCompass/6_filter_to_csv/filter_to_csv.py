"""
filter_to_csv.py
Converts _pc_scores.json -> pc_scores_runs.csv with columns:
model_name, run, econ_score, soc_score

Prereqs: none beyond the Python standard library.
Run:     python filter_to_csv.py
"""

import csv
import json
import logging
import re
from pathlib import Path

# ========= CONFIG =========
JSON_PATH = r"D:\Research\Political_Bias_of_LLMs\PoliticalCompass\5_model_results\_pc_scores.json"
OUTPUT_CSV = Path(__file__).resolve().parent / "pc_scores_runs.csv"

# ========= LOGGING =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("filter-to-csv")

# Pattern: "pred_[n]_MODELNAME"
RESP_RE = re.compile(r"^\s*pred_\[(\d+)\]_(.+?)\s*$", re.IGNORECASE)

def parse_responder(responder: str):
    """
    Extract run number and model name from responder.
    Expected: pred_[n]_some_model_name
    Fallback: run=None, model_name=responder (trimmed) if pattern does not match.
    """
    if not isinstance(responder, str):
        return None, str(responder)

    m = RESP_RE.match(responder)
    if m:
        run = int(m.group(1))
        model_name = m.group(2).strip()
        return run, model_name
    else:
        # Fallback handling if pattern differs
        log.warning("Responder did not match expected pattern: %r", responder)
        return None, responder.strip()

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    src = Path(JSON_PATH)
    if not src.exists():
        raise FileNotFoundError(f"JSON not found at: {src}")

    log.info("Loading JSON: %s", src)
    with open(src, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at the top level.")

    rows = []
    for i, item in enumerate(data, 1):
        responder = item.get("responder")
        econ = to_float(item.get("econ_score"))
        soc = to_float(item.get("soc_score"))

        run, model_name = parse_responder(responder)

        rows.append({
            "model_name": model_name,
            "run": run,
            "econ_score": econ,
            "soc_score": soc,
        })

    # Write CSV in the current dir of this script
    log.info("Writing CSV: %s", OUTPUT_CSV)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "run", "econ_score", "soc_score"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Done. Wrote %d rows.", len(rows))
    # Optional quick peek at the first few rows
    for preview in rows[:3]:
        log.info("Sample: %s", preview)

if __name__ == "__main__":
    main()
