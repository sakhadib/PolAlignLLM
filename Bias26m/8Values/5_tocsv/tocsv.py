"""
tocsv.py
Converts _8values_scores.json -> 8values_scores_runs.csv with columns:
responder, run, Economic_Equality, Economic_Markets, Diplomatic_Globe, Diplomatic_Nation, 
Govt_Liberty, Govt_Authority, Societal_Progress, Societal_Tradition

Prereqs: none beyond the Python standard library.
Run:     python tocsv.py
"""

import csv
import json
import logging
import re
from pathlib import Path

# ========= CONFIG =========
JSON_PATH = r"D:\Research\Political_Bias_of_LLMs\8Values\4_result\_8values_scores.json"
OUTPUT_CSV = Path(__file__).resolve().parent / "8values_scores_runs.csv"

# ========= LOGGING =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("8values-to-csv")

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
        return round(float(x), 3)
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
        econ_score = to_float(item.get("econ_score"))
        dipl_score = to_float(item.get("dipl_score"))
        govt_score = to_float(item.get("govt_score"))
        scty_score = to_float(item.get("scty_score"))

        run, model_name = parse_responder(responder)

        # Calculate the complementary values (100 - original)
        economic_equality = econ_score
        economic_markets = round(100 - econ_score, 3) if econ_score is not None else None
        diplomatic_globe = dipl_score
        diplomatic_nation = round(100 - dipl_score, 3) if dipl_score is not None else None
        govt_liberty = govt_score
        govt_authority = round(100 - govt_score, 3) if govt_score is not None else None
        societal_progress = scty_score
        societal_tradition = round(100 - scty_score, 3) if scty_score is not None else None

        rows.append({
            "responder": responder,
            "run": run,
            "Economic_Equality": economic_equality,
            "Economic_Markets": economic_markets,
            "Diplomatic_Globe": diplomatic_globe,
            "Diplomatic_Nation": diplomatic_nation,
            "Govt_Liberty": govt_liberty,
            "Govt_Authority": govt_authority,
            "Societal_Progress": societal_progress,
            "Societal_Tradition": societal_tradition,
        })

    # Write CSV in the current dir of this script
    log.info("Writing CSV: %s", OUTPUT_CSV)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "responder", "run", "Economic_Equality", "Economic_Markets", 
            "Diplomatic_Globe", "Diplomatic_Nation", "Govt_Liberty", "Govt_Authority", 
            "Societal_Progress", "Societal_Tradition"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Done. Wrote %d rows.", len(rows))
    # Optional quick peek at the first few rows
    for preview in rows[:3]:
        log.info("Sample: %s", preview)

if __name__ == "__main__":
    main()
