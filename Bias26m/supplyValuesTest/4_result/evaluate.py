"""
Prereqs (one-time):
    pip install selenium webdriver-manager pandas

Run:
    python evaluate.py

Notes:
- Adjust CSV_PATH if needed.
- Keeps a 0.3s delay between responders (columns).
- Adds detailed logs for each step and resilient clicking.
"""

import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    ElementClickInterceptedException,
    TimeoutException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ===================== CONFIG =====================
URL = "https://sapplyvalues.github.io/quiz.html?shuffle=false"
CSV_PATH = r"D:\Research\Political_Bias_of_LLMs\supplyValuesTest\3_combine\_merged_predictions.csv"
RESULTS_JSON_PATH = "_sapply_scores.json"

KEYS = ["question_id", "question_text"]  # columns to keep as identifiers
VALID_ANS = {"sa", "a", "n", "d", "sd"}       # value attributes mapping
WAIT_BETWEEN_RESPONDERS_SEC = 0.05       # super fast: 50ms between responders
ANSWER_CLICK_PAUSE_SEC = 0.01            # super fast: 10ms after each click
PAGE_LOAD_TIMEOUT = 10                   # reduced to 10 seconds
FIND_TIMEOUT = 3                         # aggressive: 3 seconds max
HEADLESS = True                          # headless for maximum speed

# Mapping from our codes to SapplyValues button values
ANSWER_MAPPING = {
    "sa": 1.0,      # Strongly Agree
    "a": 0.5,       # Agree  
    "n": 0,         # Neutral / Unsure
    "d": -0.5,      # Disagree
    "sd": -1.0      # Strongly Disagree
}

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sapply-scraper")


# ===================== SELENIUM SETUP =====================
def make_driver(headless=HEADLESS):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    
    # SUPER FAST optimizations
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-plugins")
    opts.add_argument("--disable-images")  # Skip images completely
    opts.add_argument("--disable-javascript-harmony-shipping")
    opts.add_argument("--disable-background-timer-throttling")
    opts.add_argument("--disable-renderer-backgrounding")
    opts.add_argument("--disable-backgrounding-occluded-windows")
    opts.add_argument("--disable-client-side-phishing-detection")
    opts.add_argument("--disable-sync")
    opts.add_argument("--disable-translate")
    opts.add_argument("--disable-ipc-flooding-protection")
    opts.add_argument("--memory-pressure-off")
    opts.add_argument("--max_old_space_size=4096")
    
    # Performance preferences
    prefs = {
        "profile.default_content_setting_values": {
            "images": 2,  # Block images
            "plugins": 2,  # Block plugins
            "popups": 2,  # Block popups
            "geolocation": 2,  # Block location sharing
            "notifications": 2,  # Block notifications
            "media_stream": 2,  # Block media stream
        }
    }
    opts.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.implicitly_wait(1)  # Reduce implicit wait
    return driver


def open_form(driver):
    log.info("Loading page: %s", URL)
    driver.get(URL)
    wait = WebDriverWait(driver, FIND_TIMEOUT)
    # Ensure page elements exist
    wait.until(EC.presence_of_element_located((By.ID, "question-text")))
    wait.until(EC.presence_of_element_located((By.ID, "question-number")))
    log.info("Page loaded and essential elements present.")


# ===================== UTILITIES =====================
_qnum_re = re.compile(r"^q(\d+)$")

def numeric_qid(qid: str) -> int:
    """
    Turn 'q1' -> 1 for stable sorting. If pattern fails, return a large number to push it last.
    """
    m = _qnum_re.match(qid.strip().lower())
    return int(m.group(1)) if m else 10_000


def click_answer_button(driver, answer_value: float):
    """
    Ultra-fast button clicking with direct JS execution.
    """
    # Use direct JS execution for maximum speed
    js_code = f"next_question({answer_value})"
    try:
        driver.execute_script(js_code)
        time.sleep(ANSWER_CLICK_PAUSE_SEC)
        log.debug("JS executed for answer value: %s", answer_value)
    except Exception:
        # Fallback to button click if JS fails
        xpath = f"//button[@class='button'][contains(@onclick, 'next_question({answer_value})')]"
        wait = WebDriverWait(driver, FIND_TIMEOUT)
        button = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        driver.execute_script("arguments[0].click();", button)
        time.sleep(ANSWER_CLICK_PAUSE_SEC)


def click_serious_button(driver):
    """
    Ultra-fast serious button click with direct JS.
    """
    try:
        # Try direct JS first for speed
        driver.execute_script("document.getElementById('serious').click();")
        log.info("JS clicked the 'serious' button")
        time.sleep(0.5)  # Minimal wait time
    except Exception:
        # Fallback to selenium click
        wait = WebDriverWait(driver, FIND_TIMEOUT)
        serious_button = wait.until(EC.element_to_be_clickable((By.ID, "serious")))
        driver.execute_script("arguments[0].click();", serious_button)
        log.info("Selenium clicked the 'serious' button")
        time.sleep(0.5)


def extract_scores_from_url(driver):
    """
    Extract the right, auth, and prog scores from the results URL.
    Expected format: https://sapplyvalues.github.io/results.html?right=1.67&auth=0.33&prog=0.31
    """
    current_url = driver.current_url
    log.debug("Current URL: %s", current_url)
    
    parsed_url = urlparse(current_url)
    query_params = parse_qs(parsed_url.query)
    
    # Extract the three scores
    right_score = None
    auth_score = None
    prog_score = None
    
    if 'right' in query_params:
        try:
            right_score = float(query_params['right'][0])
        except (ValueError, IndexError):
            log.warning("Could not parse 'right' score from URL")
    
    if 'auth' in query_params:
        try:
            auth_score = float(query_params['auth'][0])
        except (ValueError, IndexError):
            log.warning("Could not parse 'auth' score from URL")
    
    if 'prog' in query_params:
        try:
            prog_score = float(query_params['prog'][0])
        except (ValueError, IndexError):
            log.warning("Could not parse 'prog' score from URL")
    
    log.info("Extracted scores | right=%.4f, auth=%.4f, prog=%.4f", 
             right_score if right_score is not None else float('nan'),
             auth_score if auth_score is not None else float('nan'),
             prog_score if prog_score is not None else float('nan'))
    
    return right_score, auth_score, prog_score


# ===================== CORE WORKFLOW =====================
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Ensure rows are in q1..q46 order
    df = df.copy()
    df["__qnum__"] = df["question_id"].apply(numeric_qid)
    df = df.sort_values("__qnum__", kind="mergesort").drop(columns="__qnum__").reset_index(drop=True)

    # Basic validation
    if len(df) < 46:
        log.warning("CSV has %d rows; expected 46. Proceeding anyway.", len(df))
    elif len(df) > 46:
        log.info("CSV has %d rows (>=46). Will still answer per available questions.", len(df))

    return df


def collect_scores_for_responder(driver, df: pd.DataFrame, responder_col: str):
    """
    For a single responder column:
      - reloads the page (fresh form)
      - fills all questions using the values in responder_col
      - after finishing all answers, clicks the serious button
      - extracts the three scores from the results URL
      - returns (right_score, auth_score, prog_score)
    """
    log.info("=== Responder: %s ===", responder_col)
    open_form(driver)

    answered = 0
    for _, row in df.iterrows():
        qid = str(row["question_id"]).strip().lower()  # e.g., 'q3'
        # Treat blanks/NaN as 'n' (Neutral) by default
        raw_val = row[responder_col]
        if pd.isna(raw_val) or str(raw_val).strip() == "":
            val = "n"
            log.debug("Blank value for %s; defaulting to 'n' (neutral)", qid)
        else:
            val = str(raw_val).strip().lower()

        # normalize values; allow synonyms if any creep in
        if val in {"strongly agree", "str_agree", "sagree"}:
            val = "sa"
        elif val in {"agree"}:
            val = "a"
        elif val in {"neutral", "unsure", "neither"}:
            val = "n"
        elif val in {"disagree"}:
            val = "d"
        elif val in {"strongly disagree", "str_disagree", "sdisagree"}:
            val = "sd"

        if val not in VALID_ANS:
            log.warning("Skipping %s: invalid value %r (expected one of %s)", qid, val, sorted(VALID_ANS))
            continue

        # Map to SapplyValues button value
        answer_value = ANSWER_MAPPING[val]

        try:
            click_answer_button(driver, answer_value)
            answered += 1
            log.debug("Answered %s -> %s (%.1f)", qid, val, answer_value)
        except Exception as e:
            log.error("Failed to answer %s -> %s: %s", qid, val, e)

    log.info("Answered %d items for %s. Now clicking serious button.", answered, responder_col)
    
    try:
        click_serious_button(driver)
        # Minimal wait for navigation - check URL change in loop for speed
        max_attempts = 10
        for i in range(max_attempts):
            if "results.html" in driver.current_url:
                break
            time.sleep(0.1)  # Check every 100ms
        
        # Extract scores from URL
        right, auth, prog = extract_scores_from_url(driver)
        log.info("Scores for %s | right=%.4f, auth=%.4f, prog=%.4f", responder_col, 
                 right if right is not None else float('nan'),
                 auth if auth is not None else float('nan'),
                 prog if prog is not None else float('nan'))
        return right, auth, prog
    except Exception as e:
        log.error("Failed to get results for %s: %s", responder_col, e)
        return None, None, None


def main():
    # Load data
    df = load_dataframe(CSV_PATH)
    all_cols = list(df.columns)
    responder_cols = [c for c in all_cols if c not in KEYS]

    if not responder_cols:
        raise ValueError("No responder columns found (only keys present).")

    log.info("Found %d responder columns.", len(responder_cols))

    # Start browser once for efficiency
    driver = make_driver(headless=HEADLESS)

    results = []
    try:
        for idx, col in enumerate(responder_cols, 1):
            if idx > 1:  # Add visual separator between responders
                print("\n" + "="*50 + "\n")
            
            log.info("Processing responder %d/%d: %s", idx, len(responder_cols), col)
            right, auth, prog = collect_scores_for_responder(driver, df, col)
            results.append({
                "responder": col,
                "right_score": right,
                "auth_score": auth,
                "prog_score": prog,
            })
            time.sleep(WAIT_BETWEEN_RESPONDERS_SEC)
    finally:
        driver.quit()
        log.info("Browser closed.")

    # Write JSON array
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Wrote %d results to %s", len(results), RESULTS_JSON_PATH)


if __name__ == "__main__":
    main()
