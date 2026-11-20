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
URL = "https://8values.github.io/quiz.html"
CSV_PATH = r"D:\Research\Political_Bias_of_LLMs\8Values\3_combine\_merged_predictions.csv"
RESULTS_JSON_PATH = "_8values_scores.json"

KEYS = ["question_id", "question_text"]  # columns to keep as identifiers
VALID_ANS = {"sa", "a", "n", "d", "sd"}       # value attributes mapping
WAIT_BETWEEN_RESPONDERS_SEC = 0.05       # super fast: 50ms between responders
ANSWER_CLICK_PAUSE_SEC = 0.01            # super fast: 10ms after each click
PAGE_LOAD_TIMEOUT = 10                   # reduced to 10 seconds for speed
FIND_TIMEOUT = 3                         # aggressive: 3 seconds max
HEADLESS = True                          # headless for maximum speed

# Mapping from our codes to 8Values button values
ANSWER_MAPPING = {
    "sa": 1.0,      # Strongly Agree
    "a": 0.5,       # Agree  
    "n": 0.0,       # Neutral / Unsure
    "d": -0.5,      # Disagree
    "sd": -1.0      # Strongly Disagree
}

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("8values-scraper")


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
    Ultra-fast button clicking with direct JS execution as primary method.
    8Values uses specific class names for buttons and next_question() function.
    """
    # Try direct JS execution first for maximum speed
    js_code = f"next_question({answer_value})"
    try:
        driver.execute_script(js_code)
        time.sleep(ANSWER_CLICK_PAUSE_SEC)
        log.debug("JS executed for answer value: %s", answer_value)
        return
    except Exception:
        log.debug("JS execution failed, falling back to button click")
    
    # Fallback to button clicking if JS fails
    # Map answer values to button classes for 8Values
    class_mapping = {
        1.0: "stronglyAgree",
        0.5: "agree", 
        0.0: "neutral",
        -0.5: "disagree",
        -1.0: "stronglyDisagree"
    }
    
    button_class = class_mapping.get(answer_value)
    if not button_class:
        raise ValueError(f"Invalid answer value: {answer_value}")
    
    # Find button with the specific class and onclick value
    xpath = f"//button[@class='button {button_class}'][contains(@onclick, 'next_question( {answer_value})') or contains(@onclick, 'next_question({answer_value})')]"
    wait = WebDriverWait(driver, FIND_TIMEOUT)
    try:
        button = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
        driver.execute_script("arguments[0].click();", button)  # Use JS click for reliability
        time.sleep(ANSWER_CLICK_PAUSE_SEC)
        log.debug("Clicked button for answer value: %s (class: %s)", answer_value, button_class)
    except TimeoutException:
        raise NoSuchElementException(f"Could not find clickable button for answer value {answer_value} with class {button_class}")


def extract_scores_from_url(driver):
    """
    Extract the e, d, g, and s scores from the 8Values results URL.
    Expected format: https://8values.github.io/results.html?e=53.2&d=52.2&g=48.8&s=50.9
    """
    current_url = driver.current_url
    log.debug("Current URL: %s", current_url)
    
    parsed_url = urlparse(current_url)
    query_params = parse_qs(parsed_url.query)
    
    # Extract the four scores
    econ_score = None  # Economic Axis (equality score)
    dipl_score = None  # Diplomatic Axis (globe score)
    govt_score = None  # Civil Axis (liberty score)
    scty_score = None  # Societal Axis (progress score)
    
    if 'e' in query_params:
        try:
            econ_score = float(query_params['e'][0])
        except (ValueError, IndexError):
            log.warning("Could not parse 'e' (economic) score from URL")
    
    if 'd' in query_params:
        try:
            dipl_score = float(query_params['d'][0])
        except (ValueError, IndexError):
            log.warning("Could not parse 'd' (diplomatic) score from URL")
    
    if 'g' in query_params:
        try:
            govt_score = float(query_params['g'][0])
        except (ValueError, IndexError):
            log.warning("Could not parse 'g' (government) score from URL")
    
    if 's' in query_params:
        try:
            scty_score = float(query_params['s'][0])
        except (ValueError, IndexError):
            log.warning("Could not parse 's' (society) score from URL")
    
    log.info("Extracted scores | econ=%.1f, dipl=%.1f, govt=%.1f, scty=%.1f", 
             econ_score if econ_score is not None else float('nan'),
             dipl_score if dipl_score is not None else float('nan'),
             govt_score if govt_score is not None else float('nan'),
             scty_score if scty_score is not None else float('nan'))
    
    return econ_score, dipl_score, govt_score, scty_score


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
      - after finishing all answers, waits for automatic navigation to results
      - extracts the four scores from the results URL (8Values: e, d, g, s)
      - returns (econ_score, dipl_score, govt_score, scty_score)
    """
    log.info("📋 Starting evaluation for: %s", responder_col)
    open_form(driver)

    answered = 0
    total_questions = len(df)
    
    for idx, row in df.iterrows():
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
            log.warning("⚠️  Skipping %s: invalid value %r (expected one of %s)", qid, val, sorted(VALID_ANS))
            continue

        # Map to 8Values button value
        answer_value = ANSWER_MAPPING[val]

        try:
            click_answer_button(driver, answer_value)
            answered += 1
            
            # Show progress every 10 questions or for debugging
            if answered % 10 == 0 or answered == total_questions:
                log.info("📊 Progress: %d/%d questions answered", answered, total_questions)
            
            log.debug("Answered %s -> %s (%.1f)", qid, val, answer_value)
        except Exception as e:
            log.error("❌ Failed to answer %s -> %s: %s", qid, val, e)

    log.info("📝 Answered %d items for %s. Waiting for results...", answered, responder_col)
    
    try:
        # Fast check for navigation to results page with shorter polling intervals
        max_attempts = 20
        for i in range(max_attempts):
            if "results.html" in driver.current_url:
                break
            time.sleep(0.1)  # Check every 100ms for faster response
        else:
            # Fallback to WebDriverWait if fast polling fails
            wait = WebDriverWait(driver, 5)
            wait.until(lambda d: "results.html" in d.current_url)
        
        # Extract 4 scores from URL
        econ, dipl, govt, scty = extract_scores_from_url(driver)
        log.info("🎯 Scores for %s | econ=%.1f, dipl=%.1f, govt=%.1f, scty=%.1f", responder_col, 
                 econ if econ is not None else float('nan'),
                 dipl if dipl is not None else float('nan'),
                 govt if govt is not None else float('nan'),
                 scty if scty is not None else float('nan'))
        return econ, dipl, govt, scty
    except Exception as e:
        log.error("❌ Failed to get results for %s: %s", responder_col, e)
        return None, None, None, None


def main():
    print("=" * 60)
    print("🧠 8VALUES POLITICAL BIAS EVALUATION")
    print("=" * 60)
    
    # Load data
    df = load_dataframe(CSV_PATH)
    all_cols = list(df.columns)
    responder_cols = [c for c in all_cols if c not in KEYS]

    if not responder_cols:
        raise ValueError("No responder columns found (only keys present).")

    log.info("📊 Found %d responder columns to evaluate", len(responder_cols))
    log.info("📋 Questions: %d rows loaded", len(df))
    log.info("🚀 Starting evaluation with optimized settings...")
    
    # Start browser once for efficiency
    driver = make_driver(headless=HEADLESS)

    results = []
    try:
        for idx, col in enumerate(responder_cols, 1):
            if idx > 1:  # Add visual separator between responders
                print("\n" + "─" * 50 + "\n")
            
            log.info("🤖 Processing responder %d/%d: %s", idx, len(responder_cols), col)
            start_time = time.time()
            
            econ, dipl, govt, scty = collect_scores_for_responder(driver, df, col)
            
            elapsed = time.time() - start_time
            log.info("⏱️  Completed %s in %.2f seconds", col, elapsed)
            
            results.append({
                "responder": col,
                "econ_score": econ,
                "dipl_score": dipl,
                "govt_score": govt,
                "scty_score": scty,
            })
            time.sleep(WAIT_BETWEEN_RESPONDERS_SEC)
    finally:
        driver.quit()
        log.info("🔒 Browser closed.")

    # Write JSON array
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    log.info("✅ SUCCESS: Wrote %d results to %s", len(results), RESULTS_JSON_PATH)
    log.info("📁 Results saved to: %s", Path(RESULTS_JSON_PATH).resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
