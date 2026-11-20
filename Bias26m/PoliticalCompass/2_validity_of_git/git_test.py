import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, ElementClickInterceptedException

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService

URL = "https://politicalcompass.github.io/"
CSV_PATH = r"D:\Research\BiasStudyPaper\1_question_collection\data\questions_wide_git_clean.csv"

TOTAL_RUNS = 500
GAP_SECONDS = 0.2

OUT_JSON = "pc_runs_summary.json"     # final array (written once at the end)
OUT_JSONL = "pc_runs_summary.jsonl"   # append per run (atomic + crash-safe)
LOG_FILE = "pc_runs_log.txt"

def log(msg: str):
    ts = datetime.utcnow().isoformat() + "Z"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")
    print(msg)

def read_question_order(csv_path):
    order = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            order.append(row["question_id"].strip())
    if len(order) != 62:
        raise ValueError(f"CSV question count != 62 (found {len(order)}).")
    # sanity: enforce q1..q62 naming
    for i, qid in enumerate(order, start=1):
        if qid.lower() != f"q{i}":
            # not fatal, but warn loudly—clicks still go by serial index
            log(f"WARNING: CSV order label mismatch at index {i}: got '{qid}'")
    return order

def setup_driver():
    chrome_opts = Options()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--window-size=1200,2200")
    chrome_opts.page_load_strategy = "eager"
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_opts)
    driver.set_page_load_timeout(45)
    return driver

def ensure_present(wait):
    # page readiness for first question and score spans
    wait.until(EC.presence_of_element_located((By.ID, "q1a1")))
    wait.until(EC.presence_of_element_located((By.ID, "displayEcon")))
    wait.until(EC.presence_of_element_located((By.ID, "displaySoc")))

def click_and_verify(driver, wait, q_index_one_based, choice_idx_0to3):
    """
    Click qN option (0..3) and verify it is the checked input for name='qN'.
    Retries up to 3 times if necessary.
    """
    name = f"q{q_index_one_based}"
    input_id = f"{name}a{choice_idx_0to3 + 1}"
    for attempt in range(1, 4):
        el = wait.until(EC.presence_of_element_located((By.ID, input_id)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
        try:
            wait.until(EC.element_to_be_clickable((By.ID, input_id)))
            el.click()
        except ElementClickInterceptedException:
            time.sleep(0.1)
            el.click()

        # verify checked
        checked = driver.execute_script(
            "const els = document.querySelectorAll('input[name=\"%s\"]:checked');"
            "return els.length ? els[0].id : '';" % name
        )
        if checked == input_id:
            return  # success
        time.sleep(0.1)

    raise RuntimeError(f"Failed to verify selection for {input_id} after 3 attempts")

def get_scores_stable(driver, timeout_s=5, settle_ms=150):
    """
    Poll #displayEcon and #displaySoc until both parse as floats and remain stable
    for 'settle_ms' milliseconds. Timeout raises.
    """
    end = time.time() + timeout_s
    last = (None, None)
    last_t = 0

    def read_once():
        e = driver.execute_script("return document.getElementById('displayEcon')?.innerText?.trim() || '';")
        s = driver.execute_script("return document.getElementById('displaySoc')?.innerText?.trim() || '';")
        try:
            ef = float(e)
        except Exception:
            ef = None
        try:
            sf = float(s)
        except Exception:
            sf = None
        return ef, sf

    while time.time() < end:
        vals = read_once()
        if vals[0] is not None and vals[1] is not None:
            if vals == last:
                # stable long enough?
                if (time.time() - last_t) * 1000 >= settle_ms:
                    return vals
            else:
                last = vals
                last_t = time.time()
        time.sleep(0.05)
    raise TimeoutException("Timed out waiting for stable econ/soc scores")

def verify_all_checked(driver, expected_count=62):
    """
    Ensure we have exactly one checked input per question (i.e., 62 checked total).
    """
    checked_count = driver.execute_script(
        "return document.querySelectorAll('input[type=\"radio\"]:checked').length;"
    )
    if int(checked_count) != expected_count:
        raise AssertionError(f"Checked radios = {checked_count}, expected {expected_count}")

def main():
    random.seed()  # non-deterministic; set a fixed seed for reproducible runs if needed
    q_order = read_question_order(CSV_PATH)

    results = []
    Path(OUT_JSONL).unlink(missing_ok=True)
    Path(LOG_FILE).unlink(missing_ok=True)

    driver = setup_driver()
    wait = WebDriverWait(driver, 20)

    try:
        for run_idx in range(1, TOTAL_RUNS + 1):
            run_start_ts = datetime.utcnow().isoformat() + "Z"
            answer_array = [random.randint(0, 3) for _ in range(62)]

            try:
                driver.get(URL)
                ensure_present(wait)

                # click in CSV order (assumed q1..q62 serial mapping)
                for i, _qid in enumerate(q_order, start=1):
                    click_and_verify(driver, wait, i, answer_array[i - 1])

                # sanity: 62 selected
                verify_all_checked(driver, expected_count=62)

                # read stable scores
                econ, soc = get_scores_stable(driver, timeout_s=6, settle_ms=150)

                # record success
                record = {
                    "run_index": run_idx,
                    "timestamp": run_start_ts,
                    "answer_array": answer_array,     # <— your requirement #2
                    "git_economic": econ,             # <— your requirement #1
                    "git_social": soc,                # <— your requirement #1
                    "status": "ok",
                }
                results.append(record)
                with open(OUT_JSONL, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                log(f"[{run_idx}/{TOTAL_RUNS}] OK econ={econ}, soc={soc}")

            except Exception as e:
                # capture a failing run safely without corrupting prior data
                err = {
                    "run_index": run_idx,
                    "timestamp": run_start_ts,
                    "answer_array": answer_array,  # store what we intended to click
                    "status": "error",
                    "error": repr(e),
                }
                with open(OUT_JSONL, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(err, ensure_ascii=False) + "\n")
                log(f"[{run_idx}/{TOTAL_RUNS}] ERROR: {e!r}")

            time.sleep(GAP_SECONDS)  # 1s gap before next reload

        # write consolidated JSON at the end (ok + error records)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        log(f"Done. Wrote {len(results)} successful runs to {OUT_JSON}; all runs appended to {OUT_JSONL}.")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
