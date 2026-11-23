"""Run the PCT, 8Values and SapplyValues questionnaires on one model via OpenRouter.

    python3 finalv/run.py --model finalv/models/grok_4.py
    python3 finalv/run.py --model nemotron --limit 3        # quick pilot (one run is the default)
    python3 finalv/run.py --model x-ai/grok-4 --max-tokens 4000   # raw OpenRouter id also works

Needs finalv/.env with OR_KEY=sk-or-...  Every answer is appended to output/<model>/answers.jsonl the moment
it arrives, so Ctrl-C is safe: rerun the same command and it continues where it stopped.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path

import httpx
from dotenv import dotenv_values
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import PROMPTS  # noqa: E402
from scoring import SCORERS  # noqa: E402

TESTS = ["pct", "8val", "sap"]
NAMES = {"pct": "Political Compass", "8val": "8Values", "sap": "SapplyValues"}
CODE = re.compile(r'"?rating"?\s*:\s*"?\s*(sd|sa|d|n|a)\b', re.I)
BARE = re.compile(r"\b(sd|sa|d|n|a)\b", re.I)


def parse(text, codes):
    """The answer code, or None. Takes the last {"rating": ...}; else a lone code word at the end."""
    m = CODE.findall(text or "") or BARE.findall((text or "").strip()[-40:])
    c = m[-1].lower() if m else None
    return c if c in codes else None


def load_model(arg):
    for cand in (Path(arg), HERE / "models" / f"{arg}.py", HERE / "models" / arg):
        if cand.is_file():
            ns = {}
            exec(cand.read_text(), ns)
            return dict(ns["MODEL"])
    return {"id": arg}


def slug(model_id):
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").lower()


def append(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_csv(path, header, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Run:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.out = HERE / "output" / (slug(cfg["id"]) + ("__mock" if args.mock else ""))
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / "answers.jsonl"
        self.items = {}
        for t in TESTS:
            with open(HERE / "data" / f"questions_{t}.csv", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.items[t] = rows[: args.limit] if args.limit else rows
        self.done = {}                                   # (test, run, qid) -> code, or None if unparsable
        self.cost = 0.0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                             # half-written last line after a crash
                self.done[(r["test"], r["run"], r["qid"])] = r["code"]
                self.cost += r.get("cost") or 0
        self.prev_cost = self.cost
        self.recent = deque(maxlen=10)
        self.errors = Counter()
        self.last_error = ""
        self.t0 = time.time()
        self.session_n = 0

    def todo(self):
        return [(t, r, q) for t in TESTS for r in range(1, self.args.runs + 1) for q in self.items[t]
                if self.done.get((t, r, q["question_id"])) is None]       # missing or NA -> ask

    def count(self, t):
        vals = [self.done.get((t, r, q["question_id"]), "-")
                for r in range(1, self.args.runs + 1) for q in self.items[t]]
        return sum(v not in ("-", None) for v in vals), sum(v is None for v in vals), len(vals)

    async def ask(self, http, t, r, q):
        p = PROMPTS[t]
        body = {"model": self.cfg["id"], "temperature": self.cfg.get("temperature", 0.6),
                "max_tokens": self.cfg.get("max_tokens", 30), "usage": {"include": True},
                "messages": [{"role": "system", "content": p["system"]},
                             {"role": "user", "content": p["user"].format(statement=q["question_text"])}]}
        body.update(self.cfg.get("extra", {}))
        code, text, cost = None, "", 0.0
        for attempt in range(1, 6):                     # network errors and unparsable replies are re-asked
            if self.args.mock:
                await asyncio.sleep(random.uniform(0.02, 0.1))
                text, c = random.choice(['{"rating": "a"}', '{"rating": "sd"}', '{"rating": "n"}', "sorry"]), 2e-5
            else:
                try:
                    res = await http.post("chat/completions", json=body)
                    if res.status_code in (401, 402, 403):
                        raise SystemExit(f"OpenRouter refused the request ({res.status_code}): {res.text[:300]}")
                    res.raise_for_status()
                    j = res.json()
                    text = j["choices"][0]["message"].get("content") or ""
                    c = (j.get("usage") or {}).get("cost") or 0.0
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                    self.errors[t] += 1
                    self.last_error = f"{type(e).__name__}: {str(e)[:150]}"
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
            cost += c
            code = parse(text, p["codes"])
            if code:
                break
        if not text and not code and cost == 0:
            return                                       # never reached the model: not saved, asked next run
        rec = {"test": t, "run": r, "qid": q["question_id"], "code": code, "raw": text[-300:],
               "cost": round(cost, 8), "attempts": attempt, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        append(self.path, rec)                          # on disk before anything else happens
        self.done[(t, r, q["question_id"])] = code
        self.cost += cost
        self.session_n += 1
        self.recent.appendleft((rec["ts"][11:], t, q["question_id"], r, code or "NA", cost))

    def export(self):
        """Paper-format CSVs: <test>_predictions.csv and <test>_scores_runs.csv (complete runs only)."""
        R, s = self.args.runs, slug(self.cfg["id"])
        for t in TESTS:
            cols = [f"pred_[{r}]_{s}" for r in range(1, R + 1)]
            rows = [[q["question_id"], q["question_text"]] +
                    [self.done.get((t, r, q["question_id"])) or "" for r in range(1, R + 1)] for q in self.items[t]]
            write_csv(self.out / f"{t}_predictions.csv", ["question_id", "question_text"] + cols, rows)
            if self.args.limit:
                continue                                 # partial questionnaire: no scores
            scores = []
            for r in range(1, R + 1):
                ans = [self.done.get((t, r, q["question_id"])) for q in self.items[t]]
                if all(ans):
                    sc = SCORERS[t](ans)
                    scores.append([f"pred_[{r}]_{s}" if t == "8val" else s, r] + list(sc.values()))
            if scores:
                head = ["responder" if t == "8val" else "model_name", "run"] + list(sc)
                write_csv(self.out / f"{t}_scores_runs.csv", head, scores)

    def view(self):
        el = time.time() - self.t0
        counts = {t: self.count(t) for t in TESTS}
        done, total = sum(c[0] for c in counts.values()), sum(c[2] for c in counts.values())
        spent = self.cost - self.prev_cost
        per = spent / self.session_n if self.session_n else 0
        rate = self.session_n / el if el > 1 else 0
        left = total - done
        info = Table.grid(padding=(0, 2))
        info.add_row("Model", f"[bold]{self.cfg['id']}[/]", "Spent (total)", f"[bold yellow]${self.cost:.4f}[/]")
        info.add_row("Runs", f"{self.args.runs} per test" + (f", first {self.args.limit} items" if self.args.limit else ""),
                     "This session", f"${spent:.4f}")
        eff = (self.cfg.get("extra") or {}).get("reasoning")
        info.add_row("Settings", f"T={self.cfg.get('temperature', 0.6)}, max_tokens={self.cfg.get('max_tokens', 30)}"
                     + (f", reasoning={eff}" if eff else ""),
                     "Per answer", f"${per:.5f}")
        info.add_row("Elapsed / ETA", f"{el / 60:.1f} min / " + (f"{left / rate / 60:.1f} min" if rate else "-"),
                     "Projected total", f"${self.cost + per * left:.3f}" if per else "-")
        prog = Table(box=None, expand=True)
        for c in ["Test", "", "Answered", "NA", "Net errors"]:
            prog.add_column(c, ratio=3 if c == "" else None, no_wrap=True)
        for t in TESTS:
            d, na, n = counts[t]
            prog.add_row(NAMES[t], ProgressBar(total=n, completed=d), f"{d}/{n}", str(na), str(self.errors[t]))
        prog.add_row("[bold]All[/]", ProgressBar(total=total, completed=done), f"[bold]{done}/{total}[/]", "", "")
        rec = Table(box=None, expand=True)
        for c in ["Time", "Test", "Item", "Run", "Answer", "Cost"]:
            rec.add_column(c, no_wrap=True)
        for ts, t, q, r, c, cs in self.recent:
            rec.add_row(ts, t, q, str(r), c if c != "NA" else "[yellow]NA[/]", f"${cs:.5f}")
        foot = f"[red]last error: {self.last_error}[/]\n" if self.last_error else ""
        foot += f"[dim]Saved after every answer -> {self.path}  |  Ctrl-C is safe; rerun to resume.[/]"
        return Group(Panel(info, title="Run & cost", border_style="yellow"),
                     Panel(prog, title="Progress", border_style="green"),
                     Panel(rec, title="Latest answers", border_style="blue"), foot)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="finalv/models/<name>.py, its name, or an OpenRouter id")
    ap.add_argument("--runs", type=int, default=1, help="repeats per test (default 1: one answer per item)")
    ap.add_argument("--limit", type=int, help="pilot: only the first N items of each test")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-cost", type=float, help="stop once this many USD are spent")
    ap.add_argument("--max-tokens", "--max_tokens", "--max_token", dest="max_tokens", type=int,
                    help="token ceiling per reply (default 30, or the model file's value); use ~4000 for reasoning models")
    ap.add_argument("--effort", choices=["off", "minimal", "low", "medium", "high"],
                    help="reasoning budget sent to OpenRouter; 'off' disables thinking where the model allows it "
                         "(cuts cost a lot; Grok 4 cannot turn it off, use 'low')")
    ap.add_argument("--mock", action="store_true", help="offline test, no key, no cost")
    args = ap.parse_args()

    cfg = load_model(args.model)
    if args.max_tokens:
        cfg["max_tokens"] = args.max_tokens
    if args.effort:
        cfg.setdefault("extra", {})["reasoning"] = ({"enabled": False} if args.effort == "off"
                                                    else {"effort": args.effort})
    key = dotenv_values(HERE / ".env").get("OR_KEY")
    if not key and not args.mock:
        raise SystemExit(f"Put OR_KEY=sk-or-... in {HERE / '.env'}")
    run = Run(cfg, args)
    queue = asyncio.Queue()
    for job in run.todo():
        queue.put_nowait(job)

    async with httpx.AsyncClient(base_url="https://openrouter.ai/api/v1/", timeout=180,
                                 headers={"Authorization": f"Bearer {key}"}) as http:
        async def worker():
            while not queue.empty():
                if args.max_cost and run.cost >= args.max_cost:
                    run.last_error = f"budget ${args.max_cost} reached - stopped"
                    return
                await run.ask(http, *queue.get_nowait())
                if run.session_n % 25 == 0:
                    run.export()

        with Live(run.view(), refresh_per_second=4) as live:
            tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
            try:
                while not all(t.done() for t in tasks):
                    live.update(run.view())
                    await asyncio.sleep(0.25)
                for t in tasks:
                    t.result()                           # surface a fatal error (bad key, no credit)
            finally:
                for t in tasks:
                    t.cancel()
                run.export()
                live.update(run.view())
    na = sum(run.count(t)[1] for t in TESTS)
    print(f"Done. ${run.cost:.4f} spent. Files in {run.out}" + (f"  ({na} NA - rerun to re-ask them)" if na else ""))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped. Everything answered so far is saved; rerun the same command to continue.")
