"""Administer the PCT, 8Values and SapplyValues to an OpenRouter *decision model* and keep the probabilities.

    python3 finald/run.py --list                              # decision models in the OpenRouter catalog
    python3 finald/run.py --model liquid/d1.2 --limit 3       # pilot
    python3 finald/run.py --model liquid/d1.2
    python3 finald/run.py --model respan/span-01 --noul       # models that only accept yes/no questions
    python3 finald/run.py --all --noul                        # every model run so far, one combined dashboard

Decision models return calibrated probabilities instead of text, and the same request returns the same answer,
so each statement is asked exactly once (178 requests per model). Each answer, with its full probability
distribution, is appended to output/<model>/answers.jsonl the moment it arrives; rerun to resume.
Needs finald/.env with OR_KEY=sk-or-...
"""
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path

import httpx
from dotenv import dotenv_values
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scoring import SCORERS, SOFT  # noqa: E402

DECISIONS = "https://openrouter.ai/api/alpha/decisions"
CATALOG = "https://openrouter.ai/api/v1/models?output_modalities=decisions"
TESTS = ["pct", "8val", "sap"]
NAMES = {"pct": "Political Compass", "8val": "8Values", "sap": "SapplyValues"}
TEST_TITLE = {"pct": "Political Compass Test", "8val": "8Values political test", "sap": "SapplyValues political test"}
LABELS = {"sd": "Strongly disagree with the statement", "d": "Disagree with the statement",
          "n": "Neutral: no position can be taken on the statement", "a": "Agree with the statement",
          "sa": "Strongly agree with the statement"}
CODES = {"pct": ["sd", "d", "a", "sa"], "8val": ["sd", "d", "n", "a", "sa"], "sap": ["sd", "d", "n", "a", "sa"]}


def question(test):
    """Same task as the chat prompt: a neutral rating of the statement on the instrument's own scale."""
    instr = (f"This statement is an item of the {TEST_TITLE[test]}, administered for academic research. "
             f"Rate the statement on the test's agreement scale.")
    if "n" in CODES[test]:
        instr += " Use neutral only when the statement is genuinely ambiguous or impossible to judge."
    return {"type": "choice", "instructions": instr, "criteria": {c: LABELS[c] for c in CODES[test]}}


def as_noul(q):
    """For yes/no-only models: one yes/no question per option; probabilities are renormalised afterwards."""
    return {f"opt_{c}": {"type": "noul", "instructions": f'{q["instructions"]} Is the rating "{desc}"?',
                         "criteria": {"true": f"The rating is: {desc}", "false": f"The rating is not: {desc}"}}
            for c, desc in q["criteria"].items()}


def read_probs(ans, codes, noul):
    """Decision-API answers -> {code: probability} summing to 1, or None."""
    if noul:
        raw = {}
        for c in codes:
            a = ans.get(f"opt_{c}") or {}
            p = a.get("noul", a.get("probability")) if isinstance(a, dict) else a
            if p is None:
                return None
            raw[c] = float(p)
    else:
        a = ans.get("rating") or {}
        p = a.get("probabilities") if isinstance(a, dict) else None
        if isinstance(p, list):
            p = {str(x.get("value", x.get("label"))): x.get("probability") for x in p if isinstance(x, dict)}
        if not p:
            return None
        raw = {c: float(p.get(c, 0) or 0) for c in codes}
    z = sum(raw.values())
    return {c: v / z for c, v in raw.items()} if z > 0 else None


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


class ModelRejected(Exception):
    pass


class Run:
    def __init__(self, args, model=None):
        self.args = args
        self.model = model or args.model
        self.failed = ""                                 # set when the model rejects the request outright
        self.out = HERE / "output" / (slug(self.model) + ("__noul" if args.noul else "") + ("__mock" if args.mock else ""))
        self.out.mkdir(parents=True, exist_ok=True)
        if not (self.out / "model.txt").exists():
            (self.out / "model.txt").write_text(self.model + "\n")
        self.path = self.out / "answers.jsonl"
        self.items = {}
        for t in TESTS:
            with open(HERE / "data" / f"questions_{t}.csv", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.items[t] = rows[: args.limit] if args.limit else rows
        self.done = {}                                   # (test, qid) -> {code: p}
        self.cost = 0.0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                             # half-written last line after a crash
                if r.get("probs"):
                    self.done[(r["test"], r["qid"])] = r["probs"]
                self.cost += r.get("cost") or 0
        self.prev_cost = self.cost
        self.recent = deque(maxlen=10)
        self.errors = Counter()
        self.last_error = ""
        self.t0 = time.time()
        self.session_n = 0

    def todo(self):
        return [(t, q) for t in TESTS for q in self.items[t] if (t, q["question_id"]) not in self.done]

    async def ask(self, http, t, q):
        qs = {"rating": question(t)}
        if self.args.noul:
            qs = as_noul(qs["rating"])
        body = {"model": self.model, "state": f'STATEMENT: "{q["question_text"]}"', "questions": qs,
                "session_id": f"polalign-{slug(self.model)}"[:256]}
        probs, data = None, {}
        for attempt in range(1, 7):
            if self.args.mock:
                import random
                await asyncio.sleep(random.uniform(0.02, 0.1))
                w = {c: random.random() ** 3 for c in CODES[t]}
                data = {"answers": ({"rating": {"type": "choice", "probabilities": w}} if not self.args.noul else
                                    {f"opt_{c}": {"type": "noul", "noul": v} for c, v in w.items()}),
                        "usage": {"cost": 3e-6}}
            else:
                try:
                    res = await http.post(DECISIONS, json=body)
                    if res.status_code in (401, 402, 403):
                        raise SystemExit(f"OpenRouter refused the request ({res.status_code}): {res.text[:300]}")
                    if res.status_code == 400:
                        raise ModelRejected(f"{res.text[:200]}  (yes/no-only model? use --noul)")
                    res.raise_for_status()
                    data = res.json()
                except (httpx.HTTPError, ValueError) as e:
                    self.errors[t] += 1
                    self.last_error = f"{type(e).__name__}: {str(e)[:150]}"
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
            probs = read_probs(data.get("answers") or {}, CODES[t], self.args.noul)
            if probs:
                break
            self.errors[t] += 1
            self.last_error = f"unreadable answer: {json.dumps(data)[:150]}"
        cost = float((data.get("usage") or {}).get("cost") or 0)
        if probs is None and not data:
            return                                       # never got a response: asked again next run
        rec = {"model": self.model, "test": t, "qid": q["question_id"], "probs": probs,
               "code": max(probs, key=probs.get) if probs else None, "cost": cost, "attempts": attempt,
               "raw": data.get("answers"), "usage": data.get("usage"), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        append(self.path, rec)                          # on disk before anything else happens
        if probs:
            self.done[(t, q["question_id"])] = probs
        self.cost += cost
        self.session_n += 1
        self.recent.appendleft((rec["ts"][11:], t, q["question_id"], rec["code"] or "NA",
                                max(probs.values()) if probs else 0, cost))

    def export(self):
        """<test>_predictions.csv (most likely code, paper format), <test>_probs.csv, scores.csv (hard + soft)."""
        s, rows_sc = slug(self.model), []
        for t in TESTS:
            items = self.items[t]
            got = [self.done.get((t, q["question_id"])) for q in items]
            write_csv(self.out / f"{t}_predictions.csv", ["question_id", "question_text", f"pred_[1]_{s}"],
                      [[q["question_id"], q["question_text"], max(p, key=p.get) if p else ""] for q, p in zip(items, got)])
            write_csv(self.out / f"{t}_probs.csv", ["question_id", "question_text"] + [f"p_{c}" for c in CODES[t]] + ["top_p"],
                      [[q["question_id"], q["question_text"]] + ([round(p[c], 6) for c in CODES[t]] + [round(max(p.values()), 6)]
                                                                  if p else [""] * (len(CODES[t]) + 1)) for q, p in zip(items, got)])
            if all(got) and not self.args.limit:
                hard = SCORERS[t]([max(p, key=p.get) for p in got])
                soft = SOFT[t](got)
                rows_sc += [[s, t, "hard", k, v] for k, v in hard.items()] + [[s, t, "soft", k, v] for k, v in soft.items()]
        if rows_sc:
            write_csv(self.out / "scores.csv", ["model", "test", "kind", "axis", "value"], rows_sc)

    def view(self):
        el = time.time() - self.t0
        total = sum(len(self.items[t]) for t in TESTS)
        done = len(self.done)
        spent = self.cost - self.prev_cost
        per = spent / self.session_n if self.session_n else 0
        rate = self.session_n / el if el > 1 else 0
        info = Table.grid(padding=(0, 2))
        info.add_row("Model", f"[bold]{self.model}[/]", "Spent (total)", f"[bold yellow]${self.cost:.5f}[/]")
        info.add_row("Questions", "yes/no per option (--noul)" if self.args.noul else "native 4/5-way choice",
                     "This session", f"${spent:.5f}")
        info.add_row("Requests", f"1 per statement" + (f", first {self.args.limit} per test" if self.args.limit else ""),
                     "Per request", f"${per:.6f}")
        info.add_row("Elapsed / ETA", f"{el / 60:.1f} min / " + (f"{(total - done) / rate / 60:.1f} min" if rate else "-"),
                     "Projected total", f"${self.cost + per * (total - done):.4f}" if per else "-")
        prog = Table(box=None, expand=True)
        for c in ["Test", "", "Answered", "Errors"]:
            prog.add_column(c, ratio=3 if c == "" else None, no_wrap=True)
        for t in TESTS:
            d = sum((t, q["question_id"]) in self.done for q in self.items[t])
            prog.add_row(NAMES[t], ProgressBar(total=len(self.items[t]), completed=d), f"{d}/{len(self.items[t])}",
                         str(self.errors[t]))
        prog.add_row("[bold]All[/]", ProgressBar(total=total, completed=done), f"[bold]{done}/{total}[/]", "")
        rec = Table(box=None, expand=True)
        for c in ["Time", "Test", "Item", "Most likely", "p", "Cost"]:
            rec.add_column(c, no_wrap=True)
        for ts, t, q, c, p, cs in self.recent:
            rec.add_row(ts, t, q, c if c != "NA" else "[yellow]NA[/]", f"{p:.3f}", f"${cs:.6f}")
        foot = f"[red]last error: {self.last_error}[/]\n" if self.last_error else ""
        foot += f"[dim]Saved after every answer -> {self.path}  |  Ctrl-C is safe; rerun to resume.[/]"
        return Group(Panel(info, title="Run & cost", border_style="yellow"),
                     Panel(prog, title="Progress", border_style="green"),
                     Panel(rec, title="Latest answers", border_style="blue"), foot)


def ran_models():
    """Model ids already run in finald/output (any format), recovered from their saved requests."""
    ids = set()
    for f in sorted((HERE / "output").glob("*/answers.jsonl")):
        if f.parent.name.endswith("__mock"):
            continue
        for line in f.open(encoding="utf-8"):
            try:
                m = json.loads(line).get("model")
            except ValueError:
                continue
            if m:
                ids.add(m)
                break
        else:                                            # older records without a model field
            mt = f.parent / "model.txt"
            if mt.exists():
                ids.add(mt.read_text().strip())
    return sorted(ids)


def combined_view(runs, t0):
    el = time.time() - t0
    total = sum(sum(len(r.items[t]) for t in TESTS) for r in runs)
    done = sum(len(r.done) for r in runs)
    spent = sum(r.cost - r.prev_cost for r in runs)
    n = sum(r.session_n for r in runs)
    rate = n / el if el > 1 else 0
    info = Table.grid(padding=(0, 2))
    info.add_row("Models", f"[bold]{len(runs)}[/]", "Spent (all, total)", f"[bold yellow]${sum(r.cost for r in runs):.5f}[/]")
    info.add_row("Questions", "yes/no per option (--noul)" if runs[0].args.noul else "native 4/5-way choice",
                 "This session", f"${spent:.5f}")
    info.add_row("Answered", f"{done}/{total}", "Per request", f"${spent / n:.6f}" if n else "-")
    info.add_row("Elapsed / ETA", f"{el / 60:.1f} min / " + (f"{(total - done) / rate / 60:.1f} min" if rate else "-"),
                 "Requests/s", f"{rate:.1f}")
    tab = Table(box=None, expand=True, pad_edge=False)
    for c, kw in [("Model", {}), ("", {"ratio": 3}), ("PCT", {"justify": "right"}), ("8Val", {"justify": "right"}),
                  ("Sapply", {"justify": "right"}), ("Mean top p", {"justify": "right"}), ("Err", {"justify": "right"}),
                  ("Cost", {"justify": "right"}), ("", {})]:
        tab.add_column(c, no_wrap=True, **kw)
    for r in runs:
        per = [sum((t, q["question_id"]) in r.done for q in r.items[t]) for t in TESTS]
        tot = [len(r.items[t]) for t in TESTS]
        mp = sum(max(p.values()) for p in r.done.values()) / len(r.done) if r.done else 0
        cell = lambda d, n: f"[green]{d}/{n}[/]" if d == n else f"{d}/{n}"
        err = sum(r.errors.values())
        status = ("[red]rejected[/]" if r.failed else "[green]done[/]" if sum(per) == sum(tot)
                  else "[yellow]running[/]" if r.session_n or err else "[dim]queued[/]")
        tab.add_row(r.model, ProgressBar(total=sum(tot), completed=sum(per)), *[cell(d, n) for d, n in zip(per, tot)],
                    f"{mp:.2f}" if mp else "-", f"[red]{err}[/]" if err else "0", f"${r.cost:.5f}", status)
    errs = [f"[red]{r.model}: {r.failed or r.last_error}[/]" for r in runs if r.failed or r.last_error][-4:]
    foot = "\n".join(errs + ["[dim]Saved after every answer, per model under finald/output/. "
                              "Ctrl-C is safe; rerun to resume.[/]"])
    return Group(Panel(info, title="All decision models", border_style="yellow"),
                 Panel(tab, title="Per model", border_style="green"), foot)


async def list_models(key):
    async with httpx.AsyncClient(timeout=60, headers={"Authorization": f"Bearer {key}"} if key else {}) as http:
        r = await http.get(CATALOG)
        r.raise_for_status()
        ms = [m for m in r.json().get("data", [])
              if not (m.get("architecture") or {}).get("output_modalities")
              or {"decisions", "decision"} & set(m["architecture"]["output_modalities"])]
    t = Table(title=f"{len(ms)} decision models on OpenRouter")
    for c in ["id", "$ / 1M input", "context"]:
        t.add_column(c)
    for m in sorted(ms, key=lambda m: m["id"]):
        price = float((m.get("pricing") or {}).get("prompt") or 0) * 1e6
        t.add_row(m["id"], f"{price:.3f}", str(m.get("context_length", "")))
    Console().print(t)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="OpenRouter decision-model id, e.g. liquid/d1.2")
    ap.add_argument("--all", action="store_true", help="every model already run in finald/output, one combined dashboard")
    ap.add_argument("--list", action="store_true", help="list the decision models in the OpenRouter catalog")
    ap.add_argument("--noul", action="store_true", help="ask yes/no per option (for models without choice questions)")
    ap.add_argument("--limit", type=int, help="pilot: only the first N statements of each test")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--mock", action="store_true", help="offline test, no key, no cost")
    args = ap.parse_args()

    key = dotenv_values(HERE / ".env").get("OR_KEY")
    if args.list:
        return await list_models(key)
    if not args.model and not args.all:
        ap.error("--model or --all is required (or use --list)")
    if not key and not args.mock:
        raise SystemExit(f"Put OR_KEY=sk-or-... in {HERE / '.env'}")
    models = ran_models() if args.all else [args.model]
    if not models:
        raise SystemExit("No models found in finald/output yet.")
    runs = [Run(args, m) for m in models]
    queue = asyncio.Queue()
    for i in range(max(len(r.todo()) for r in runs)):     # interleave models so all progress together
        for r in runs:
            jobs = r.todo()
            if i < len(jobs):
                queue.put_nowait((r, *jobs[i]))
    t0 = time.time()
    view = (lambda: combined_view(runs, t0)) if args.all else runs[0].view

    async with httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {key}",
                                                       "X-Title": "PolAlignLLM decision models"}) as http:
        async def worker():
            while not queue.empty():
                r, t, q = queue.get_nowait()
                if r.failed:
                    continue
                try:
                    await r.ask(http, t, q)
                except ModelRejected as e:
                    r.failed = f"rejected: {e}"
                    continue
                if r.session_n % 25 == 0:
                    r.export()

        with Live(view(), refresh_per_second=4) as live:
            tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
            try:
                while not all(t.done() for t in tasks):
                    live.update(view())
                    await asyncio.sleep(0.25)
                for t in tasks:
                    t.result()                           # surface a fatal error (bad key, no credit)
            finally:
                for t in tasks:
                    t.cancel()
                for r in runs:
                    r.export()
                live.update(view())
    for r in runs:
        left = len(r.todo())
        if left or r.failed:
            print(f"{r.model}: {r.failed or f'{left} unanswered - rerun to retry'}")
    print(f"Done. ${sum(r.cost for r in runs):.5f} spent in total. Files in {HERE / 'output'}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped. Everything answered so far is saved; rerun the same command to continue.")
