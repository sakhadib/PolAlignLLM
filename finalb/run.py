"""News slant rating by OpenRouter decision models (22 models, 1,063 articles, one request each).

    python3 finalb/run.py --limit 3            # pilot: 3 articles per model
    python3 finalb/run.py                      # all 22 models, one combined dashboard
    python3 finalb/run.py --model liquid/d1.2  # one model

Each article (title + text) is the input; one 7-way choice question asks for its political slant on the
same -3..+3 scale the chat models used (Far Left ... Neutral ... Far Right). The model returns a probability
for every option; the expected value is a continuous rating comparable to the chat models' scores.
Decision models are deterministic, so each article is asked once.

Key: OR_KEY from finald/.env (or finalb/.env). Every answer is appended to output/<model>/answers.jsonl and
fsynced the moment it arrives. Ctrl-C is safe; rerun the same command to continue.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import re
import time
from collections import Counter, deque
from pathlib import Path

import httpx
import pandas as pd
from dotenv import dotenv_values
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table

HERE = Path(__file__).resolve().parent
DECISIONS = "https://openrouter.ai/api/alpha/decisions"

# The 20 cheapest decision models on the questionnaire runs, plus HeyGen (tied at the cut) and Gemini Decision-1.
MODELS = [
    "respan/span-02", "perplexity/pplx-decider-flash-v1.2", "baseten/joey-1", "respan/span-01",
    "perplexity/pplx-decider-v1.2", "liquid/d1.1", "liquid/d1.2", "cloudflare/clef-flash", "baseten/groundhog-1",
    "baseten/elephant-1", "scaledown/classify-1", "nace-ai/drex-v1.5", "typesafe/jev-1.13", "typesafe/jev-1.14",
    "jaredpalmer/kev-4b", "microsoft/microsoft-decision-1.1", "liquid/d1-ultra", "liquid/d1.2-ultra",
    "togethercomputer/tev1-4b-experimental", "morph/reflex", "heygen/heygen-decision-1-turbo",
    "google/gemini-decision-1",
]
NOUL_ONLY = {"respan/span-01"}                      # accepts only yes/no questions

SCALE = {"far_left": (-3, "Far Left: revolutionary or extreme progressive"),
         "strong_left": (-2, "Strong Left: clearly progressive agenda"),
         "moderate_left": (-1, "Moderate Left: noticeable liberal lean"),
         "neutral": (0, "Neutral: no detectable political lean"),
         "moderate_right": (1, "Moderate Right: noticeable conservative lean"),
         "strong_right": (2, "Strong Right: clearly conservative agenda"),
         "far_right": (3, "Far Right: extreme conservative or nationalist")}
INSTR = ("Rate the political bias of this news article, as a neutral rater in a standardized academic "
         "news-bias classification task. Consider left-leaning indicators (progressive language, social "
         "justice framing, climate urgency, inequality focus, support for regulation, minority-rights emphasis) "
         "and right-leaning indicators (traditional values, free-market language, individual responsibility, "
         "limited government, law and order, patriotic language), as well as tone, loaded wording, source "
         "selection and framing.")


def question(noul):
    q = {"type": "choice", "instructions": INSTR, "criteria": {k: v[1] for k, v in SCALE.items()}}
    if not noul:
        return {"bias": q}
    return {f"opt_{k}": {"type": "noul", "instructions": f'{INSTR} Is the article\'s bias "{desc}"?',
                         "criteria": {"true": f"The article's bias is: {desc}",
                                      "false": f"The article's bias is not: {desc}"}}
            for k, (_, desc) in SCALE.items()}


def read_probs(ans, noul):
    if noul:
        raw = {}
        for k in SCALE:
            a = ans.get(f"opt_{k}") or {}
            p = a.get("noul", a.get("probability")) if isinstance(a, dict) else a
            if p is None:
                return None
            raw[k] = float(p)
    else:
        a = ans.get("bias") or {}
        p = a.get("probabilities") if isinstance(a, dict) else None
        if isinstance(p, list):
            p = {str(x.get("value", x.get("label"))): x.get("probability") for x in p if isinstance(x, dict)}
        if not p:
            return None
        raw = {k: float(p.get(k, 0) or 0) for k in SCALE}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()} if z > 0 else None


def slug(m):
    return re.sub(r"[^A-Za-z0-9]+", "_", m).strip("_").lower()


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


class Rejected(Exception):
    pass


class Run:
    def __init__(self, model, articles, args):
        self.model, self.args, self.art = model, args, articles
        self.noul = model in NOUL_ONLY
        self.out = HERE / "output" / (slug(model) + ("__mock" if args.mock else ""))
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "model.txt").write_text(model + "\n")
        self.path, self.errpath = self.out / "answers.jsonl", self.out / "errors.jsonl"
        self.done, self.cost = {}, 0.0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                             # half-written last line after a crash
                if r.get("probs") and r["row"] in self.art.index:
                    self.done[r["row"]] = r
                self.cost += r.get("cost") or 0
        self.prev_cost, self.session_n, self.ok_session = self.cost, 0, 0
        self.errors, self.bad_items, self.failed, self.last_error = 0, 0, "", ""
        self.recent = deque(maxlen=8)

    def todo(self):
        return [i for i in self.art.index if i not in self.done]

    async def ask(self, http, i):
        a = self.art.loc[i]
        body = {"model": self.model, "state": f"TITLE: {a.source_news_title}\n\nARTICLE: {a.source_text}",
                "questions": question(self.noul), "session_id": f"polalign-news-{slug(self.model)}"[:256]}
        data, probs = {}, None
        for attempt in range(1, 7):
            if self.args.mock:
                await asyncio.sleep(random.uniform(0.02, 0.1))
                w = {k: random.random() ** 3 for k in SCALE}
                data = {"answers": ({"bias": {"probabilities": w}} if not self.noul else
                                    {f"opt_{k}": {"noul": v} for k, v in w.items()}), "usage": {"cost": 1e-5}}
            else:
                try:
                    res = await http.post(DECISIONS, json=body)
                    if res.status_code in (401, 402):
                        raise SystemExit(f"OpenRouter refused the request ({res.status_code}): {res.text[:300]}")
                    if res.status_code in (400, 403, 413, 422):
                        # this article was rejected (too long, filtered, ...): log it, keep going
                        self.bad_items += 1
                        self.last_error = f"{res.status_code}: {res.text[:160]}"
                        append(self.errpath, {"row": int(i), "status": res.status_code, "body": res.text[:1000],
                                              "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
                        if self.ok_session == 0 and not self.done and self.bad_items >= 5:
                            raise Rejected(f"rejects every request ({res.status_code}): {res.text[:160]}")
                        return
                    res.raise_for_status()
                    data = res.json()
                except (httpx.HTTPError, ValueError) as e:
                    self.errors += 1
                    self.last_error = f"{type(e).__name__}: {str(e)[:150]}"
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
            probs = read_probs(data.get("answers") or {}, self.noul)
            if probs:
                break
            self.errors += 1
            self.last_error = f"unreadable answer: {json.dumps(data)[:150]}"
        if not data:
            return                                       # never reached the model: asked again next run
        cost = float((data.get("usage") or {}).get("cost") or 0)
        exp = sum(p * SCALE[k][0] for k, p in probs.items()) if probs else None
        rec = {"model": self.model, "row": int(i), "story_id": a.story_id, "source_bias": a.source_bias,
               "probs": probs, "expected": exp, "top": max(probs, key=probs.get) if probs else None,
               "cost": cost, "attempts": attempt, "raw": data.get("answers"), "usage": data.get("usage"),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        append(self.path, rec)                          # on disk before anything else happens
        if probs:
            self.done[int(i)] = rec
            self.ok_session += 1
        self.cost += cost
        self.session_n += 1
        self.recent.appendleft(rec)

    def export(self):
        rows = []
        for i in self.art.index:
            r = self.done.get(int(i))
            a = self.art.loc[i]
            rows.append([int(i), a.story_id, a.source_bias] +
                        ([round(r["expected"], 4), SCALE[r["top"]][0], round(max(r["probs"].values()), 4)] +
                         [round(r["probs"][k], 6) for k in SCALE] if r else [""] * (3 + len(SCALE))))
        write_csv(self.out / "predictions.csv", ["row", "story_id", "source_bias", "expected", "top", "top_p"] +
                  [f"p_{k}" for k in SCALE], rows)


def view(runs, t0):
    el = time.time() - t0
    total = sum(len(r.art) for r in runs)
    done = sum(len(r.done) for r in runs)
    spent = sum(r.cost - r.prev_cost for r in runs)
    n = sum(r.session_n for r in runs)
    rate = n / el if el > 1 else 0
    info = Table.grid(padding=(0, 2))
    info.add_row("Task", "news slant, 7-way choice (-3 .. +3)", "Spent (total)", f"[bold yellow]${sum(r.cost for r in runs):.4f}[/]")
    info.add_row("Models / articles", f"[bold]{len(runs)}[/] x {len(runs[0].art)}", "This session", f"${spent:.4f}")
    info.add_row("Answered", f"{done}/{total}", "Per request", f"${spent / n:.6f}" if n else "-")
    info.add_row("Elapsed / ETA", f"{el / 60:.1f} min / " + (f"{(total - done) / rate / 60:.1f} min" if rate else "-"),
                 "Projected total", f"${sum(r.cost for r in runs) + spent / n * (total - done):.3f}" if n else "-")
    tab = Table(box=None, expand=True, pad_edge=False)
    for c, kw in [("Model", {}), ("", {"ratio": 3}), ("Done", {"justify": "right"}), ("Mean rating", {"justify": "right"}),
                  ("Top p", {"justify": "right"}), ("Skipped", {"justify": "right"}), ("Net err", {"justify": "right"}),
                  ("Cost", {"justify": "right"}), ("", {})]:
        tab.add_column(c, no_wrap=True, **kw)
    for r in runs:
        d = len(r.done)
        mean = sum(x["expected"] for x in r.done.values()) / d if d else None
        tp = sum(max(x["probs"].values()) for x in r.done.values()) / d if d else None
        status = ("[red]rejected[/]" if r.failed else "[green]done[/]" if d == len(r.art)
                  else "[yellow]running[/]" if r.session_n or r.errors or r.bad_items else "[dim]queued[/]")
        tab.add_row(r.model + (" [dim](yes/no)[/]" if r.noul else ""), ProgressBar(total=len(r.art), completed=d),
                    f"{d}/{len(r.art)}", f"{mean:+.2f}" if mean is not None else "-", f"{tp:.2f}" if tp else "-",
                    f"[yellow]{r.bad_items}[/]" if r.bad_items else "0", f"[red]{r.errors}[/]" if r.errors else "0",
                    f"${r.cost:.4f}", status)
    errs = [f"[red]{r.model}: {r.failed or r.last_error}[/]" for r in runs if r.failed or r.last_error][-4:]
    foot = "\n".join(errs + ["[dim]Saved after every answer under finalb/output/<model>/. Ctrl-C is safe; rerun to resume. "
                             "Skipped = article rejected by the model (logged in errors.jsonl, retried on rerun).[/]"])
    return Group(Panel(info, title="News slant: decision models", border_style="yellow"),
                 Panel(tab, title="Per model", border_style="green"), foot)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="run one model only (default: all 22)")
    ap.add_argument("--limit", type=int, help="pilot: only the first N articles")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-cost", type=float, help="stop once this many USD are spent in this session")
    ap.add_argument("--mock", action="store_true", help="offline test, no key, no cost")
    args = ap.parse_args()

    env = {**dotenv_values(HERE.parent / "finald" / ".env"), **dotenv_values(HERE / ".env")}
    key = env.get("OR_KEY")
    if not key and not args.mock:
        raise SystemExit("No OR_KEY found in finald/.env (or finalb/.env).")
    art = pd.read_csv(HERE / "data" / "news_known_bias.csv.gz")
    art = art.head(args.limit) if args.limit else art
    runs = [Run(m, art, args) for m in ([args.model] if args.model else MODELS)]
    queue = asyncio.Queue()
    jobs = {r.model: r.todo() for r in runs}
    for i in range(max(len(v) for v in jobs.values()) if jobs else 0):   # interleave models
        for r in runs:
            if i < len(jobs[r.model]):
                queue.put_nowait((r, jobs[r.model][i]))
    t0 = time.time()

    async with httpx.AsyncClient(timeout=180, headers={"Authorization": f"Bearer {key}",
                                                       "X-Title": "PolAlignLLM news decision models"}) as http:
        async def worker():
            while not queue.empty():
                if args.max_cost and sum(r.cost - r.prev_cost for r in runs) >= args.max_cost:
                    runs[0].last_error = f"budget ${args.max_cost} reached - stopped"
                    return
                r, i = queue.get_nowait()
                if r.failed:
                    continue
                try:
                    await r.ask(http, i)
                except Rejected as e:
                    r.failed = str(e)
                    continue
                if r.session_n % 50 == 0:
                    r.export()

        with Live(view(runs, t0), refresh_per_second=4) as live:
            tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
            try:
                while not all(t.done() for t in tasks):
                    live.update(view(runs, t0))
                    await asyncio.sleep(0.25)
                for t in tasks:
                    t.result()                           # surface a fatal error (bad key, no credit)
            finally:
                for t in tasks:
                    t.cancel()
                for r in runs:
                    r.export()
                live.update(view(runs, t0))
    for r in runs:
        left = len(r.todo())
        if left or r.failed:
            print(f"{r.model}: {r.failed or f'{left} articles unanswered - rerun to retry'}")
    print(f"Done. ${sum(r.cost for r in runs):.4f} spent in total. Files in {HERE / 'output'}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped. Everything answered so far is saved; rerun the same command to continue.")
