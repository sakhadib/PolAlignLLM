# finalv — questionnaire runs for new models

Runs the Political Compass (62 items), 8Values (70) and SapplyValues (46) on one model through OpenRouter,
one run each by default (178 answers per model; `--t N` for N runs), with the same prompts as the paper's cohort.

```bash
pip install -r finalv/requirements.txt
# finalv/.env must contain:  OR_KEY=sk-or-...
python3 finalv/run.py --model finalv/models/grok_4.py --limit 3   # pilot: 3 items per test
python3 finalv/run.py --model grok_4
python3 finalv/run.py --model nemotron
```

`--model` takes a file in `models/`, its name, or a raw OpenRouter id. Other flags: `--max-tokens N` (default 30; use ~4000 for reasoning models with a raw id), `--effort off|minimal|low|medium|high` (reasoning budget; the main cost driver for thinking models), `--t N` (runs per test, default 1), `--limit`,
`--concurrency` (default 32), `--max-cost USD`, `--mock` (offline test).

**Saving.** Each answer is appended to `output/<model>/answers.jsonl` (fsynced) as soon as it arrives.
Ctrl-C or a crash loses nothing; rerun the same command to continue. Unparsable answers (NA) are re-asked
up to 5 times per call, and again on the next rerun.

**Outputs** (same format as the paper's data): `<test>_predictions.csv` (columns `pred_[k]_<slug>`) and
`<test>_scores_runs.csv` (scored only for complete runs).

**Settings.** T = 0.6, as in the OpenRouter runs. `max_tokens` is 4000 in the model files because Grok 4 and
Nemotron reason before answering; the answer itself is a few tokens.
