# finald — questionnaires on decision models

Decision models (OpenRouter Decisions API, as used in BLAD) return calibrated probabilities instead of text.
This runs the Political Compass (62), 8Values (70) and SapplyValues (46) on one, **one request per
statement** (178 per model). The same request always returns the same answer, so repeats add nothing.

```bash
pip install -r finald/requirements.txt          # finald/.env must contain OR_KEY=sk-or-...
python3 finald/run.py --list                    # decision models currently on OpenRouter
python3 finald/run.py --model liquid/d1.2 --limit 3
python3 finald/run.py --model liquid/d1.2
python3 finald/run.py --model respan/span-01 --noul   # yes/no-only models
python3 finald/run.py --all --noul                    # every model run so far, combined dashboard
```

**Question.** State = the statement; one native `choice` question with the instrument's own scale
(sd/d/a/sa for PCT; sd/d/n/a/sa for 8Values and SapplyValues, neutral only when genuinely undecidable).
With `--noul`, each option becomes a yes/no question and the probabilities are renormalised (BLAD's encoding).

**Saving.** Each answer (normalised probabilities, raw API answer, usage, cost) is appended to
`output/<model>/answers.jsonl` and fsynced as it arrives. Ctrl-C is safe; rerun to resume.

**Outputs.**
- `<test>_predictions.csv` — most likely code per item, paper format (`pred_[1]_<slug>`)
- `<test>_probs.csv` — full distribution per item and top probability
- `scores.csv` — per axis: `hard` (score of the most likely answers, as for chat models) and `soft`
  (expected score under the model's distribution; exact, because all three instruments score answers linearly)
