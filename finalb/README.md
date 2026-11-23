# finalb — news slant rating by decision models

22 OpenRouter decision models (the 20 cheapest on the questionnaire runs, plus HeyGen Decision-1 Turbo and
Gemini Decision-1) rate the 1,063 articles of the paper's news task, one request per article.

```bash
python3 finalb/run.py --limit 3        # pilot
python3 finalb/run.py                  # all 22 models, 32 parallel requests, combined dashboard
```

Uses `OR_KEY` from `finald/.env`. Input = title + article text; one 7-way choice question on the chat
models' scale (Far Left -3 ... Neutral 0 ... Far Right +3). Span-01 only accepts yes/no questions, so it gets
one per option (renormalised). Each answer is fsynced to `output/<model>/answers.jsonl`; rerun to resume.
Articles a model rejects (e.g. too long) are logged in `errors.jsonl` and retried on the next run.

`output/<model>/predictions.csv`: row, story_id, source_bias, `expected` (probability-weighted rating,
continuous -3..+3, comparable to the chat models' scores), `top` (most likely option), `top_p`, p_* per option.
Flags: `--model ID`, `--limit N`, `--concurrency N`, `--max-cost USD`, `--mock`.
