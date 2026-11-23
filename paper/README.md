# Political Alignment in Large Language Models: A Multidimensional Audit of Psychometric Identity and Behavioral Bias — MultiDems @ EACL 2027 submission

Self-contained paper directory. Everything in the PDF is regenerated from `data/`.

```
data/        instruments/ (scoring source of PCT, SapplyValues, 8Values)
             questionnaire runs (26 models x 10 runs, item level + scores), news predictions,
             7-model prompt-paraphrase study
             single_run/  4 chat models added Nov 2026 (same prompt, one run; from ../finalv)
             decision/    29 OpenRouter decision models, answer probabilities per item,
                          choice and yes/no formats (from ../finald)
             decision_news/  22 decision models rating the news articles (from ../finalb)
analysis/    common.py (models, palette, loaders) · analyze.py (all statistics)
             keys.py (instrument keys + exact scoring re-implementations) · items_deep.py (RQ4) · tsne.py (App. E)
             figures.py (all figures) · tables.py (LaTeX tables + tables/numbers.tex macros)
             scoring_inst.py (instrument scoring, incl. exact expected scores from probabilities)
             decision.py + decision_out.py (decision-model analysis, tables, figures, numbers_dec.tex)
results/     outputs for all 30 chat models (news analyses: the 26 with news ratings; sensitivity)
results24/   outputs for the 28 same-protocol chat models (main analyses) and decision models
figures/ tables/ logos/   generated assets; logos from @lobehub/icons-static-png
main.tex custom.bib acl.sty acl_natbib.bst
```

Rebuild:
```
python analysis/analyze.py && COHORT=24 python analysis/analyze.py && COHORT=24 python analysis/items_deep.py && COHORT=24 python analysis/tsne.py && COHORT=24 python analysis/similarity.py
COHORT=24 python analysis/decision.py && COHORT=24 python analysis/decision_out.py && COHORT=24 python analysis/decision_news.py
python analysis/tables.py && python analysis/figures.py
latexmk -pdf main.tex
```
Requires pandas, scipy, scikit-learn, statsmodels, matplotlib, adjustText.

Note: Grok 4 and Gemini 2.5 Pro answered the questionnaires under a different prompt
(own beliefs, 10 ratings per call, T=1.0); they are excluded from questionnaire analyses (COHORT=24)
and reported as a sensitivity check. The four Nov 2026 additions have one run each: they enter
every analysis except run-to-run reliability, run-level t-SNE and the news task.
