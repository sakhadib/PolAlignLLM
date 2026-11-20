# Many Scales, One Axis — MultiDems @ EACL 2027 submission

Self-contained paper directory. Everything in the PDF is regenerated from `data/`.

```
data/        instruments/ (scoring source of PCT, SapplyValues, 8Values)
             questionnaire runs (26 models x 10 runs, item level + scores), news predictions,
             7-model prompt-paraphrase study
analysis/    common.py (models, palette, loaders) · analyze.py (all statistics)
             keys.py (instrument keys + exact scoring re-implementations) · items_deep.py (RQ4)
             figures.py (all figures) · tables.py (LaTeX tables + tables/numbers.tex macros)
results/     outputs for all 26 models (news analyses, sensitivity)
results24/   outputs for the 24 same-protocol models (main questionnaire analyses)
figures/ tables/ logos/   generated assets; logos from @lobehub/icons-static-png
main.tex custom.bib acl.sty acl_natbib.bst
```

Rebuild:
```
python analysis/analyze.py && COHORT=24 python analysis/analyze.py && COHORT=24 python analysis/items_deep.py
python analysis/tables.py && python analysis/figures.py
latexmk -pdf main.tex
```
Requires pandas, scipy, scikit-learn, statsmodels, matplotlib, adjustText.

Note: Grok 4 and Gemini 2.5 Pro answered the questionnaires under a different prompt
(own beliefs, 10 ratings per call, T=1.0); they are excluded from questionnaire analyses (COHORT=24)
and reported as a sensitivity check.
