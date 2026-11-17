# FDS Kaggle Challenge: Pokémon Battle Prediction (Gen 2 OU)

Team *Shadow Forces* · Sapienza University of Rome, Foundations of Data Science 2026–27

Predict whether Player 1 wins a Gen 2 OU battle from Player 1's team, Player 2's lead and the first 30 active turns. Classical models only (scikit-learn), competition data plus the supplied type chart only.

## Our three final submissions

| # | Submission | Script | Dev 5-fold CV | Holdout | Private LB | Public LB |
|---|---|---|---|---|---|---|
| 1 | Original ensemble (base + composition features) | `final.py` | 67.7% | 67.4% | 66.80 | 68.73 |
| 2 | v4 (69 selected features, tuned HGB) | `final_v4.py` | 68.6% | 68.1% | 66.97 | 69.27 |
| 3 | `submission_ens` (149 engineered features) | `src/make_submission_ens.py` | ≈ 67.4% | n/a | 67.00 | 68.80 |

All three are soft-voting ensembles of HistGradientBoosting, random forest and logistic regression.

## Repository layout

```
features.py              base features + composition flags (used by final.py, final_v4.py)
features2.py             extra feature groups A-E (used by final_v4.py)
selected_features.txt    the 69 features kept for v4 (permutation importance, dev split only)
final.py                 submission 1
final_v4.py              submission 2
src/features.py          feature builder for submission 3
src/make_submission_ens.py   submission 3
requirements.txt         numpy, pandas, scikit-learn
```

## How to run

Put `train.jsonl`, `test.jsonl` and `type_chart_gen2.csv` in one folder (`DATA`), then:

```bash
pip install -r requirements.txt
python final.py DATA submission_ensemble.csv
python final_v4.py DATA submission_v4.csv          # add --eval for CV + holdout
python src/make_submission_ens.py --data DATA --out OUT --cv
```

The Kaggle notebook clones this repository and runs exactly these commands.

## Validation and leakage control

- A stratified 20% holdout of `train.jsonl` (random_state=123) is never used for feature or model selection; features and hyper-parameters are chosen with stratified 5-fold CV on the other 80%.
- The name-to-stats lookup and the composition vocabulary are fitted on training data only.
- Feature selection (permutation importance) uses the development split only.

## Reproducibility note

All seeds are fixed and the scripts run from the raw competition files. Regenerating on Kaggle agrees with our originally submitted files on about 99% of the 5,000 test battles (roughly 50 predictions differ). The differing predictions are borderline cases (for v4, within 0.02 of 50% win probability), which we attribute to library-version differences.
