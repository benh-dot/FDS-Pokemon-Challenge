"""Reproduces the final submission. Run: python final.py <data_dir> <out_csv>

Leakage control: the name->stats lookup and the composition vocabulary are fitted on TRAIN ONLY.
The model is selected/tuned with CV on a development split; a 20% holdout that was never used
for any decision gives the final unbiased estimate.
"""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from features import load_jsonl, load_type_chart, build_stat_lookup, make_matrix, composition_features

D = (sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads").rstrip("/") + "/"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs/submission_ensemble.csv"
SEED = 0

tr, te = load_jsonl(D + "train.jsonl"), load_jsonl(D + "test.jsonl")
chart = load_type_chart(D + "type_chart_gen2.csv")
lut = build_stat_lookup(tr)                      # train only
y = np.array([b["player_won"] for b in tr])

Xtr, _ = make_matrix(tr, chart, lut)
Xte, ids = make_matrix(te, chart, lut)
Ctr = composition_features(tr)
keep = sorted(c for c in Ctr if (Ctr[c] > 0).sum() >= 100)   # vocabulary from train only
Cte = composition_features(te).reindex(columns=keep, fill_value=0)
Xtr = pd.concat([Xtr, Ctr[keep]], axis=1)
Xte = pd.concat([Xte, Cte], axis=1)[Xtr.columns]


def make_model():
    hgb = HistGradientBoostingClassifier(learning_rate=0.04, max_iter=300, max_leaf_nodes=8,
                                         l2_regularization=2.0, random_state=SEED)
    rf = make_pipeline(SimpleImputer(), RandomForestClassifier(500, min_samples_leaf=3, random_state=SEED))
    lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.02, max_iter=3000))
    return VotingClassifier([("hgb", hgb), ("rf", rf), ("lr", lr)], voting="soft")


# 1) never-touched holdout for an unbiased estimate
idx_dev, idx_ho = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=123)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
p = cross_val_predict(make_model(), Xtr.iloc[idx_dev], y[idx_dev], cv=cv, method="predict_proba")[:, 1]
print(f"dev 5-fold CV acc: {((p > .5) == y[idx_dev]).mean():.4f}")
m = make_model().fit(Xtr.iloc[idx_dev], y[idx_dev])
print(f"holdout acc (20%, untouched): {(m.predict(Xtr.iloc[idx_ho]) == y[idx_ho]).mean():.4f}")

# 2) refit on all training data, predict test
final = make_model().fit(Xtr, y)
pred = final.predict(Xte).astype(int)
pd.DataFrame({"battle_id": ids, "player_won": pred}).to_csv(OUT, index=False)
print("written", OUT, len(pred), "mean", pred.mean())
