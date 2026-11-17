"""v4 submission, reproducible from raw data (no cached files).
Run: python final_v4.py <data_dir> <out_csv> [--eval]
Pipeline: base features (features.py) + extra groups A-E (features2.py) + composition flags,
pruned to the 69 features in selected_features.txt (permutation importance on the dev split only),
soft-voting ensemble of HistGradientBoosting (500 iter) + RandomForest + LogisticRegression.
Leakage control: name->stats lookup and composition vocabulary are fitted on TRAIN ONLY.
--eval prints dev 5-fold CV and the untouched 20% holdout (random_state=123)."""
import sys, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from features import load_jsonl, load_type_chart, build_stat_lookup, make_matrix, composition_features
from features2 import make_extra

args = [a for a in sys.argv[1:] if not a.startswith("--")]
D = (args[0] if args else ".").rstrip("/") + "/"
OUT = args[1] if len(args) > 1 else "submission_v4.csv"
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))

tr, te = load_jsonl(D + "train.jsonl"), load_jsonl(D + "test.jsonl")
chart = load_type_chart(D + "type_chart_gen2.csv")
lut = build_stat_lookup(tr)                                   # train only
y = np.array([b["player_won"] for b in tr])

def features(battles, ctr_cols=None):
    X, ids = make_matrix(battles, chart, lut)
    E = make_extra(battles, chart, lut)
    C = composition_features(battles)
    return X, E, C, ids

Xb, Eb, Cb, _ = features(tr)
keep = sorted(c for c in Cb if (Cb[c] > 0).sum() >= 100)       # vocabulary from train only
Xtr = pd.concat([Xb, Eb, Cb[keep]], axis=1)
Xb2, Eb2, Cb2, ids = features(te)
Xte = pd.concat([Xb2, Eb2, Cb2.reindex(columns=keep, fill_value=0)], axis=1)[Xtr.columns]
SEL = open(os.path.join(HERE, "selected_features.txt")).read().split("\n")
Xtr, Xte = Xtr[SEL], Xte[SEL]

def make_model():
    hgb = HistGradientBoostingClassifier(learning_rate=0.04, max_iter=500, max_leaf_nodes=8,
                                         l2_regularization=2.0, min_samples_leaf=20, random_state=SEED)
    rf = make_pipeline(SimpleImputer(), RandomForestClassifier(300, min_samples_leaf=3, max_features=0.2, random_state=SEED))
    lr = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.02, max_iter=3000))
    return VotingClassifier([("hgb", hgb), ("rf", rf), ("lr", lr)], voting="soft")

if "--eval" in sys.argv:
    idx_dev, idx_ho = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=123)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(make_model(), Xtr.iloc[idx_dev], y[idx_dev], cv=cv, method="predict_proba")[:, 1]
    print(f"dev 5-fold CV acc: {((p > .5) == y[idx_dev]).mean():.4f}")
    m = make_model().fit(Xtr.iloc[idx_dev], y[idx_dev])
    print(f"holdout acc: {(m.predict(Xtr.iloc[idx_ho]) == y[idx_ho]).mean():.4f}")

pred = make_model().fit(Xtr, y).predict(Xte).astype(int)
pd.DataFrame({"battle_id": ids, "player_won": pred}).to_csv(OUT, index=False)
print("written", OUT, len(pred), "mean", pred.mean())
