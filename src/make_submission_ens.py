"""Reproduce `submission_ens.csv`: soft-voting average of HistGradientBoosting,
logistic regression and random forest on the engineered features in features.py.
All seeds fixed (random_state=42) so the output is reproducible.

Usage: python src/make_submission_ens.py --data DATA_DIR --out OUT_DIR [--cv]
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def models():
    return {
        "hgb": HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
                                              l2_regularization=2.0, min_samples_leaf=40, random_state=42),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000)),
        "rf": RandomForestClassifier(800, min_samples_leaf=5, max_features=0.3, n_jobs=-1, random_state=42),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train.jsonl, test.jsonl, type_chart_gen2.csv")
    ap.add_argument("--out", default=".", help="where to write submission_ens.csv")
    ap.add_argument("--cv", action="store_true", help="also print 5-fold CV accuracy of the ensemble")
    a = ap.parse_args()
    chart = os.path.join(a.data, "type_chart_gen2.csv")
    X, y, _ = build(os.path.join(a.data, "train.jsonl"), chart)
    Xt, _, ids = build(os.path.join(a.data, "test.jsonl"), chart, with_target=False)
    Xt = Xt[X.columns]
    if a.cv:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        P = [cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1] for m in models().values()]
        print("5-fold CV accuracy (ensemble):", round((((sum(P) / 3) > .5) == y).mean(), 4))
    p = sum(m.fit(X, y).predict_proba(Xt)[:, 1] for m in models().values()) / 3
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame({"battle_id": ids, "player_won": (p > .5).astype(int)}).to_csv(
        os.path.join(a.out, "submission_ens.csv"), index=False)
    print("wrote", os.path.join(a.out, "submission_ens.csv"))

if __name__ == "__main__":
    main()
