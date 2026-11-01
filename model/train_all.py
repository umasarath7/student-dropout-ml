"""
Train all five models, save them, and write the test split + results.

Run from the project root:
    python -m model.train_all

Outputs
- model/saved/<model>.joblib   fitted pipelines (preprocessing + classifier)
- test_data.csv                the 20% hold-out set used for every model
- model/saved/results.csv      the comparison table shown in the README
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from model import decision_tree, knn, logistic_regression, naive_bayes, svm
from model.features import TARGET, build_preprocessor, feature_columns, load_csv
from model.metrics import score_all

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "students.csv"
SAVE_DIR = ROOT / "model" / "saved"
SEED = 26

MODELS = [logistic_regression, decision_tree, knn, naive_bayes, svm]


def main():
    df = load_csv(DATA)
    X, y = df[feature_columns(df)], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"train rows: {len(X_train)}, test rows: {len(X_test)}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}

    for m in MODELS:
        pipe = Pipeline([("prep", build_preprocessor(list(X.columns))),
                         ("clf", m.make_estimator())])
        search = GridSearchCV(pipe, m.PARAM_GRID, scoring="f1_macro", cv=folds, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)
        scores = score_all(y_test, y_pred, y_proba, list(best.classes_))
        rows.append({"ML Model Name": m.NAME, **scores})
        best_params[m.NAME] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        best_params[m.NAME] = {k.replace("estimator__", ""): v for k, v in best_params[m.NAME].items()}

        joblib.dump(best, SAVE_DIR / m.FILE)
        print(f"{m.NAME:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    results = pd.DataFrame(rows)
    results.to_csv(SAVE_DIR / "results.csv", index=False)
    (SAVE_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    print("\n" + results.round(4).to_string(index=False))
    print(json.dumps(best_params, indent=2, default=str))


if __name__ == "__main__":
    main()
