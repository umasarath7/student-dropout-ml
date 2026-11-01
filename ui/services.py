"""
Central data handling for every EduPulse page.

The ML backend in model/ is used as-is: the five saved pipelines are loaded
once, and every number the app shows is computed here from their real
predictions on the current dataset. Nothing is hardcoded.
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)

from model import decision_tree, knn, logistic_regression, naive_bayes, svm
from model.features import CLASS_ORDER, TARGET, add_study_features, load_csv
from model.metrics import AVERAGE, score_all

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "model" / "saved"
SAMPLE_CSV = ROOT / "test_data.csv"
FULL_CSV = ROOT / "data" / "students.csv"
MODEL_MODULES = [logistic_regression, decision_tree, knn, naive_bayes, svm]
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
ID_COLUMNS = ["Student ID", "student_id", "StudentID", "ID", "id"]


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner=False)
def load_models():
    return {m.NAME: joblib.load(SAVED / m.FILE) for m in MODEL_MODULES}


def model_names():
    return list(load_models().keys())


@st.cache_resource(show_spinner=False)
def required_columns():
    return list(next(iter(load_models().values())).feature_names_in_)


@st.cache_data(show_spinner=False)
def tuned_params():
    path = SAVED / "best_params.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def split_summary():
    """How the original data was divided when the models were trained."""
    total = len(load_csv(FULL_CSV))
    test = len(load_csv(SAMPLE_CSV))
    train = total - test
    return {"total": total, "train": train, "test": test,
            "train_pct": round(100 * train / total), "test_pct": round(100 * test / total)}


@st.cache_data(show_spinner=False)
def bundled_dataset():
    return load_csv(SAMPLE_CSV)


# ---------------------------------------------------------------- session state
def init_state():
    ss = st.session_state
    if "dataset" not in ss:
        ss.dataset = bundled_dataset()
        ss.dataset_name = SAMPLE_CSV.name
        ss.dataset_source = "bundled"
        ss.dataset_loaded_at = datetime.now()
    ss.setdefault("selected_model", model_names()[-1])
    ss.setdefault("risk_high", 0.50)
    ss.setdefault("risk_medium", 0.25)
    ss.setdefault("flash", None)


def set_dataset(df, name, source):
    ss = st.session_state
    ss.dataset = df.reset_index(drop=True)
    ss.dataset_name = name
    ss.dataset_source = source
    ss.dataset_loaded_at = datetime.now()


def reset_dataset():
    set_dataset(bundled_dataset(), SAMPLE_CSV.name, "bundled")
    set_flash("info", "Switched back to the bundled test set.")


def set_flash(kind, message, details=None):
    st.session_state.flash = {"kind": kind, "message": message, "details": details or []}


# ---------------------------------------------------------------- validation
def validate_dataset(df):
    """Return (errors, warnings). Errors block the upload; warnings don't."""
    errors, warnings = [], []
    if df.empty:
        return ["The file has a header but no student rows."], warnings

    needed = required_columns()
    missing = [c for c in needed if c not in df.columns]
    if missing:
        shown = ", ".join(f"'{c}'" for c in missing[:6])
        more = f" and {len(missing) - 6} more" if len(missing) > 6 else ""
        errors.append(f"Missing {len(missing)} required column(s): {shown}{more}.")
        return errors, warnings

    non_numeric = [c for c in needed if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append("These columns must contain numbers only: "
                      + ", ".join(f"'{c}'" for c in non_numeric[:6]) + ".")
    blanks = df[needed].isna().sum()
    blanks = blanks[blanks > 0]
    if len(blanks):
        errors.append("Some required columns have empty cells: "
                      + ", ".join(f"'{c}' ({n})" for c, n in blanks.head(6).items()) + ".")

    if TARGET in df.columns:
        unknown = (~df[TARGET].isin(CLASS_ORDER)).sum()
        if unknown == len(df):
            warnings.append("The 'Target' column has no Dropout/Enrolled/Graduate values, "
                            "so predictions will work but scores can't be calculated.")
        elif unknown:
            warnings.append(f"{unknown} row(s) have an unknown 'Target' value. They get "
                            "predictions but are left out of the scores.")
    else:
        warnings.append("No 'Target' column, so the app can predict but can't score the models.")

    extra = [c for c in df.columns if c not in needed and c != TARGET and c not in ID_COLUMNS]
    if extra:
        warnings.append(f"{len(extra)} extra column(s) will be ignored.")
    return errors, warnings


def student_ids(df):
    for col in ID_COLUMNS:
        if col in df.columns:
            return df[col].astype(str).to_numpy()
    return np.array([f"STU-{i + 1:04d}" for i in range(len(df))])


# ---------------------------------------------------------------- evaluation
def _scores(y, y_pred, proba):
    try:
        return score_all(y, y_pred, proba, CLASS_ORDER)
    except ValueError:
        # AUC needs every class present in the labels; score the rest anyway.
        return {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC": float("nan"),
            "Precision": precision_score(y, y_pred, average=AVERAGE, zero_division=0),
            "Recall": recall_score(y, y_pred, average=AVERAGE, zero_division=0),
            "F1": f1_score(y, y_pred, average=AVERAGE, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred),
        }


@st.cache_data(show_spinner="Running all five models on the dataset...")
def evaluate(df):
    """Predict with every saved model and score them when labels exist."""
    X = df[required_columns()]
    mask = (df[TARGET].isin(CLASS_ORDER).to_numpy() if TARGET in df.columns
            else np.zeros(len(df), dtype=bool))
    has_labels = bool(mask.any())
    y = df.loc[mask, TARGET].to_numpy() if has_labels else None

    results = {}
    for name, mdl in load_models().items():
        pred = mdl.predict(X)
        order = [list(mdl.classes_).index(c) for c in CLASS_ORDER]
        proba = mdl.predict_proba(X)[:, order]
        res = {"pred": pred, "proba": proba}
        if has_labels:
            yp, pp = pred[mask], proba[mask]
            res["metrics"] = _scores(y, yp, pp)
            res["report"] = classification_report(y, yp, labels=CLASS_ORDER,
                                                  output_dict=True, zero_division=0)
            res["cm"] = confusion_matrix(y, yp, labels=CLASS_ORDER)
            res["recall_by_class"] = dict(zip(CLASS_ORDER, recall_score(
                y, yp, labels=CLASS_ORDER, average=None, zero_division=0)))
            roc = {}
            for i, cls in enumerate(CLASS_ORDER):
                positives = (y == cls).astype(int)
                if 0 < positives.sum() < len(positives):
                    fpr, tpr, _ = roc_curve(positives, pp[:, i])
                    roc[cls] = (fpr, tpr)
            res["roc"] = roc
        results[name] = res
    return {"has_labels": has_labels, "mask": mask, "y": y, "n": len(df),
            "n_labeled": int(mask.sum()), "models": results}


def current_evaluation():
    return evaluate(st.session_state.dataset)


def comparison_table(ev):
    rows = {name: r["metrics"] for name, r in ev["models"].items() if "metrics" in r}
    return pd.DataFrame(rows).T[METRICS] if rows else pd.DataFrame(columns=METRICS)


def ranked_model_names(ev=None):
    """Models in order of preference: best macro F1 first (ties broken by MCC).

    Falls back to the training order when the data has no labels to rank on.
    """
    ev = ev or current_evaluation()
    table = comparison_table(ev)
    if table.empty:
        return model_names()
    return table.sort_values(["F1", "MCC"], ascending=False).index.tolist()


def report_table(report):
    rows = []
    for key, label in [*[(c, c) for c in CLASS_ORDER],
                       ("macro avg", "Macro average"), ("weighted avg", "Weighted average")]:
        r = report[key]
        rows.append({"Class": label, "Precision": r["precision"], "Recall": r["recall"],
                     "F1 Score": r["f1-score"], "Support": int(r["support"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- predictions
def risk_labels(p_dropout, high, medium):
    return np.select([p_dropout >= high, p_dropout >= medium], ["High", "Medium"], "Low")


def predictions_frame(df, ev, model_name):
    ss = st.session_state
    res = ev["models"][model_name]
    proba = res["proba"]
    out = pd.DataFrame({
        "Student ID": student_ids(df),
        "Predicted": res["pred"],
        "Confidence": proba.max(axis=1),
        "P(Dropout)": proba[:, 0],
        "P(Enrolled)": proba[:, 1],
        "P(Graduate)": proba[:, 2],
        "Dropout risk": risk_labels(proba[:, 0], ss.risk_high, ss.risk_medium),
    })
    if ev["has_labels"]:
        actual = np.where(ev["mask"], df[TARGET].astype(str).to_numpy()
                          if TARGET in df.columns else "", None)
        out["Actual"] = actual
        out["Correct?"] = np.where(ev["mask"], out["Predicted"] == out["Actual"], None)
    return out


KEY_FEATURES = [
    "Age at enrollment",
    "Scholarship holder",
    "Tuition fees up to date",
    "Debtor",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Pass rate 1st sem",
    "Pass rate 2nd sem",
]


@st.cache_data(show_spinner=False)
def enriched(df):
    """Raw columns plus the engineered features the models use internally."""
    return add_study_features(df[required_columns()])


# ---------------------------------------------------------------- diagnostics
def readable_feature_names(mdl):
    names = mdl.named_steps["prep"].named_steps["encode"].get_feature_names_out()
    clean = []
    for n in names:
        n = n.replace("numbers__", "")
        if n.startswith("codes__"):
            col, _, code = n.replace("codes__", "").rpartition("_")
            code = "rare codes" if code == "sklearn" else code
            col = col.replace("_infrequent", "")
            n = f"{col} = {code}"
        clean.append(n)
    return clean


def model_diagnostics(model_name):
    """Model-specific facts read straight from the fitted estimator."""
    mdl = load_models()[model_name]
    clf = mdl.named_steps["clf"]
    names = readable_feature_names(mdl)
    out = {"n_features": len(names)}
    if hasattr(clf, "feature_importances_"):
        out["importance"] = pd.Series(clf.feature_importances_, index=names)
        out["depth"] = clf.get_depth()
        out["leaves"] = clf.get_n_leaves()
    if hasattr(clf, "coef_"):
        out["coef"] = pd.DataFrame(clf.coef_, index=list(clf.classes_), columns=names)
    if hasattr(clf, "n_neighbors"):
        out["neighbors"] = clf.n_neighbors
        out["weights"] = clf.weights
        out["n_train"] = clf.n_samples_fit_
    if hasattr(clf, "var_smoothing"):
        out["var_smoothing"] = clf.var_smoothing
        out["class_prior"] = dict(zip(clf.classes_, clf.class_prior_))
    if hasattr(clf, "calibrated_classifiers_"):
        svc = clf.calibrated_classifiers_[0].estimator
        out["support_vectors"] = dict(zip(svc.classes_, svc.n_support_))
    return out
