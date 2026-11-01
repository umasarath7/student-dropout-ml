"""
Student Outcome Predictor - Streamlit app for ML Assignment 2.

Upload the test CSV, pick one of the five trained models and see how well
it predicts whether a student drops out, stays enrolled or graduates.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model import decision_tree, knn, logistic_regression, naive_bayes, svm
from model.features import CLASS_ORDER, TARGET, load_csv
from model.metrics import score_all

ROOT = Path(__file__).resolve().parent
SAVED = ROOT / "model" / "saved"
SAMPLE_CSV = ROOT / "test_data.csv"
MODEL_MODULES = [logistic_regression, decision_tree, knn, naive_bayes, svm]

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_models():
    return {m.NAME: joblib.load(SAVED / m.FILE) for m in MODEL_MODULES}


@st.cache_data
def read_upload(file):
    return load_csv(file)


def expected_columns(model):
    return list(model.feature_names_in_)


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


models = load_models()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"],
                                help="Use the same columns as test_data.csv. "
                                     "Keep the 'Target' column to see scores.")
    st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                       file_name="test_data.csv", mime="text/csv")
    use_sample = st.checkbox("Or use the bundled test set", value=uploaded is None)

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()),
                              index=len(models) - 1)

    st.caption("All models were trained on 80% of the data. The bundled "
               "test set is the other 20% that the models never saw.")

# ---------------- Header ----------------
st.title("🎓 Student Outcome Predictor")
st.write(
    "Will a student **drop out**, **stay enrolled** or **graduate**? "
    "This app runs five classic machine learning models on student records "
    "(background, finances and first-year grades) and shows how accurate each one is."
)

if uploaded is not None:
    data = read_upload(uploaded)
    source = uploaded.name
elif use_sample:
    data = load_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("Upload a CSV in the sidebar, or tick **use the bundled test set**.")
    st.stop()

needed = expected_columns(models[model_name])
missing = [c for c in needed if c not in data.columns]
if missing:
    st.error(f"The file is missing {len(missing)} required column(s): "
             + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    st.stop()

has_labels = TARGET in data.columns
if has_labels:
    data = data[data[TARGET].isin(CLASS_ORDER)]
X = data[needed]

st.caption(f"Data source: **{source}** · {len(X)} students · {len(needed)} input columns")
with st.expander("Peek at the data"):
    st.dataframe(data.head(20), width="stretch")

tab_one, tab_all, tab_pred = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "🔮 Predictions"]
)

model = models[model_name]
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

# ---------------- Tab 1: one model ----------------
with tab_one:
    if not has_labels:
        st.warning("No 'Target' column found, so scores can't be calculated. "
                   "See the Predictions tab for the model's guesses.")
    else:
        y_true = data[TARGET]
        scores = score_all(y_true, y_pred, y_proba, list(model.classes_))
        st.subheader(f"{model_name}: evaluation metrics")
        cols = st.columns(6)
        for col, (k, v) in zip(cols, scores.items()):
            col.metric(k, f"{v:.3f}")
        st.caption("Precision, Recall and F1 are macro averages, so the small "
                   "'Enrolled' group counts as much as the others. AUC is one-vs-rest.")

        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(plot_confusion(y_true, y_pred, model_name))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(y_true, y_pred, labels=CLASS_ORDER,
                                           output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")
            weakest = min(CLASS_ORDER, key=lambda c: report[c]["recall"])
            st.info(f"This model finds it hardest to recognise **{weakest}** students "
                    f"(recall {report[weakest]['recall']:.2f}).")

# ---------------- Tab 2: all models ----------------
with tab_all:
    if not has_labels:
        st.warning("Add a 'Target' column to compare models.")
    else:
        y_true = data[TARGET]
        rows = []
        for name, mdl in models.items():
            p = mdl.predict(X)
            rows.append({"Model": name,
                         **score_all(y_true, p, mdl.predict_proba(X), list(mdl.classes_))})
        table = pd.DataFrame(rows).set_index("Model")
        st.subheader("All five models on this data")
        st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c7e9c0"),
                     width="stretch")
        best = table["F1"].idxmax()
        st.success(f"Best macro F1 on this data: **{best}** ({table.loc[best, 'F1']:.4f})")

        metric = st.radio("Metric to chart", list(table.columns), horizontal=True, index=4)
        st.bar_chart(table[metric].sort_values(), horizontal=True)

# ---------------- Tab 3: predictions ----------------
with tab_pred:
    out = data.copy()
    out.insert(0, "Predicted", y_pred)
    for i, cls in enumerate(model.classes_):
        out.insert(1 + i, f"P({cls})", y_proba[:, i].round(3))
    if has_labels:
        out.insert(0, "Correct?", out["Predicted"] == out[TARGET])

    st.subheader(f"What {model_name} predicts")
    counts = pd.Series(y_pred).value_counts().reindex(CLASS_ORDER, fill_value=0)
    c1, c2, c3 = st.columns(3)
    for col, cls in zip((c1, c2, c3), CLASS_ORDER):
        col.metric(f"Predicted {cls}", int(counts[cls]))

    show_risk = st.checkbox("Show only students predicted to drop out")
    view = out[out["Predicted"] == "Dropout"] if show_risk else out
    st.dataframe(view, width="stretch", height=380)
    st.download_button("Download predictions as CSV", view.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")
