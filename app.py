"""
Student Outcome Predictor - Streamlit app for ML Assignment 2.

Upload the test CSV, pick one of the five trained models and see how well
it predicts whether a student drops out, stays enrolled or graduates.
"""
from pathlib import Path

import altair as alt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, recall_score

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


def ranked_bar_chart(series, metric):
    """Horizontal bars, best at the top, with the exact value printed on each bar."""
    df = series.rename(metric).rename_axis("Model").reset_index()
    order = df.sort_values(metric, ascending=False)["Model"].tolist()
    base = alt.Chart(df).encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=0)),
        x=alt.X(f"{metric}:Q", title=metric, scale=alt.Scale(domain=[0, 1])),
    )
    bars = base.mark_bar(color="#4c78a8")
    labels = base.mark_text(align="left", dx=4).encode(
        text=alt.Text(f"{metric}:Q", format=".4f")
    )
    return (bars + labels).properties(height=230)


models = load_models()

# ---------------- Header ----------------
st.title("🎓 Student Outcome Predictor")
st.write(
    "Will a student **drop out**, **stay enrolled** or **graduate**? "
    "This app runs five classic machine learning models on student records "
    "(background, finances and first-year grades) and shows how accurate each one is."
)

# ---------------- Controls (on the main page so they are always visible) ----------------
with st.container(border=True):
    c_upload, c_model = st.columns([3, 2])
    with c_upload:
        st.markdown("**Step 1. Upload test data (CSV)**")
        uploaded = st.file_uploader(
            "Upload CSV", type=["csv"], label_visibility="collapsed",
            help="Use the same columns as test_data.csv. Keep the 'Target' column to see scores.",
        )
        st.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    with c_model:
        st.markdown("**Step 2. Choose a classifier**")
        model_name = st.selectbox("Choose a classifier", list(models.keys()),
                                  index=len(models) - 1, label_visibility="collapsed")
        st.caption("All models were trained on 80% of the data. The bundled test set "
                   "is the other 20%, which the models never saw.")

if uploaded is not None:
    data = read_upload(uploaded)
    from_upload = True
else:
    data = load_csv(SAMPLE_CSV)
    from_upload = False

model = models[model_name]
needed = list(model.feature_names_in_)
missing = [c for c in needed if c not in data.columns]
if missing:
    st.error(f"The file is missing {len(missing)} required column(s): "
             + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    st.stop()

has_labels = TARGET in data.columns
if has_labels:
    data = data[data[TARGET].isin(CLASS_ORDER)].reset_index(drop=True)
X = data[needed]

if from_upload:
    st.success(f"Using your file **{uploaded.name}**: {len(X)} students.")
else:
    st.info(f"No file uploaded yet, so the bundled **test_data.csv** is being used "
            f"({len(X)} students). Upload your own CSV above to replace it.")

with st.expander("Peek at the data"):
    st.dataframe(data.head(20), width="stretch")

tab_one, tab_all, tab_pred = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "🔮 Predictions"]
)

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
            col.metric(k, f"{v:.4f}")
        st.caption("Precision, Recall and F1 are macro averages, so the small "
                   "'Enrolled' group counts as much as the others. AUC is one-vs-rest.")

        left, right = st.columns([1, 1.2])
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(plot_confusion(y_true, y_pred, model_name))
            st.caption("How to read it: each **row** is the students' actual outcome and "
                       "each **column** is what the model predicted. The **diagonal** "
                       "(top-left to bottom-right) counts correct predictions; every other "
                       "cell is a mistake.")
        with right:
            st.markdown("**Classification report**")
            report = classification_report(y_true, y_pred, labels=CLASS_ORDER,
                                           output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
            weakest = min(CLASS_ORDER, key=lambda c: report[c]["recall"])
            st.info(f"This model finds it hardest to recognise **{weakest}** students "
                    f"(recall {report[weakest]['recall']:.2f}).")

# ---------------- Tab 2: all models ----------------
with tab_all:
    if not has_labels:
        st.warning("Add a 'Target' column to compare models.")
    else:
        y_true = data[TARGET]
        rows, class_recall = [], {}
        for name, mdl in models.items():
            p = mdl.predict(X)
            rows.append({"Model": name,
                         **score_all(y_true, p, mdl.predict_proba(X), list(mdl.classes_))})
            class_recall[name] = recall_score(y_true, p, labels=CLASS_ORDER, average=None)
        table = pd.DataFrame(rows).set_index("Model")

        st.subheader("All five models on this data")
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(
                axis=0, props="font-weight: bold; background-color: #c7e9c0; color: #111111;"
            ),
            width="stretch",
        )
        st.caption("The best score in each column is shown in **bold** on a green background.")
        best = table["F1"].idxmax()
        st.success(f"Best macro F1 on this data: **{best}** ({table.loc[best, 'F1']:.4f})")

        metric = st.radio("Metric to chart", list(table.columns), horizontal=True, index=4)
        st.altair_chart(ranked_bar_chart(table[metric], metric), width="stretch")

        recall_df = pd.DataFrame(class_recall, index=CLASS_ORDER).T
        hardest = recall_df.mean().idxmin()
        st.markdown("**Recall for each class** (share of each group the model finds)")
        st.dataframe(recall_df.style.format("{:.3f}"), width="stretch")
        st.info(f"Across all five models, **{hardest}** students are the hardest to recognise "
                f"(average recall {recall_df[hardest].mean():.2f}). They are partly passing and "
                f"partly failing, so they look like a mix of the other two groups.")

# ---------------- Tab 3: predictions ----------------
with tab_pred:
    out = data.drop(columns=[TARGET], errors="ignore").copy()
    prob_cols = {f"P({cls})": y_proba[:, i].round(3) for i, cls in enumerate(model.classes_)}
    for i, (col, vals) in enumerate(prob_cols.items()):
        out.insert(i, col, vals)
    out.insert(0, "Predicted", y_pred)
    if has_labels:
        out.insert(0, "Actual", data[TARGET].values)
        out.insert(2, "Correct?", out["Actual"] == out["Predicted"])

    st.subheader(f"What {model_name} predicts")
    counts = pd.Series(y_pred).value_counts().reindex(CLASS_ORDER, fill_value=0)
    c1, c2, c3 = st.columns(3)
    for col, cls in zip((c1, c2, c3), CLASS_ORDER):
        col.metric(f"Predicted {cls}", int(counts[cls]))

    filters = ["All students", "Only students predicted to drop out"]
    if has_labels:
        filters.append("Only wrong predictions")
    choice = st.radio("Show", filters, horizontal=True)
    if choice == filters[1]:
        view = out[out["Predicted"] == "Dropout"]
    elif has_labels and choice == filters[2]:
        view = out[~out["Correct?"]]
    else:
        view = out
    st.caption(f"Showing {len(view)} of {len(out)} students.")
    st.dataframe(view, width="stretch", height=380)
    st.download_button("Download these predictions as CSV", view.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")
