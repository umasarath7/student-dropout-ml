"""Model Evaluation: a full report on one classifier."""
from html import escape

import pandas as pd
import streamlit as st

from ui import charts
from ui import components as ui
from ui import services as svc
from views.shared import comparison_block, confusion_and_report

DESCRIPTIONS = {
    "Logistic Regression": "A linear model that learns one weighted score per outcome and turns the scores into probabilities.",
    "Decision Tree": "A flowchart of yes/no questions on the features, learned from the training data.",
    "kNN": "Predicts the outcome most common among the most similar students in the training data.",
    "Naive Bayes": "A probabilistic model (Gaussian) that assumes features are independent given the outcome.",
    "Support Vector Machine": "Finds curved boundaries (RBF kernel) that separate the outcomes; probabilities come from a calibration step.",
}


def render():
    ss = st.session_state
    ui.page_header("Model Evaluation", "A complete report on the selected classifier", "gauge")
    ui.show_flash()
    c1, _ = st.columns([1, 2])
    with c1:
        ui.model_picker("eval_model_select")
    name = ss.selected_model
    ev = svc.current_evaluation()
    split = svc.split_summary()
    params = svc.tuned_params().get(name, {})

    left, right = st.columns(2, gap="medium")
    with left:
        ui.section("Model configuration", "", "settings")
        rows = "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in params.items())
        ui.html(f"""<div class="ep-card"><p class="ep-note" style="margin:0 0 10px">{escape(DESCRIPTIONS[name])}</p>
        <dl class="ep-kv"><dt>Algorithm</dt><dd>{escape(name)}</dd>{rows}
        <dt>Tuning</dt><dd>5-fold cross-validation grid search, scored on macro F1</dd>
        <dt>Preprocessing</dt><dd>4 engineered features · one-hot codes · standard scaling</dd></dl></div>""")
    with right:
        ui.section("Data and split", "", "split")
        ui.html(f"""<div class="ep-card"><dl class="ep-kv">
        <dt>Original dataset</dt><dd>{split['total']:,} students</dd>
        <dt>Training set</dt><dd>{split['train']:,} students ({split['train_pct']}%)</dd>
        <dt>Hold-out test set</dt><dd>{split['test']:,} students ({split['test_pct']}%), stratified</dd>
        <dt>Evaluating on</dt><dd>{escape(ss.dataset_name)} · {ev['n']:,} rows · {ev['n_labeled']:,} labelled</dd>
        </dl></div>""")

    if not ev["has_labels"]:
        ui.section("Evaluation metrics", "", "gauge")
        ui.empty_state("This dataset has no 'Target' column, so the model can't be scored. "
                       "Upload a file with actual outcomes, or switch back to the bundled test set.")
        return
    res = ev["models"][name]

    ui.section("Evaluation metrics", f"{name} on {ev['n_labeled']:,} labelled students", "gauge")
    ui.metric_kpis(res["metrics"])

    confusion_and_report(res, name)

    ui.section("ROC curves", "One-vs-rest: each outcome against the other two", "activity")
    if res["roc"]:
        charts.show(charts.roc(res["roc"]), f"ROC curves per outcome for {name}.")
        ui.note(f"Each line shows how many students of that outcome are caught (true positive rate) "
                f"for a given false-alarm rate. The dashed diagonal is random guessing. "
                f"Macro AUC: <b>{res['metrics']['AUC']:.4f}</b>." if res['metrics']['AUC'] == res['metrics']['AUC'] else "")
    else:
        ui.empty_state("ROC curves need students from more than one outcome in the data.")

    ui.section("Model comparison", "How the selected model stacks up", "scale")
    comparison_block(ev, "eval_metric")

    ui.section("Diagnostics", "Facts read directly from the fitted model", "search")
    recall = pd.DataFrame({m: r["recall_by_class"] for m, r in ev["models"].items()}).T
    st.markdown("**Recall for each outcome, all models**")
    st.dataframe(recall, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in recall.columns})
    _model_specific(name)


def _model_specific(name):
    d = svc.model_diagnostics(name)
    st.markdown(f"**{name}: model-specific details**")
    if "importance" in d:
        imp = d["importance"].sort_values(ascending=False).head(10)
        ui.note(f"Tree depth {d['depth']}, {d['leaves']} leaves. Importance = how much each feature "
                "reduced uncertainty across the tree's splits (global, not per student).")
        charts.show(charts.ranked_bars(imp, "Importance"), "Top 10 decision tree feature importances.")
    elif "coef" in d:
        cls = st.segmented_control("Outcome", list(d["coef"].index), default="Dropout", key="coef_cls") or "Dropout"
        top = d["coef"].loc[cls]
        top = top.reindex(top.abs().sort_values(ascending=False).index).head(10)
        ui.note("Largest coefficients on standardised features. A positive weight means higher values "
                f"are <b>associated</b> with {cls} in this model; it is not evidence of cause.")
        charts.show(charts.ranked_bars(top, "Coefficient", signed=True),
                    f"Top 10 logistic regression coefficients for {cls}.")
    elif "neighbors" in d:
        ui.html(f"""<div class="ep-card"><dl class="ep-kv"><dt>Neighbours (k)</dt><dd>{d['neighbors']}</dd>
        <dt>Vote weighting</dt><dd>{escape(d['weights'])}</dd><dt>Stored training students</dt><dd>{d['n_train']:,}</dd>
        <dt>Input dimensions</dt><dd>{d['n_features']}</dd></dl></div>""")
        ui.note("kNN has no learned weights. With distance weighting it reproduces the training set "
                "perfectly, so test-set scores are the honest measure.")
    elif "var_smoothing" in d:
        priors = "".join(f"<dt>Prior: {escape(k)}</dt><dd>{v:.3f}</dd>" for k, v in d["class_prior"].items())
        ui.html(f"""<div class="ep-card"><dl class="ep-kv"><dt>var_smoothing</dt><dd>{d['var_smoothing']}</dd>
        {priors}<dt>Input dimensions</dt><dd>{d['n_features']}</dd></dl></div>""")
        ui.note("Priors are the outcome shares in the training data. Large smoothing was needed because "
                "the one-hot columns have tiny variances.")
    elif "support_vectors" in d:
        sv = "".join(f"<dt>Support vectors: {escape(k)}</dt><dd>{v:,}</dd>" for k, v in d["support_vectors"].items())
        ui.html(f'<div class="ep-card"><dl class="ep-kv">{sv}<dt>Input dimensions</dt><dd>{d["n_features"]}</dd></dl></div>')
        ui.note("Support vectors are the training students closest to the boundaries. Many of them in a "
                "class means that class overlaps a lot with the others.")
