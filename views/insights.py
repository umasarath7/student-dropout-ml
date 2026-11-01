"""Insights: findings and suggested actions generated from the live results."""
import numpy as np
import streamlit as st

from model.features import CLASS_ORDER
from ui import components as ui
from ui import services as svc


def render():
    ss = st.session_state
    ui.page_header("Insights", "Plain-language findings from the current results", "lightbulb")
    ui.show_flash()
    c1, _ = st.columns([1, 2])
    with c1:
        ui.model_picker("ins_model_select")
    name = ss.selected_model
    ev = svc.current_evaluation()
    preds = svc.predictions_frame(ss.dataset, ev, name)
    ui.note("<b>Findings</b> are measured facts about this dataset. <b>Suggested actions</b> are ideas to "
            "test, not guaranteed improvements. Everything below is recalculated whenever you change the "
            "model or dataset.")

    if not ev["has_labels"]:
        ui.section("Prediction insights", "", "users")
        _risk_insight(preds, ev, name)
        ui.empty_state("Upload a dataset with a 'Target' column to unlock performance insights.")
        return

    res = ev["models"][name]
    m = res["metrics"]
    table = svc.comparison_table(ev)
    y = ev["y"]
    majority_cls = max(CLASS_ORDER, key=lambda c: (y == c).sum())
    baseline = (y == majority_cls).mean()

    ui.section("Model performance", "", "gauge")
    ui.insight_card(
        "finding", "Overall model performance",
        f"{name} predicts <b>{m['Accuracy']:.2%}</b> of students correctly, with an AUC of "
        f"<b>{_fmt(m['AUC'])}</b>, macro F1 of <b>{m['F1']:.4f}</b> and MCC of <b>{m['MCC']:.4f}</b>. "
        f"Always guessing the most common outcome ({majority_cls}) would score {baseline:.2%} accuracy, "
        f"so the model adds <b>{(m['Accuracy'] - baseline) * 100:.1f} percentage points</b>.",
        "A baseline shows whether the model learned anything useful beyond the class balance.")

    rec = res["recall_by_class"]
    weakest = min(rec, key=rec.get)
    support = int(res["report"][weakest]["support"])
    found = int(round(rec[weakest] * support))
    ui.insight_card(
        "finding", f"{weakest} students are the hardest to identify",
        f"Recall for {weakest} is <b>{rec[weakest]:.4f}</b>: the model correctly finds {found:,} of "
        f"{support:,} {weakest} students. For comparison, recall is "
        + ", ".join(f"{c} {v:.4f}" for c, v in rec.items() if c != weakest) + ".",
        "Students the model misses would not be flagged for support, even though their outcome is "
        "the one being predicted.")

    cm = res["cm"]
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    ui.insight_card(
        "finding", "Most common mistake",
        f"The most frequent error is predicting <b>{CLASS_ORDER[j]}</b> for students who were actually "
        f"<b>{CLASS_ORDER[i]}</b>: {int(off[i, j]):,} students ({off[i, j] / cm[i].sum():.1%} of that group).",
        "Knowing which outcomes get confused tells you where the model's picture of students overlaps.")

    ui.insight_card(
        "action", f"Ideas to improve {weakest} detection",
        "Consider testing class weights or oversampling for the smaller groups, adjusting the decision "
        "threshold per outcome, and ensemble methods such as Random Forest or Gradient Boosting. "
        "Validate any change with cross-validation on training data only.",
        "These are standard remedies for imbalanced classes, but whether they help here has to be measured.")

    ui.section("Model comparison", "", "scale")
    best_f1, best_mcc = table["F1"].idxmax(), table["MCC"].idxmax()
    has_auc = table["AUC"].notna().any()
    best_auc = table["AUC"].idxmax() if has_auc else best_f1
    leaders = (f"<b>{best_f1}</b> leads on macro F1 ({table.loc[best_f1, 'F1']:.4f}) and "
               f"<b>{best_mcc}</b> on MCC ({table.loc[best_mcc, 'MCC']:.4f})")
    leaders += (f", while <b>{best_auc}</b> has the best AUC ({table.loc[best_auc, 'AUC']:.4f})."
                if has_auc else ". AUC isn't available because not every outcome appears in this data.")
    spread = table["F1"].max() - table["F1"].min()
    ui.insight_card(
        "finding", "Which model is strongest depends on the metric" if best_f1 != best_auc
        else f"{best_f1} leads across the key metrics",
        leaders + f" The gap between the best and worst macro F1 is {spread:.4f}.",
        "F1 and MCC judge the final labels; AUC judges how well students are ranked by risk. "
        "Pick based on how the predictions will be used.")

    agree = np.all([r["pred"] == ev["models"][name]["pred"] for r in ev["models"].values()], axis=0)
    mask = ev["mask"]
    correct = res["pred"][mask] == y
    a_mask = agree[mask]
    if a_mask.any() and (~a_mask).any():
        ui.insight_card(
            "finding", "Agreement between models signals reliability",
            f"All five models agree on <b>{agree.mean():.1%}</b> of students. Where they agree, {name} "
            f"is right <b>{correct[a_mask].mean():.1%}</b> of the time; where they disagree, only "
            f"<b>{correct[~a_mask].mean():.1%}</b>.",
            "Students the models disagree on are borderline cases that deserve a human look.")
    ui.insight_card(
        "action", "Choosing a production candidate",
        f"Compare {best_f1} and {best_auc if best_auc != best_f1 else table['F1'].drop(best_f1).idxmax()} "
        "on all six metrics and on per-class recall before choosing. Re-check on new intakes, as "
        "results can shift with different students or economic conditions.",
        "A model chosen on one metric or one test set can disappoint on the next cohort.")

    ui.section("Prediction insights", "", "users")
    _risk_insight(preds, ev, name)


def _fmt(v):
    return "n/a" if v != v else f"{v:.4f}"


def _risk_insight(preds, ev, name):
    ss = st.session_state
    high = preds["Dropout risk"] == "High"
    n_high = int(high.sum())
    body = (f"{name} places <b>{n_high:,}</b> students ({n_high / len(preds):.1%}) in the High dropout-risk "
            f"band (P(Dropout) ≥ {ss.risk_high:.0%}).")
    if ev["has_labels"] and n_high:
        labelled = preds[high & preds["Actual"].notna()]
        if len(labelled):
            hit = (labelled["Actual"] == "Dropout").mean()
            body += f" Of those with a known outcome, <b>{hit:.1%}</b> actually dropped out."
    ui.insight_card("finding", "Students flagged as high risk", body,
                    "This is the group an early-warning process would contact first; the hit rate shows "
                    "how many contacts would reach students who really were at risk.")
    ui.insight_card("caution", "Predictions are estimates, not verdicts",
                    "Use the risk bands to prioritise conversations, not to make decisions about "
                    "individual students on their own. The model sees correlations in past data, not causes.")
