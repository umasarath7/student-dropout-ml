"""Blocks used on more than one page."""
import streamlit as st

from ui import charts
from ui import components as ui
from ui import services as svc

METRIC_EXPLAIN = {
    "Accuracy": "Accuracy is the share of all students predicted correctly. It can look good even "
                "if a small group (like Enrolled) is mostly missed, so read it together with F1 and MCC.",
    "AUC": "AUC measures how well a model ranks students by likelihood of each outcome, independent "
           "of any cut-off. Good for comparing risk scores, but it doesn't measure the final labels.",
    "Precision": "Precision asks: when the model names an outcome, how often is it right? Low precision "
                 "means many false alarms.",
    "Recall": "Recall asks: of the students who truly had an outcome, how many did the model find? Low "
              "recall means students are missed.",
    "F1": "F1 balances precision and recall. As a macro average it treats all three outcomes equally, "
          "so it is a fair single score for this imbalanced data.",
    "MCC": "MCC uses the whole confusion matrix and stays honest when classes are imbalanced. 0 means "
           "no better than guessing, 1 means perfect.",
}


def comparison_block(ev, key):
    table = svc.comparison_table(ev)
    if table.empty:
        ui.empty_state("Model comparison needs a 'Target' column with actual outcomes.")
        return
    metric = st.segmented_control("Metric", svc.METRICS, default="F1", key=key,
                                  label_visibility="collapsed") or "F1"
    if table[metric].isna().all():
        ui.empty_state(f"{metric} can't be calculated because not every outcome appears in this data.")
        return
    charts.show(charts.comparison(table, metric, st.session_state.selected_model),
                f"Bar chart ranking the five models by {metric}.")
    best = table[metric].idxmax()
    ui.note(f"<b>{best}</b> leads on {metric} with {table.loc[best, metric]:.4f}. "
            f"{METRIC_EXPLAIN[metric]} The highest bar on one metric isn't automatically the best "
            "model, so check the others too.")


def confusion_and_report(res, model_name):
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        ui.section("Confusion matrix", "Actual outcomes vs predictions", "columns")
        charts.show(charts.confusion(res["cm"]),
                    f"Confusion matrix heatmap for {model_name}: rows are actual outcomes, columns are predictions.")
        ui.note("Rows represent <b>actual</b> outcomes and columns represent <b>predicted</b> outcomes. "
                "Diagonal cells (teal) are correct predictions; the other cells (rose) are mistakes. "
                "The percentage is the share of that row.")
    with right:
        ui.section("Classification report", "Per-outcome precision, recall and F1", "file")
        st.dataframe(
            svc.report_table(res["report"]), hide_index=True, width="stretch",
            column_config={
                "Class": st.column_config.TextColumn("Class"),
                **{c: st.column_config.NumberColumn(c, format="%.4f") for c in ["Precision", "Recall", "F1 Score"]},
                "Support": st.column_config.NumberColumn("Support", format="%d",
                                                         help="Number of students with this actual outcome"),
            },
        )
        weakest = min(res["recall_by_class"], key=res["recall_by_class"].get)
        ui.note(f"{model_name} finds it hardest to recognise <b>{weakest}</b> students "
                f"(recall {res['recall_by_class'][weakest]:.4f}).")
