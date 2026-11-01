"""Predictions workspace: batch overview, searchable table, student detail and export."""
from html import escape

import pandas as pd
import streamlit as st

from model.features import CLASS_ORDER
from ui import charts
from ui import components as ui
from ui import services as svc
from ui.theme import OUTCOME_COLORS, RISK_COLORS


def _style(df):
    def color(v, palette):
        return f"color: {palette[v]}; font-weight: 600" if v in palette else ""
    styler = df.style.map(lambda v: color(v, OUTCOME_COLORS),
                          subset=[c for c in ["Predicted", "Actual"] if c in df.columns])
    return styler.map(lambda v: color(v, RISK_COLORS), subset=["Dropout risk"])


def render():
    ss = st.session_state
    ui.page_header("Predictions", "Outcome predictions for every student in the dataset", "users")
    ui.show_flash()
    c1, _ = st.columns([1, 2])
    with c1:
        ui.model_picker("pred_model_select")
    name = ss.selected_model
    ev = svc.current_evaluation()
    df = ss.dataset
    preds = svc.predictions_frame(df, ev, name)

    # ---------------- overview
    ui.section("Prediction overview", f"{name} on {ss.dataset_name}", "activity")
    counts = preds["Predicted"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    total = len(preds)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        ui.kpi_grid(cols=2, items=[
            {"label": "Students analysed", "value": f"{total:,}", "icon": "users", "sub": "rows in the dataset"},
            {"label": "Predicted Dropout", "value": f"{counts['Dropout']:,}", "icon": "user-x",
             "tone": "tone-danger", "sub": f"{counts['Dropout'] / total:.1%} of students"},
            {"label": "Predicted Enrolled", "value": f"{counts['Enrolled']:,}", "icon": "hourglass",
             "tone": "tone-warning", "sub": f"{counts['Enrolled'] / total:.1%} of students"},
            {"label": "Predicted Graduate", "value": f"{counts['Graduate']:,}", "icon": "user-check",
             "tone": "tone-success", "sub": f"{counts['Graduate'] / total:.1%} of students"},
        ])
        high = int((preds["Dropout risk"] == "High").sum())
        ui.note(f"<b>{high:,}</b> students are in the <b>High</b> dropout-risk band "
                f"(P(Dropout) ≥ {ss.risk_high:.0%}); Medium starts at {ss.risk_medium:.0%}. "
                "Change the bands in Settings.")
        ui.note("Probabilities are the model's <b>estimates</b>, not guaranteed outcomes. "
                "<b>Confidence</b> is simply the largest of the three class probabilities; it is not the "
                "probability that a student will succeed.")
    with right:
        charts.show(charts.donut(counts.to_dict()), "Donut chart of predicted outcome counts.")

    # ---------------- filters + table
    ui.section("Individual student predictions", "Search, sort, filter, then click a row for details", "search")
    f1, f2, f3 = st.columns([1.2, 1.25, 1.1], vertical_alignment="bottom")
    query = f1.text_input("Search Student ID", placeholder="e.g. STU-0042", key="pred_search")
    outcomes = f2.pills("Predicted outcome", CLASS_ORDER, selection_mode="multi", default=CLASS_ORDER,
                        key="pred_outcomes") or []
    risks = f3.pills("Dropout risk", ["High", "Medium", "Low"], selection_mode="multi",
                     default=["High", "Medium", "Low"], key="pred_risks") or []
    wrong_only = st.toggle("Show only wrong predictions", key="pred_wrong", disabled=not ev["has_labels"],
                           help="Rows where the prediction differs from the actual outcome. Needs a Target column.")

    view = preds[preds["Predicted"].isin(outcomes) & preds["Dropout risk"].isin(risks)]
    if query.strip():
        view = view[view["Student ID"].str.contains(query.strip(), case=False, regex=False)]
    if wrong_only and ev["has_labels"]:
        view = view[view["Correct?"] == False]  # noqa: E712 - None means unlabelled

    if view.empty:
        ui.empty_state("No students match these filters. Try clearing the search or adding outcomes back.")
        selected_row = None
    else:
        st.caption(f"Showing {len(view):,} of {total:,} students")
        prob_cfg = {c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
                    for c in ["P(Dropout)", "P(Enrolled)", "P(Graduate)"]}
        event = st.dataframe(
            _style(view), hide_index=True, width="stretch", height=420, key="pred_table",
            on_select="rerun", selection_mode="single-row",
            column_order=[c for c in ["Student ID", "Predicted", "Actual", "Correct?", "Confidence",
                                      "Dropout risk", "P(Dropout)", "P(Enrolled)", "P(Graduate)"]
                          if c in view.columns],
            column_config={
                "Confidence": st.column_config.NumberColumn(
                    "Confidence", format="percent", help="Largest class probability (an estimate)"),
                **prob_cfg,
                "Dropout risk": st.column_config.TextColumn("Dropout risk", help="Band based on P(Dropout)"),
                "Correct?": st.column_config.CheckboxColumn("Correct?"),
            },
        )
        rows = event.selection.rows if event else []
        selected_row = view.iloc[rows[0]] if rows else None

    c_dl, _ = st.columns([1, 3])
    with c_dl:
        export = view.merge(df.assign(**{"Student ID": svc.student_ids(df)}).drop(
            columns=[c for c in svc.ID_COLUMNS if c in df.columns and c != "Student ID"], errors="ignore"),
            on="Student ID", how="left", suffixes=("", " (input)")) if not view.empty else view
        st.download_button("Export predictions (CSV)", export.to_csv(index=False).encode(),
                           file_name=f"edupulse_predictions_{name.replace(' ', '_').lower()}.csv",
                           mime="text/csv", icon=":material/download:", disabled=view.empty)
    ui.note(f"The export contains the {len(view):,} rows shown above, for {escape(name)} on "
            f"{escape(ss.dataset_name)}, with every input column included.")

    # ---------------- detail
    ui.section("Student detail", "Select a row in the table to inspect one student", "user-check")
    if selected_row is None:
        ui.empty_state("Click any row in the table above to see that student's prediction in detail.")
        return
    _detail(selected_row, df, ev, name)


def _detail(row, df, ev, name):
    ids = svc.student_ids(df)
    idx = int((ids == row["Student ID"]).nonzero()[0][0])
    probs = ev["models"][name]["proba"][idx]
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        actual = row.get("Actual") if ev["has_labels"] else None
        actual_html = (f'<dt>Actual outcome</dt><dd style="color:{OUTCOME_COLORS.get(actual, "inherit")}">'
                       f'{escape(str(actual))}</dd>') if actual else "<dt>Actual outcome</dt><dd>Not in data</dd>"
        ui.html(f"""<div class="ep-card"><dl class="ep-kv">
        <dt>Student ID</dt><dd>{escape(row['Student ID'])}</dd>
        <dt>Predicted outcome</dt><dd style="color:{OUTCOME_COLORS[row['Predicted']]}">{row['Predicted']}</dd>
        {actual_html}
        <dt>Dropout risk band</dt><dd>{row['Dropout risk']}</dd>
        <dt>Model used</dt><dd>{escape(name)}</dd></dl></div>""")
        st.markdown("**Class probabilities (estimates)**")
        charts.show(charts.probability_bars(probs), f"Predicted probabilities for student {row['Student ID']}.")
    with right:
        st.markdown("**Key study features compared with the dataset average**")
        feats = svc.enriched(df)
        table = pd.DataFrame({
            "Feature": svc.KEY_FEATURES,
            "This student": [feats.iloc[idx][f] for f in svc.KEY_FEATURES],
            "Dataset average": [feats[f].mean() for f in svc.KEY_FEATURES],
        })
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"This student": st.column_config.NumberColumn(format="%.2f"),
                                    "Dataset average": st.column_config.NumberColumn(format="%.2f")})
        ui.note("These are descriptive comparisons, not explanations. The model combines all "
                f"{len(svc.required_columns())} inputs, so no single value 'causes' the prediction.")
        with st.expander("All input values for this student"):
            st.dataframe(df.iloc[[idx]][svc.required_columns()].T.rename(columns={idx: "Value"}),
                         width="stretch")
