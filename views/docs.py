"""Documentation: what was done in this project and how to use EduPulse."""
import pandas as pd
import streamlit as st

from model.features import CLASS_ORDER, NOMINAL_COLS, TARGET
from ui import components as ui
from ui import guide
from ui import services as svc
from ui.components import METRIC_HELP


def render():
    ui.page_header("Documentation", "What was built, how it works, and how to use it", "file")
    ui.show_flash()
    tabs = st.tabs(["Project", "How it works", "Using the app", "Preparing your CSV", "Limitations"])
    with tabs[0]:
        _project()
    with tabs[1]:
        _how_it_works()
    with tabs[2]:
        _using_the_app()
    with tabs[3]:
        _preparing_csv()
    with tabs[4]:
        _limitations()


def _project():
    full = svc.bundled_dataset()
    split = svc.split_summary()
    counts = pd.read_csv(svc.FULL_CSV, encoding="utf-8-sig", usecols=[TARGET])[TARGET].value_counts()

    ui.section("The problem", "", "target")
    ui.note("Many students who start a degree never finish it. If a university can spot them early, "
            "for example after the first year, it can offer mentoring, fee support or academic help in time. "
            "EduPulse predicts from first-year records whether a student will <b>drop out</b>, "
            "<b>stay enrolled</b> (take longer than planned) or <b>graduate</b>: a three-class "
            "classification problem.")

    ui.section("The data", "", "database")
    ui.kpi_grid([
        {"label": "Students", "value": f"{split['total']:,}", "icon": "users", "sub": "one row each"},
        {"label": "Input features", "value": str(len(svc.required_columns())), "icon": "columns",
         "tone": "tone-cyan", "sub": "background, money, grades, economy"},
        *[{"label": c, "value": f"{counts[c]:,}", "icon": i, "tone": t, "sub": f"{counts[c] / counts.sum():.1%} of students"}
          for c, i, t in [("Graduate", "user-check", "tone-success"), ("Dropout", "user-x", "tone-danger"),
                          ("Enrolled", "hourglass", "tone-warning")]],
    ])
    ui.note("Source: UCI Machine Learning Repository, <i>Predict Students' Dropout and Academic Success</i> "
            "(also on Kaggle). Records come from a higher-education institution in Portugal. "
            "No missing values and no duplicate rows. No names or contact details.")

    ui.section("Results on the hold-out test set", f"All five models on the same {len(full):,} unseen students", "gauge")
    table = svc.comparison_table(svc.evaluate(full)).sort_values("F1", ascending=False)
    st.dataframe(table, width="stretch",
                 column_config={m: st.column_config.NumberColumn(m, format="%.4f") for m in svc.METRICS})
    best = table.index[0]
    ui.note(f"<b>Winner: {best}</b>, with the highest macro F1 ({table.loc[best, 'F1']:.4f}) and MCC "
            f"({table.loc[best, 'MCC']:.4f}). Logistic Regression is a close runner-up with the best AUC. "
            "Every model finds Graduate and Dropout students well but struggles with Enrolled students, "
            "who look like a mix of the other two groups.")


def _how_it_works():
    split = svc.split_summary()
    steps = [
        ("1", "Split", f"{split['train']:,} students ({split['train_pct']}%) for training and "
                       f"{split['test']:,} ({split['test_pct']}%) held back for testing. Stratified, so both "
                       "keep the same outcome mix. The test set is never used for training or tuning."),
        ("2", "New features", "Four extra columns: pass rate in each semester (units passed ÷ units "
                              "enrolled), change in average grade from semester 1 to 2, and total units passed."),
        ("3", "Encoding", f"The {len(NOMINAL_COLS)} code columns (e.g. Course) are one-hot encoded, with codes "
                          "used by fewer than 20 students grouped together. All number columns are scaled "
                          "to mean 0 and standard deviation 1."),
        ("4", "Training", "Five models: Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes and an "
                          "RBF Support Vector Machine. Each one's settings were picked by 5-fold "
                          "cross-validation on the training set, scored on macro F1."),
        ("5", "Evaluation", "Each tuned model predicts the test set and is scored on six metrics. Steps 2–4 "
                            "are saved inside each model, so EduPulse can take raw CSV columns directly."),
    ]
    ui.section("The pipeline", "From raw records to predictions", "split")
    ui.html('<div class="ep-steps">' + "".join(
        f'<div class="ep-step"><span class="ep-step-num">{n}</span><div class="ep-step-title">{t}</div>'
        f'<div class="ep-step-body">{b}</div></div>' for n, t, b in steps) + "</div>")

    ui.section("Settings chosen by cross-validation", "", "settings")
    params = svc.tuned_params()
    st.dataframe(pd.DataFrame([{"Model": m, "Settings": ", ".join(f"{k} = {v}" for k, v in p.items())}
                               for m, p in params.items()]), hide_index=True, width="stretch")

    ui.section("The six metrics", "Precision, Recall and F1 are macro averages: each outcome counts equally", "scale")
    st.dataframe(pd.DataFrame({"Metric": list(METRIC_HELP), "What it measures": list(METRIC_HELP.values())}),
                 hide_index=True, width="stretch")

    ui.section("Code structure", "", "columns")
    st.code("""model/        the ML backend: features, metrics, the five model definitions, training script
model/saved/  trained pipelines (.joblib), results.csv, best_params.json
ui/           EduPulse services (data, validation, live metrics), components, charts, theme
views/        one file per page
app.py        entry point: theme, sidebar and navigation

Retrain:  python -m model.train_all
Run:      streamlit run app.py""", language="text")


def _using_the_app():
    ui.section("Pages", "", "activity")
    st.dataframe(pd.DataFrame([
        ("Dashboard", "Dataset summary, upload, ranked model cards and dropdown, the six metrics, model comparison, confusion matrix and classification report"),
        ("Model Evaluation", "Full report on one model: settings, split, metrics, ROC curves, comparison and model-specific diagnostics"),
        ("Predictions", "Outcome counts, a searchable table of every student with probabilities and risk bands, a detail view per student, and CSV export"),
        ("Dataset", "Upload with step-by-step checks, quality summary, column types and a paged preview"),
        ("Insights", "Findings and suggested actions calculated from the current results"),
        ("Settings", "Dropout-risk band thresholds and a reset to the bundled test set"),
        ("Documentation", "This page"),
    ], columns=["Page", "What you'll find"]), hide_index=True, width="stretch")

    ui.section("Quick start", "", "lightbulb")
    ui.note("1. Open the <b>Dashboard</b>. The bundled test set is already loaded and scored.<br>"
            "2. Click a <b>model card</b> (or use the dropdown); every chart and number updates.<br>"
            "3. To use your own students, click <b>Upload New Dataset</b> and choose a CSV prepared as in "
            "<i>Preparing your CSV</i>.<br>"
            "4. Go to <b>Predictions</b> to find students at high risk of dropping out, open one for details, "
            "and export the list.")

    ui.section("Reading the risk bands", "", "shield-alert")
    ss = st.session_state
    ui.note(f"Each student gets a <b>dropout risk band</b> from the model's estimated probability of dropping "
            f"out: <b>High</b> at {ss.risk_high:.0%} or more, <b>Medium</b> from {ss.risk_medium:.0%}, "
            "otherwise <b>Low</b>. You can change these thresholds in Settings; it never changes the model.")


def _preparing_csv():
    ui.section("Do I need to preprocess my data?", "", "info")
    ui.note("<b>No machine-learning preprocessing is needed.</b> Scaling, encoding and the extra features "
            "all happen inside the saved models. You only need the right <b>columns</b> and the same "
            "<b>number codes</b> as the original data: every input column is numeric, and categories such "
            "as Course are stored as codes. The easiest way is to start from the template below.")

    ui.section("Rules", "", "check-circle")
    ui.note("• Comma-separated CSV with one header row.<br>"
            "• All 34 input columns, named exactly as in the template, including the original spelling "
            "<b>Nacionality</b>.<br>"
            "• Numbers only and no empty cells in those 34 columns.<br>"
            "• Optional: <b>Student ID</b> (shown in Predictions) and <b>Target</b> "
            f"({', '.join(CLASS_ORDER)}) to have the models scored.<br>"
            "• Any other columns are ignored.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Upload template (CSV)", guide.template_csv(), file_name="edupulse_upload_template.csv",
                       mime="text/csv", icon=":material/download:", width="stretch", key="docs_tpl")
    c2.download_button("Column guide (CSV)", guide.guide_csv(), file_name="edupulse_column_guide.csv",
                       mime="text/csv", icon=":material/download:", width="stretch", key="docs_guide")
    c3.download_button("Sample test set (CSV)", svc.SAMPLE_CSV.read_bytes(), file_name="test_data.csv",
                       mime="text/csv", icon=":material/download:", width="stretch", key="docs_sample")
    ui.note("The template has every column plus three example students (one per outcome). Replace those rows "
            "with your own.")

    ui.section("Column guide", "Allowed values come from the data the models were trained on", "columns")
    query = st.text_input("Filter columns", placeholder="e.g. grade, mother, fees", key="docs_col_filter")
    g = guide.column_guide()
    if query.strip():
        q = query.strip().lower()
        g = g[g["Column"].str.lower().str.contains(q, regex=False) | g["Meaning"].str.lower().str.contains(q, regex=False)]
    st.dataframe(g, hide_index=True, width="stretch", height=420,
                 column_config={"Meaning": st.column_config.TextColumn(width="large")})

    ui.section("What happens when you upload", "", "split")
    ui.note("EduPulse reads the file and runs these checks before replacing anything.<br>"
            "<b>Blocking errors</b> (your current data is kept): not a CSV, unreadable file, no rows, missing "
            "required columns, text in a number column, empty cells.<br>"
            "<b>Warnings</b> (the file is still used): no Target column, unknown Target values, codes the "
            "models have never seen, values outside the training range, extra columns. Unknown codes are "
            "treated as rare categories, so check warnings before trusting those students' predictions.")


def _limitations():
    ui.section("Things to keep in mind", "", "alert")
    for title, body in [
        ("Predictions are estimates", "The models learn patterns from past students. Probabilities are "
         "estimates, not guarantees, and a pattern is not a cause."),
        ("Enrolled students are hard to spot", "All five models find fewer than half of the Enrolled "
         "students. Treat an Enrolled prediction, or a missed one, with extra care."),
        ("One institution, one period", "The data comes from one Portuguese institution. Results may differ "
         "for other universities, countries or years, so re-check on new intakes."),
        ("Codes aren't labelled", "This version of the dataset doesn't say which course or occupation each "
         "code means, so EduPulse can't accept labels such as 'Nursing' directly."),
        ("Use responsibly", "Use risk bands to decide who to talk to first, not to make decisions about a "
         "student on their own."),
    ]:
        ui.insight_card("caution", title, body)
