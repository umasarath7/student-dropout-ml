"""Dashboard: the EduPulse overview and landing page."""
from html import escape

import streamlit as st

from ui import components as ui
from ui import services as svc
from ui.icons import icon
from views.shared import comparison_block, confusion_and_report


def render():
    ss = st.session_state
    ev = svc.current_evaluation()
    errors, _ = svc.validate_dataset(ss.dataset)
    status = ("Dataset ready", "var(--ep-success)") if not errors else ("Needs attention", "var(--ep-danger)")

    with st.container(key="ep_hero"):
        ui.html(f"""<div class="ep-hero-body">
        <h1>EduPulse</h1><div class="ep-tagline">Predict. Understand. Empower.</div>
        <p class="ep-desc">AI-powered insights that help identify at-risk students, predict academic
        outcomes, and enable better student success.</p>
        <div class="ep-chips">
          <span class="ep-chip"><span class="ep-dot" style="background:{status[1]}"></span>{status[0]}</span>
          <span class="ep-chip">{icon('users', 14)} {len(ss.dataset):,} students</span>
          <span class="ep-chip">{icon('file', 14)} {escape(ss.dataset_name)}</span>
          <span class="ep-chip">{icon('clock', 14)} Evaluated {ss.dataset_loaded_at.strftime('%d %b, %H:%M')}</span>
        </div></div>""")
        with st.container(key="ep_cta"):
            text, button = st.columns([3, 1.15], vertical_alignment="center", gap="medium")
            with text:
                ui.html(f"""<div class="ep-cta-text"><span class="ep-cta-icon">{icon('upload', 22)}</span>
                <div><div class="ep-cta-title">Want to know how your students will do?</div>
                <div class="ep-cta-sub">Upload your own student data (CSV) and EduPulse will predict who is
                likely to drop out, stay enrolled or graduate, in seconds, with all five models.</div></div></div>""")
            with button:
                with st.popover("Upload New Dataset", icon=":material/upload_file:", type="primary",
                                width="stretch"):
                    st.markdown("**Upload test data (CSV)**")
                    ui.upload_panel("upload_dashboard")
                    ui.sample_download("sample_dashboard")
    ui.show_flash()

    ui.section("Dataset", "What the models are being evaluated on", "database")
    ui.dataset_summary_card()

    ui.section("Machine learning models", "Ranked best to worst by macro F1. Pick one and every panel below updates to match.", "activity")
    ui.model_cards(ev, "dash_card")
    c1, _ = st.columns([1, 2])
    with c1:
        ui.model_picker("dash_model_select", "Or choose a classifier from the list")

    name = ss.selected_model
    res = ev["models"][name]
    ui.section(f"Model performance · {name}", "Scores on the current dataset. Hover the ⓘ for meanings.", "gauge")
    if not ev["has_labels"]:
        ui.empty_state("This dataset has no 'Target' column, so scores can't be calculated. "
                       "Predictions are still available on the Predictions page.")
        return
    ui.metric_kpis(res["metrics"])
    ui.note("Precision, Recall and F1 are macro averages, so the small Enrolled group counts as much "
            "as the others. AUC is one-vs-rest.")

    ui.section("Model comparison", "All five models on the same students", "scale")
    comparison_block(ev, "dash_metric")

    confusion_and_report(res, name)
