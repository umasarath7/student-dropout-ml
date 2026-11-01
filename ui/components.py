"""Reusable EduPulse building blocks shared by every page."""
from html import escape

import streamlit as st

from model.features import load_csv
from ui import services as svc
from ui.icons import MODEL_ICONS, icon

METRIC_HELP = {
    "Accuracy": "Share of all students whose outcome the model predicted correctly.",
    "AUC": "How well the model ranks students, averaged one-vs-rest over the three outcomes. "
           "1.0 is perfect ranking, 0.5 is no better than chance.",
    "Precision": "When the model predicts an outcome, how often it is right. Macro average, "
                 "so each outcome counts equally.",
    "Recall": "Of the students who truly had an outcome, how many the model found. Macro average.",
    "F1": "Balance of precision and recall (their harmonic mean). Macro average.",
    "MCC": "Matthews correlation between predicted and actual outcomes. From -1 to 1, where 0 "
           "means no better than guessing. Robust when classes are imbalanced.",
}
METRIC_ICONS = {"Accuracy": ("target", ""), "AUC": ("activity", "tone-cyan"),
                "Precision": ("crosshair", "tone-teal"), "Recall": ("search", "tone-success"),
                "F1": ("scale", "tone-warning"), "MCC": ("gauge", "tone-danger")}
METRIC_LABELS = {"F1": "F1 Score", "MCC": "MCC"}


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def sidebar_brand():
    html(f"""<div class="ep-brand"><div class="ep-logo">{icon('activity', 20)}</div>
    <div><div class="ep-brand-name">EduPulse</div>
    <div class="ep-brand-tag">Predict. Understand. Empower.</div></div></div>""")


def sidebar_footer():
    ss = st.session_state
    html(f"""<div class="ep-side-footer"><b>EduPulse</b> · student outcome intelligence<br>
    Dataset: {escape(ss.dataset_name)}<br>Model: {escape(ss.selected_model)}</div>""")


def page_header(title, subtitle, icon_name):
    html(f"""<div class="ep-page"><span class="ep-icon" style="width:40px;height:40px">{icon(icon_name, 21)}</span>
    <div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div></div>""")


def section(title, subtitle="", icon_name=None):
    ic = f'<span class="ep-icon">{icon(icon_name)}</span>' if icon_name else ""
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    html(f'<div class="ep-section">{ic}<div><h3>{escape(title)}</h3>{sub}</div></div>')


def badge(text, tone="neutral"):
    return f'<span class="ep-badge badge-{tone}">{escape(str(text))}</span>'


def kpi_grid(items, cols=None):
    """items: dicts with label, value, sub, icon, tone, tip."""
    cards = []
    for it in items:
        tip = (f'<span class="ep-tip" tabindex="0" role="img" aria-label="{escape(it["tip"])}" '
               f'data-tip="{escape(it["tip"])}">{icon("info", 15)}</span>') if it.get("tip") else ""
        cards.append(f"""<div class="ep-kpi {it.get('tone', '')}">
        <div class="ep-kpi-label"><span class="ep-icon">{icon(it.get('icon', 'info'), 16)}</span>
        {escape(it['label'])}{tip}</div>
        <div class="ep-kpi-value">{escape(it['value'])}</div>
        <div class="ep-kpi-sub">{escape(it.get('sub', ''))}</div></div>""")
    extra = f" cols-{cols}" if cols else ""
    html(f'<div class="ep-kpi-grid{extra}">{"".join(cards)}</div>')


def metric_kpis(metrics):
    items = []
    for m in svc.METRICS:
        v = metrics.get(m)
        ic, tone = METRIC_ICONS[m]
        if v is None or v != v:  # NaN
            value, sub = "n/a", "needs all three outcomes in the data"
        else:
            value = f"{v:.4f}"
            sub = f"{v:.2%} of students correct" if m == "Accuracy" else (
                "−1 to 1 scale" if m == "MCC" else "0 to 1 scale")
        items.append({"label": METRIC_LABELS.get(m, m), "value": value, "sub": sub,
                      "icon": ic, "tone": tone, "tip": METRIC_HELP[m]})
    kpi_grid(items)


def empty_state(message, icon_name="info"):
    html(f'<div class="ep-empty"><span class="ep-icon">{icon(icon_name)}</span>'
         f'<div>{escape(message)}</div></div>')


def note(text_html):
    html(f'<div class="ep-note">{text_html}</div>')


def show_flash():
    flash = st.session_state.get("flash")
    if not flash:
        return
    box = {"success": st.success, "error": st.error, "info": st.info, "warning": st.warning}[flash["kind"]]
    with st.container():
        details = "".join(f"\n- {d}" for d in flash["details"])
        box(flash["message"] + details, icon={"success": ":material/check_circle:",
                                              "error": ":material/error:", "info": ":material/info:",
                                              "warning": ":material/warning:"}[flash["kind"]])
        if st.button("Dismiss", key="dismiss_flash", type="tertiary"):
            st.session_state.flash = None
            st.rerun()


# ---------------------------------------------------------------- model selection
def _set_model(name):
    st.session_state.selected_model = name


def _sync_from_widget(key):
    st.session_state.selected_model = st.session_state[key]


def model_picker(key, label="Choose a classifier"):
    """Dropdown bound to the app-wide selected model."""
    st.session_state[key] = st.session_state.selected_model
    st.selectbox(label, svc.model_names(), key=key, on_change=_sync_from_widget, args=(key,))


def model_cards(ev, key):
    table = svc.comparison_table(ev)
    selected = st.session_state.selected_model
    best = table["F1"].idxmax() if len(table) else None
    cols = st.columns(len(svc.model_names()))
    for col, name in zip(cols, svc.model_names()):
        is_sel = name == selected
        if name in table.index:
            acc = table.loc[name, "Accuracy"]
            acc_html = f'<div class="ep-model-acc">{acc:.4f}<span>accuracy</span></div>'
            status = badge("Best F1", "success") if name == best else badge("Evaluated", "info")
        else:
            acc_html = '<div class="ep-model-acc" style="color:var(--ep-muted)">—<span>no labels</span></div>'
            status = badge("Predictions only", "neutral")
        with col:
            html(f"""<div class="ep-model {'selected' if is_sel else ''}">
            <div class="ep-model-top"><span class="ep-icon">{icon(MODEL_ICONS.get(name, 'activity'))}</span>{status}</div>
            <div class="ep-model-name">{escape(name)}</div>{acc_html}</div>""")
            st.button("Selected" if is_sel else "Select", key=f"{key}_{name}", on_click=_set_model,
                      args=(name,), disabled=is_sel, type="primary" if is_sel else "secondary",
                      width="stretch", icon=":material/check:" if is_sel else None)


# ---------------------------------------------------------------- dataset
def dataset_summary_card():
    ss = st.session_state
    ev = svc.current_evaluation()
    split = svc.split_summary()
    errors, _ = svc.validate_dataset(ss.dataset)
    status = badge("Dataset ready", "success") if not errors else badge("Needs attention", "danger")
    if ss.dataset_source == "bundled":
        split_txt = f"{split['train_pct']}% train · {split['test_pct']}% test"
        role = "Bundled hold-out test set"
    else:
        split_txt = f"Trained on {split['train']:,} · scoring your file"
        role = "Your uploaded file"
    labels = (f"{ev['n_labeled']:,} labelled" if ev["has_labels"] else "No labels")
    fields = [
        ("Dataset", f"{icon('file', 15)} {escape(ss.dataset_name)}"),
        ("Students", f"{len(ss.dataset):,}"),
        ("Training / testing", split_txt),
        ("Labels", labels),
        ("Status", status),
        ("Last updated", ss.dataset_loaded_at.strftime("%d %b %Y, %H:%M")),
    ]
    cells = "".join(f'<div><div class="ep-field-label">{k}</div><div class="ep-field-value">{v}</div></div>'
                    for k, v in fields)
    html(f'<div class="ep-card"><div class="ep-note" style="margin-bottom:10px">{escape(role)}</div>'
         f'<div class="ep-dataset">{cells}</div></div>')


def upload_panel(key):
    """Drag-and-drop upload with step-by-step validation. Replaces data only on success."""
    f = st.file_uploader("Drag and drop a CSV file, or browse", type=["csv"], key=key,
                         help="Use the same columns as test_data.csv. Keep the 'Target' "
                              "column if you want the models scored.")
    if f is None or st.session_state.get("_handled_upload") == f.file_id:
        return
    st.session_state._handled_upload = f.file_id
    progress = st.progress(0, text=f"Reading {f.name}...")
    if not f.name.lower().endswith(".csv"):
        progress.empty()
        svc.set_flash("error", f"{f.name} is not a CSV file. Your current dataset was kept.")
        st.rerun()
    try:
        df = load_csv(f)
    except Exception as exc:  # noqa: BLE001 - shown to the user
        progress.empty()
        svc.set_flash("error", f"Couldn't read {f.name} as a CSV ({type(exc).__name__}). "
                               "Check it is comma-separated with one header row. "
                               "Your current dataset was kept.")
        st.rerun()
    progress.progress(40, text="Checking required columns and values...")
    errors, warnings = svc.validate_dataset(df)
    if errors:
        progress.empty()
        svc.set_flash("error", f"{f.name} can't be used yet. Your current dataset was kept.", errors)
        st.rerun()
    progress.progress(70, text="Running all five models...")
    svc.evaluate(df.reset_index(drop=True))
    progress.progress(100, text="Done")
    svc.set_dataset(df, f.name, "upload")
    svc.set_flash("success", f"Using your file {f.name}: {len(df):,} students, "
                             "evaluated with all five models.", warnings)
    st.toast(f"{f.name} loaded", icon=":material/check_circle:")
    st.rerun()


def sample_download(key):
    st.download_button("Download sample CSV", svc.SAMPLE_CSV.read_bytes(), file_name="test_data.csv",
                       mime="text/csv", key=key, icon=":material/download:")


def insight_card(kind, title, body, why=None):
    """kind: finding | action | caution."""
    labels = {"finding": ("Finding", "info", "search"), "action": ("Suggested action", "success", "lightbulb"),
              "caution": ("Watch out", "warning", "alert")}
    label, tone, ic = labels[kind]
    why_html = f'<p class="ep-why"><b>Why it matters:</b> {why}</p>' if why else ""
    html(f"""<div class="ep-insight kind-{kind}"><span class="ep-badge badge-{tone}">{icon(ic, 13)} {label}</span>
    <h4>{escape(title)}</h4><p>{body}</p>{why_html}</div>""")
