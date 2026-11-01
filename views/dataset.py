"""Dataset management: upload, validate, inspect and preview."""
import math

import pandas as pd
import streamlit as st

from model.features import CLASS_ORDER, TARGET
from ui import charts
from ui import components as ui
from ui import services as svc


def render():
    ss = st.session_state
    df = ss.dataset
    ui.page_header("Dataset", "Upload, validate and inspect the data being evaluated", "database")
    ui.show_flash()

    up, side = st.columns([2, 1], gap="large")
    with up:
        ui.section("Upload a dataset", "Your current data is only replaced if the new file passes every check", "file")
        ui.upload_panel("upload_dataset_page")
    with side:
        ui.section("Sample and reset", "", "copy")
        ui.sample_download("sample_dataset_page")
        if ss.dataset_source != "bundled":
            st.button("Use bundled test set again", on_click=svc.reset_dataset, icon=":material/restart_alt:")
        ui.note("The sample is the 20% hold-out test set (885 students) that the models never saw "
                "during training.")

    ui.section("Current dataset", "", "database")
    ui.dataset_summary_card()

    errors, warnings = svc.validate_dataset(df)
    needed = svc.required_columns()
    present = sum(c in df.columns for c in needed)
    missing_cells = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())
    ui.section("Quality summary", "", "check-circle")
    ui.kpi_grid([
        {"label": "Records", "value": f"{len(df):,}", "icon": "users", "sub": "student rows"},
        {"label": "Columns", "value": f"{df.shape[1]}", "icon": "columns", "tone": "tone-cyan",
         "sub": f"{present}/{len(needed)} required present"},
        {"label": "Missing values", "value": f"{missing_cells:,}", "icon": "alert",
         "tone": "tone-success" if missing_cells == 0 else "tone-warning", "sub": "empty cells"},
        {"label": "Duplicate rows", "value": f"{dupes:,}", "icon": "copy",
         "tone": "tone-success" if dupes == 0 else "tone-warning", "sub": "identical records"},
        {"label": "Validation", "value": "Passed" if not errors else "Failed", "icon": "check-circle",
         "tone": "tone-success" if not errors else "tone-danger",
         "sub": f"{len(warnings)} note(s)" if warnings else "no issues"},
    ])
    for e in errors:
        st.error(e, icon=":material/error:")
    for w in warnings:
        st.info(w, icon=":material/info:")
    if dupes:
        ui.note("Duplicate rows aren't removed automatically: two students can genuinely share the same "
                "values, since the data has no names or IDs.")

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        ui.section("Columns", "Name, type, completeness and role", "columns")
        cols = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Missing": df.isna().sum().to_numpy(),
            "Unique values": df.nunique().to_numpy(),
            "Role": ["Target" if c == TARGET else "Required feature" if c in needed else "Ignored"
                     for c in df.columns],
        })
        st.dataframe(cols, hide_index=True, width="stretch", height=360)
    with right:
        ui.section("Actual outcomes", "Distribution of the Target column", "activity")
        if TARGET in df.columns and df[TARGET].isin(CLASS_ORDER).any():
            counts = df[TARGET].value_counts().reindex(CLASS_ORDER, fill_value=0)
            charts.show(charts.class_distribution(counts), "Bar chart of actual outcome counts.")
            ui.note(" · ".join(f"{c}: <b>{n:,}</b> ({n / counts.sum():.1%})" for c, n in counts.items()))
        else:
            ui.empty_state("No Target column in this dataset.")

    ui.section("Data preview", "Records are anonymised: the data has no names or contact details", "file")
    p1, p2, _ = st.columns([1, 1, 3])
    size = p1.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="preview_size")
    pages = max(1, math.ceil(len(df) / size))
    page = p2.number_input(f"Page (1–{pages})", min_value=1, max_value=pages, value=1, step=1, key="preview_page")
    start = (page - 1) * size
    st.dataframe(df.iloc[start:start + size], width="stretch")
    st.caption(f"Rows {start + 1:,}–{min(start + size, len(df)):,} of {len(df):,}")
