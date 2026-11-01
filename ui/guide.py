"""
CSV upload template, column guide and range checks.

Allowed values come from the original dataset (data/students.csv), so the
guide always matches what the models were trained on.
"""
import io

import pandas as pd
import streamlit as st

from model.features import NOMINAL_COLS, TARGET, load_csv

# Plain-English meaning of each column. Code columns are described without
# naming individual codes: this version of the dataset doesn't publish them.
MEANINGS = {
    "Marital status": "Student's marital status",
    "Application mode": "Route the student used to apply",
    "Application order": "Where this course ranked among the student's application choices (0 = first)",
    "Course": "Degree programme",
    "Daytime/evening attendance": "1 = daytime classes, 0 = evening classes",
    "Previous qualification": "Highest qualification before enrolling",
    "Nacionality": "Nationality (keep the original spelling 'Nacionality')",
    "Mother's qualification": "Mother's highest qualification",
    "Father's qualification": "Father's highest qualification",
    "Mother's occupation": "Mother's occupation",
    "Father's occupation": "Father's occupation",
    "Displaced": "1 = studying away from home, 0 = not",
    "Educational special needs": "1 = has special educational needs, 0 = not",
    "Debtor": "1 = owes money to the institution, 0 = not",
    "Tuition fees up to date": "1 = fees paid up, 0 = behind on fees",
    "Gender": "1 = male, 0 = female",
    "Scholarship holder": "1 = holds a scholarship, 0 = not",
    "Age at enrollment": "Age in years when the student enrolled",
    "International": "1 = international student, 0 = not",
    "Curricular units 1st sem (credited)": "Semester 1 units credited from earlier study",
    "Curricular units 1st sem (enrolled)": "Semester 1 units the student enrolled in",
    "Curricular units 1st sem (evaluations)": "Semester 1 assessments taken",
    "Curricular units 1st sem (approved)": "Semester 1 units passed",
    "Curricular units 1st sem (grade)": "Semester 1 average grade on the 0–20 scale (0 if nothing passed)",
    "Curricular units 1st sem (without evaluations)": "Semester 1 units with no assessment taken",
    "Curricular units 2nd sem (credited)": "Semester 2 units credited from earlier study",
    "Curricular units 2nd sem (enrolled)": "Semester 2 units the student enrolled in",
    "Curricular units 2nd sem (evaluations)": "Semester 2 assessments taken",
    "Curricular units 2nd sem (approved)": "Semester 2 units passed",
    "Curricular units 2nd sem (grade)": "Semester 2 average grade on the 0–20 scale (0 if nothing passed)",
    "Curricular units 2nd sem (without evaluations)": "Semester 2 units with no assessment taken",
    "Unemployment rate": "National unemployment rate (%) when the student enrolled",
    "Inflation rate": "National inflation rate (%) when the student enrolled",
    "GDP": "GDP indicator when the student enrolled",
}


@st.cache_data(show_spinner=False)
def column_profile():
    """Type and allowed values for every input column, from the original data."""
    from ui.services import FULL_CSV, required_columns
    df = load_csv(FULL_CSV)
    rows = []
    for col in required_columns():
        v = df[col]
        if col in NOMINAL_COLS:
            kind, allowed = "Code", f"Whole number {int(v.min())}–{int(v.max())}"
        elif set(v.unique()) <= {0, 1}:
            kind, allowed = "Yes/No flag", "0 or 1"
        elif (v % 1 == 0).all():
            kind, allowed = "Whole number", f"{int(v.min())}–{int(v.max())}"
        else:
            kind, allowed = "Decimal", f"{v.min():g} to {v.max():g}"
        rows.append({"Column": col, "Required": "Yes", "Type": kind, "Allowed values": allowed,
                     "Meaning": MEANINGS.get(col, ""), "_min": float(v.min()), "_max": float(v.max()),
                     "_codes": sorted(v.unique().tolist()) if col in NOMINAL_COLS else None})
    rows.insert(0, {"Column": "Student ID", "Required": "Optional", "Type": "Text",
                    "Allowed values": "Any label", "Meaning": "Your own identifier, shown in Predictions",
                    "_min": None, "_max": None, "_codes": None})
    rows.append({"Column": TARGET, "Required": "Optional", "Type": "Text",
                 "Allowed values": "Dropout, Enrolled or Graduate",
                 "Meaning": "Actual outcome. Include it to have the models scored",
                 "_min": None, "_max": None, "_codes": None})
    return pd.DataFrame(rows)


def column_guide():
    return column_profile().drop(columns=["_min", "_max", "_codes"])


@st.cache_data(show_spinner=False)
def guide_csv():
    return column_guide().to_csv(index=False).encode()


@st.cache_data(show_spinner=False)
def template_csv():
    """Header with every column plus three example students from the test set."""
    from ui.services import SAMPLE_CSV, required_columns
    sample = load_csv(SAMPLE_CSV)
    picks = [sample[sample[TARGET] == c].iloc[0] for c in ["Dropout", "Enrolled", "Graduate"]]
    rows = pd.DataFrame(picks)[required_columns() + [TARGET]].reset_index(drop=True)
    rows.insert(0, "Student ID", ["EXAMPLE-001", "EXAMPLE-002", "EXAMPLE-003"])
    buf = io.StringIO()
    rows.to_csv(buf, index=False)
    return buf.getvalue().encode()


def range_warnings(df):
    """Warn (don't block) when values fall outside what the models were trained on."""
    warnings = []
    for _, r in column_profile().iterrows():
        col = r["Column"]
        if col not in df.columns or r["_min"] is None or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        v = df[col]
        if r["_codes"] is not None:
            bad = ~v.isin(r["_codes"])
            if bad.any():
                warnings.append(f"'{col}': {int(bad.sum())} row(s) use codes the models have never seen "
                                f"(e.g. {v[bad].iloc[0]:g}); allowed {r['Allowed values'].lower()}. "
                                "These are treated as rare categories, so predictions may be less reliable.")
        else:
            bad = (v < r["_min"]) | (v > r["_max"])
            if bad.any():
                warnings.append(f"'{col}': {int(bad.sum())} row(s) are outside the training range "
                                f"({r['Allowed values']}), e.g. {v[bad].iloc[0]:g}. Check for typos; "
                                "predictions for these students may be less reliable.")
    return warnings


def csv_help(key):
    """Template, column guide and the upload rules, shown next to upload boxes."""
    with st.expander("CSV template & rules", icon=":material/description:"):
        st.markdown(
            "- Save as a **comma-separated CSV** with one header row.\n"
            "- Use the **34 column names exactly** as in the template, including the original spelling "
            "**Nacionality**.\n"
            "- **Numbers only**, no empty cells. Category columns use number codes, the same as the "
            "sample file; no other preprocessing is needed.\n"
            "- `Student ID` and `Target` (Dropout / Enrolled / Graduate) are optional. "
            "Other extra columns are ignored."
        )
        c1, c2 = st.columns(2)
        c1.download_button("Upload template", template_csv(), file_name="edupulse_upload_template.csv",
                           mime="text/csv", key=f"{key}_tpl", icon=":material/download:", width="stretch")
        c2.download_button("Column guide", guide_csv(), file_name="edupulse_column_guide.csv",
                           mime="text/csv", key=f"{key}_guide", icon=":material/download:", width="stretch")
