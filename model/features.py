"""
Shared data loading, feature engineering and preprocessing.

Every model in this folder uses the same steps defined here, so the
comparison between the five classifiers is fair.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

TARGET = "Target"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]

# These columns hold codes (e.g. Course 1..17), not real quantities,
# so they are one-hot encoded instead of treated as numbers.
NOMINAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Extra columns we build from the raw ones (see add_study_features).
ENGINEERED_COLS = [
    "Pass rate 1st sem",
    "Pass rate 2nd sem",
    "Grade change",
    "Units approved total",
]


def load_csv(path_or_buffer):
    """Read a CSV and tidy up the header (the original file has a BOM)."""
    df = pd.read_csv(path_or_buffer, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def add_study_features(X):
    """
    Add a few features that describe how the student is coping:
    - pass rate in each semester (approved units / enrolled units)
    - change in average grade from semester 1 to semester 2
    - total units approved across both semesters
    """
    X = X.copy()
    enr1 = X["Curricular units 1st sem (enrolled)"]
    enr2 = X["Curricular units 2nd sem (enrolled)"]
    app1 = X["Curricular units 1st sem (approved)"]
    app2 = X["Curricular units 2nd sem (approved)"]
    X["Pass rate 1st sem"] = np.where(enr1 > 0, app1 / enr1.replace(0, 1), 0.0)
    X["Pass rate 2nd sem"] = np.where(enr2 > 0, app2 / enr2.replace(0, 1), 0.0)
    X["Grade change"] = (
        X["Curricular units 2nd sem (grade)"] - X["Curricular units 1st sem (grade)"]
    )
    X["Units approved total"] = app1 + app2
    return X


def feature_columns(df):
    return [c for c in df.columns if c != TARGET]


def build_preprocessor(raw_feature_cols):
    numeric_cols = [c for c in raw_feature_cols if c not in NOMINAL_COLS] + ENGINEERED_COLS
    encode = ColumnTransformer(
        transformers=[
            (
                "codes",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              sparse_output=False),
                NOMINAL_COLS,
            ),
            ("numbers", StandardScaler(), numeric_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("study_features", FunctionTransformer(add_study_features)),
            ("encode", encode),
        ]
    )
