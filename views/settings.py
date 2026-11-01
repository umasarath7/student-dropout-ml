"""Settings: risk bands, data reset and model information."""
import json

import sklearn
import streamlit as st

from ui import components as ui
from ui import services as svc


def _sync(src, dst):
    st.session_state[dst] = st.session_state[src]


def render():
    ss = st.session_state
    ui.page_header("Settings", "Adjust how EduPulse presents predictions", "settings")
    ui.show_flash()

    ui.section("Dropout risk bands", "Used in Predictions and Insights", "shield-alert")
    c1, c2 = st.columns(2)
    ss["w_risk_high"] = ss.risk_high
    ss["w_risk_medium"] = ss.risk_medium
    c1.slider("High risk when P(Dropout) is at least", 0.30, 0.90, step=0.05, key="w_risk_high",
              on_change=_sync, args=("w_risk_high", "risk_high"), format="%.2f")
    c2.slider("Medium risk when P(Dropout) is at least", 0.05, 0.60, step=0.05, key="w_risk_medium",
              on_change=_sync, args=("w_risk_medium", "risk_medium"), format="%.2f")
    if ss.risk_medium >= ss.risk_high:
        st.warning("The Medium threshold should be lower than the High threshold.", icon=":material/warning:")
    ui.note("Bands only change how predictions are grouped; they never change the model or its scores.")

    ui.section("Data", "", "database")
    ui.note(f"Current dataset: <b>{ss.dataset_name}</b> ({len(ss.dataset):,} rows).")
    st.button("Reset to bundled test set", on_click=svc.reset_dataset, icon=":material/restart_alt:",
              disabled=ss.dataset_source == "bundled")

    ui.section("About the models", "", "info")
    split = svc.split_summary()
    ui.note(f"Five scikit-learn {sklearn.__version__} pipelines trained on {split['train']:,} students "
            f"and tuned with 5-fold cross-validation. Retrain with <code>python -m model.train_all</code>.")
    st.code(json.dumps(svc.tuned_params(), indent=2), language="json")
