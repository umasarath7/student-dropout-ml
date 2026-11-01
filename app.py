"""
EduPulse: student outcome prediction (ML Assignment 2).

Entry point. Sets up the theme, shared state and sidebar navigation; each
page lives in views/. The ML models themselves live in model/.
"""
import streamlit as st

from ui import components as ui
from ui import services as svc
from ui.theme import inject_css, mark_active_nav
from views import dashboard, dataset, docs, evaluation, insights, predictions, settings

st.set_page_config(page_title="EduPulse · Student Outcome Intelligence", page_icon="🎓",
                   layout="wide", initial_sidebar_state="auto")
inject_css()
svc.init_state()

pages = [
    st.Page(dashboard.render, title="Dashboard", icon=":material/space_dashboard:", url_path="dashboard", default=True),
    st.Page(evaluation.render, title="Model Evaluation", icon=":material/analytics:", url_path="evaluation"),
    st.Page(predictions.render, title="Predictions", icon=":material/online_prediction:", url_path="predictions"),
    st.Page(dataset.render, title="Dataset", icon=":material/database:", url_path="dataset"),
    st.Page(insights.render, title="Insights", icon=":material/lightbulb:", url_path="insights"),
    st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings"),
    st.Page(docs.render, title="Documentation", icon=":material/menu_book:", url_path="docs"),
]
current = st.navigation(pages, position="hidden")
mark_active_nav("" if current.url_path in ("", "dashboard") else current.url_path)

with st.sidebar:
    ui.sidebar_brand()
    for page in pages:
        st.page_link(page, width="stretch")
    ui.sidebar_footer()

current.run()
