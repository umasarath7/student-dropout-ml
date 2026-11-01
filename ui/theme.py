"""EduPulse design tokens and global styles."""
import streamlit as st

C = {
    "bg": "#0B0D18", "sidebar": "#111529", "card": "#171D33", "elevated": "#202741",
    "primary": "#7650E8", "violet": "#A78BFA", "cyan": "#38BDF8", "teal": "#2DD4BF",
    "success": "#34D399", "warning": "#FBBF24", "danger": "#FB7185",
    "text": "#F8FAFC", "muted": "#A1A1AA", "border": "#30364D", "slate": "#3B4263",
}
OUTCOME_COLORS = {"Dropout": C["danger"], "Enrolled": C["warning"], "Graduate": C["success"]}
RISK_COLORS = {"High": C["danger"], "Medium": C["warning"], "Low": C["success"]}

_CSS = f"""
<style>
:root {{
  --ep-bg:{C['bg']}; --ep-sidebar:{C['sidebar']}; --ep-card:{C['card']}; --ep-elev:{C['elevated']};
  --ep-primary:{C['primary']}; --ep-violet:{C['violet']}; --ep-cyan:{C['cyan']}; --ep-teal:{C['teal']};
  --ep-success:{C['success']}; --ep-warning:{C['warning']}; --ep-danger:{C['danger']};
  --ep-text:{C['text']}; --ep-muted:{C['muted']}; --ep-border:{C['border']};
}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1360px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--ep-border); }}

/* Sidebar navigation */
[data-testid="stSidebar"] [data-testid="stPageLink"] a {{
  border-radius: 10px; padding: 9px 12px; margin: 2px 0; transition: background .15s ease, color .15s ease;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{ background: rgba(118,80,232,.12); }}
.ep-brand {{ display:flex; gap:12px; align-items:center; padding: 4px 4px 16px; margin-bottom: 8px;
  border-bottom: 1px solid var(--ep-border); }}
.ep-logo {{ width:38px; height:38px; border-radius:11px; flex-shrink:0; display:flex; align-items:center;
  justify-content:center; color:#fff; background: linear-gradient(135deg, var(--ep-primary), var(--ep-cyan)); }}
.ep-brand-name {{ font-weight:700; font-size:1.18rem; color:var(--ep-text); line-height:1.1; }}
.ep-brand-tag {{ font-size:.74rem; color:var(--ep-muted); margin-top:2px; }}
.ep-side-footer {{ margin-top: 22px; padding-top: 14px; border-top:1px solid var(--ep-border);
  color: var(--ep-muted); font-size: .76rem; line-height:1.45; }}
.ep-side-footer b {{ color: var(--ep-text); font-weight:600; }}

/* Hero */
.ep-hero {{ display:flex; flex-wrap:wrap; gap:18px; justify-content:space-between; align-items:flex-end;
  padding: 24px 26px; border:1px solid var(--ep-border); border-radius:14px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(118,80,232,.22), transparent 60%), var(--ep-card); }}
.ep-hero h1, .ep-hero-body h1 {{ font-size:2.05rem; font-weight:700; margin:0; padding:0; color:var(--ep-text); letter-spacing:-.02em; }}
.ep-tagline {{ color:var(--ep-violet); font-weight:600; margin: 4px 0 8px; font-size:1rem; }}
.ep-desc {{ color:var(--ep-muted); max-width: 680px; margin:0; line-height:1.5; }}
.ep-chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top: 14px; }}
.ep-chip {{ display:inline-flex; align-items:center; gap:6px; padding:5px 11px; border-radius:999px;
  border:1px solid var(--ep-border); background: var(--ep-elev); color:var(--ep-text); font-size:.8rem; }}
.ep-chip svg {{ color: var(--ep-muted); }}
.ep-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}

/* Hero box with the upload call-to-action inside it */
.st-key-ep_hero {{ padding: 24px 26px 20px; border:1px solid var(--ep-border); border-radius:14px; gap: 18px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(118,80,232,.22), transparent 60%), var(--ep-card); }}
.st-key-ep_cta {{ position: relative; overflow: hidden; padding: 16px 18px; border-radius: 12px;
  border: 1px solid rgba(167,139,250,.55);
  background: linear-gradient(110deg, rgba(118,80,232,.55) 0%, rgba(99,66,214,.35) 45%, rgba(56,189,248,.28) 100%);
  box-shadow: 0 8px 28px rgba(118,80,232,.22); }}
.st-key-ep_cta::before {{ content:""; position:absolute; inset:0; pointer-events:none;
  background: linear-gradient(100deg, transparent 30%, rgba(255,255,255,.10) 50%, transparent 70%);
  transform: translateX(-100%); animation: ep-sheen 6s ease-in-out infinite; }}
@keyframes ep-sheen {{ 0%, 60% {{ transform: translateX(-100%); }} 100% {{ transform: translateX(100%); }} }}
@media (prefers-reduced-motion: reduce) {{ .st-key-ep_cta::before {{ animation: none; }} }}
.ep-hero-body {{ margin-bottom: 6px; }}
/* Streamlit gives text blocks a -16px bottom margin; cancel it inside the hero so padding is honoured */
.st-key-ep_hero [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
.st-key-ep_cta {{ padding: 18px 20px !important; }}
.st-key-ep_cta [data-testid="stHorizontalBlock"] {{ align-items: center; }}
.ep-cta-text {{ display:flex; gap:14px; align-items:center; }}
.ep-cta-icon {{ width:44px; height:44px; border-radius:12px; flex-shrink:0; display:flex; align-items:center;
  justify-content:center; color:#fff; background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25); }}
.ep-cta-title {{ font-weight:700; font-size:1.08rem; color:#fff; }}
.ep-cta-sub {{ color: rgba(248,250,252,.85); font-size:.88rem; line-height:1.45; margin-top:2px; }}
.st-key-ep_cta button[kind="primary"], .st-key-ep_cta [data-testid="stPopoverButton"] {{
  background: #fff !important; color: #3b21a8 !important; border: none !important; font-weight: 700;
  min-height: 46px; box-shadow: 0 6px 18px rgba(11,13,24,.35); transition: transform .15s ease, box-shadow .15s ease; }}
.st-key-ep_cta [data-testid="stPopoverButton"]:hover {{ transform: translateY(-1px); box-shadow: 0 10px 24px rgba(11,13,24,.45); }}
.st-key-ep_cta [data-testid="stPopoverButton"] * {{ color: #3b21a8 !important; }}

/* Documentation pipeline steps */
.ep-steps {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:12px; }}
.ep-step {{ background:var(--ep-card); border:1px solid var(--ep-border); border-radius:12px; padding:16px; }}
.ep-step-num {{ display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px;
  border-radius:50%; font-weight:700; font-size:.85rem; color:#fff;
  background: linear-gradient(135deg, var(--ep-primary), var(--ep-cyan)); }}
.ep-step-title {{ font-weight:600; margin-top:10px; color:var(--ep-text); }}
.ep-step-body {{ color:var(--ep-muted); font-size:.85rem; line-height:1.5; margin-top:4px; }}

/* Page header (non-dashboard pages) */
.ep-page {{ display:flex; gap:14px; align-items:center; margin-bottom: 6px; }}
.ep-page h1 {{ font-size:1.65rem; font-weight:700; margin:0; padding:0; letter-spacing:-.01em; }}
.ep-page p {{ color:var(--ep-muted); margin: 2px 0 0; }}

/* Section headers */
.ep-section {{ display:flex; align-items:flex-start; gap:10px; margin: 30px 0 12px; }}
.ep-section h3 {{ margin:0; padding:0; font-size:1.12rem; font-weight:600; color:var(--ep-text); }}
.ep-section p {{ margin:2px 0 0; color:var(--ep-muted); font-size:.86rem; }}

/* Generic card + icon */
.ep-card {{ background:var(--ep-card); border:1px solid var(--ep-border); border-radius:12px; padding:16px 18px; }}
.ep-icon {{ display:inline-flex; align-items:center; justify-content:center; width:32px; height:32px;
  border-radius:9px; flex-shrink:0; background: rgba(118,80,232,.16); color: var(--ep-violet); }}
.tone-cyan .ep-icon, .ep-icon.tone-cyan {{ background: rgba(56,189,248,.14); color: var(--ep-cyan); }}
.tone-teal .ep-icon, .ep-icon.tone-teal {{ background: rgba(45,212,191,.14); color: var(--ep-teal); }}
.tone-success .ep-icon, .ep-icon.tone-success {{ background: rgba(52,211,153,.14); color: var(--ep-success); }}
.tone-warning .ep-icon, .ep-icon.tone-warning {{ background: rgba(251,191,36,.14); color: var(--ep-warning); }}
.tone-danger .ep-icon, .ep-icon.tone-danger {{ background: rgba(251,113,133,.14); color: var(--ep-danger); }}

/* KPI grid */
.ep-kpi-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(138px, 1fr)); gap:12px; }}
.ep-kpi-grid.cols-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.ep-kpi {{ background:var(--ep-card); border:1px solid var(--ep-border); border-radius:12px; padding:14px 16px;
  transition: border-color .15s ease; }}
.ep-kpi:hover {{ border-color: #4a5275; }}
.ep-kpi-label {{ display:flex; align-items:center; gap:9px; color:var(--ep-muted); font-size:.84rem; font-weight:500; }}
.ep-kpi-value {{ font-size:1.65rem; font-weight:700; color:var(--ep-text); margin-top:10px;
  font-variant-numeric: tabular-nums; letter-spacing:-.01em; }}
.ep-kpi-sub {{ color:var(--ep-muted); font-size:.76rem; margin-top:2px; }}
.ep-tip {{ margin-left:auto; position:relative; display:inline-flex; color:var(--ep-muted); cursor:help;
  border-radius:50%; outline:none; }}
.ep-tip:hover, .ep-tip:focus-visible {{ color: var(--ep-violet); }}
.ep-tip:hover::after, .ep-tip:focus-visible::after {{
  content: attr(data-tip); position:absolute; right:-6px; top:24px; width: 230px; z-index: 50;
  padding:9px 11px; border-radius:9px; background: var(--ep-elev); border:1px solid var(--ep-border);
  color: var(--ep-text); font-size:.78rem; font-weight:400; line-height:1.4; white-space:normal;
  box-shadow: 0 10px 30px rgba(0,0,0,.4); }}

/* Dataset summary */
.ep-dataset {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap:16px; align-items:center; }}
.ep-field-label {{ color:var(--ep-muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; }}
.ep-field-value {{ font-weight:600; color:var(--ep-text); margin-top:3px; overflow-wrap:anywhere; }}

/* Badges */
.ep-badge {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px;
  font-size:.74rem; font-weight:600; white-space:nowrap; }}
.badge-success {{ background: rgba(52,211,153,.14); color: var(--ep-success); }}
.badge-warning {{ background: rgba(251,191,36,.14); color: var(--ep-warning); }}
.badge-danger  {{ background: rgba(251,113,133,.14); color: var(--ep-danger); }}
.badge-info    {{ background: rgba(56,189,248,.14); color: var(--ep-cyan); }}
.badge-primary {{ background: rgba(118,80,232,.22); color: var(--ep-violet); }}
.badge-neutral {{ background: var(--ep-elev); color: var(--ep-muted); }}

/* Model cards */
.ep-model {{ background:var(--ep-card); border:1px solid var(--ep-border); border-radius:12px; padding:14px 15px;
  margin-bottom: 4px; box-sizing: border-box; transition: border-color .15s ease, background .15s ease; }}
.ep-model:hover {{ border-color:#4a5275; }}
.ep-model.selected {{ border-color: var(--ep-primary); box-shadow: inset 0 0 0 1px var(--ep-primary);
  background: linear-gradient(180deg, rgba(118,80,232,.18), rgba(118,80,232,.04)), var(--ep-card); }}
.ep-model-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:6px; }}
.ep-model-name {{ font-weight:600; margin-top:10px; color:var(--ep-text); line-height:1.25;
  min-height: 2.5em; display:flex; align-items:flex-start; }}
.ep-model-acc {{ font-size:1.25rem; font-weight:700; margin-top:4px; font-variant-numeric: tabular-nums; }}
.ep-model-acc span {{ font-size:.74rem; color:var(--ep-muted); font-weight:500; margin-left:4px; }}

/* Insight cards */
.ep-insight {{ background:var(--ep-card); border:1px solid var(--ep-border); border-left:3px solid var(--ep-cyan);
  border-radius:12px; padding:16px 18px; margin-bottom:12px; }}
.ep-insight.kind-action {{ border-left-color: var(--ep-teal); }}
.ep-insight.kind-caution {{ border-left-color: var(--ep-warning); }}
.ep-insight h4 {{ margin: 8px 0 6px; padding:0; font-size:1.02rem; font-weight:600; color:var(--ep-text); }}
.ep-insight p {{ margin:0; color:var(--ep-text); line-height:1.5; }}
.ep-why {{ color:var(--ep-muted) !important; font-size:.86rem; margin-top:8px !important; }}

.ep-note {{ color:var(--ep-muted); font-size:.84rem; line-height:1.5; }}
.ep-note b {{ color: var(--ep-text); }}
.ep-empty {{ text-align:center; padding:34px 18px; border:1px dashed var(--ep-border); border-radius:12px; color:var(--ep-muted); }}
.ep-empty .ep-icon {{ margin-bottom:10px; }}
.ep-kv {{ display:grid; grid-template-columns: max-content 1fr; gap:6px 14px; font-size:.88rem; }}
.ep-kv dt {{ color: var(--ep-muted); }} .ep-kv dd {{ margin:0; color: var(--ep-text); font-weight:500; overflow-wrap:anywhere; }}

/* Streamlit widgets */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px; }}
div[data-testid="stFileUploaderDropzone"] {{ background: var(--ep-elev); border: 1px dashed #4a5275; }}
button[kind="primary"] {{ background: linear-gradient(135deg, var(--ep-primary), #6342d6); border: none; }}
:focus-visible {{ outline: 2px solid var(--ep-violet) !important; outline-offset: 2px; }}

@media (max-width: 900px) {{
  .ep-hero h1 {{ font-size: 1.7rem; }}
}}
@media (min-width: 641px) and (max-width: 1180px) {{
  .ep-model-name {{ min-height: 3.75em; }}
}}
@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 1rem; }}
  .ep-hero {{ padding: 18px; }}
  .ep-hero h1, .ep-hero-body h1 {{ font-size: 1.5rem; }}
  .st-key-ep_hero {{ padding: 18px; }}
  .ep-cta-text {{ align-items:flex-start; }}
  .ep-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; }}
  .ep-kpi {{ padding: 12px; }}
  .ep-kpi-value {{ font-size: 1.3rem; margin-top: 7px; }}
  .ep-tip:hover::after, .ep-tip:focus-visible::after {{ width: 180px; }}
  .ep-model-name {{ min-height: 0; }}
  .ep-page h1 {{ font-size: 1.35rem; }}
}}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def mark_active_nav(url_path):
    """Highlight the current page's sidebar link (Streamlit's own cue is very faint)."""
    st.markdown(f"""<style>
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{url_path}"] {{
      background: linear-gradient(90deg, rgba(118,80,232,.34), rgba(118,80,232,.10)) !important;
      box-shadow: inset 3px 0 0 {C['primary']};
    }}
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{url_path}"] span {{
      color: {C['text']} !important; font-weight: 600;
    }}</style>""", unsafe_allow_html=True)
