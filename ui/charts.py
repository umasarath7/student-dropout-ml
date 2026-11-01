"""Altair charts styled for the EduPulse dark theme."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from model.features import CLASS_ORDER
from ui.theme import C, OUTCOME_COLORS

METRIC_TOOLTIPS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def show(chart, description):
    """Render with our own styling and an accessible text description."""
    chart = (chart.configure(background="transparent", font="Inter")
             .configure_view(stroke=None)
             .configure_axis(labelColor=C["muted"], titleColor=C["muted"], gridColor=C["border"],
                             domainColor=C["border"], tickColor=C["border"], labelFontSize=12,
                             titleFontSize=12, titleFontWeight=500, gridOpacity=0.6)
             .configure_legend(labelColor=C["text"], titleColor=C["muted"], labelFontSize=12,
                               orient="bottom", symbolType="circle")
             .configure_title(color=C["text"]))
    st.altair_chart(chart, theme=None, width="stretch", alt=description)


def comparison(table, metric, selected):
    df = table.reset_index(names="Model")
    best = df.loc[df[metric].idxmax(), "Model"]

    def role(m):
        if m == best and m == selected:
            return "Best & selected"
        return "Best" if m == best else "Selected" if m == selected else "Other models"

    df["Role"] = df["Model"].map(role)
    roles = [("Best & selected", C["teal"]), ("Best", C["teal"]),
             ("Selected", C["primary"]), ("Other models", C["slate"])]
    present = [r for r in roles if r[0] in set(df["Role"])]
    order = df.sort_values(metric, ascending=False)["Model"].tolist()
    lo = min(0.0, float(df[metric].min()))
    base = alt.Chart(df).encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=0, labelFontSize=13,
                                                                  labelColor=C["text"])),
        x=alt.X(f"{metric}:Q", title=metric, scale=alt.Scale(domain=[lo, 1]),
              axis=alt.Axis(values=[v / 4 for v in range(int(lo * 4), 5)], format=".2f")),
        tooltip=["Model:N", "Role:N"] + [alt.Tooltip(f"{m}:Q", format=".4f") for m in METRIC_TOOLTIPS],
    )
    bars = base.mark_bar(cornerRadiusEnd=5, height=26).encode(
        color=alt.Color("Role:N", title=None, scale=alt.Scale(domain=[r[0] for r in present],
                                                              range=[r[1] for r in present])))
    labels = base.mark_text(align="left", dx=6, fontSize=13, fontWeight=600, color=C["text"]).encode(
        text=alt.Text(f"{metric}:Q", format=".4f"))
    return (bars + labels).properties(height=260)


def confusion(cm):
    rows = []
    for i, actual in enumerate(CLASS_ORDER):
        total = cm[i].sum()
        for j, pred in enumerate(CLASS_ORDER):
            share = cm[i, j] / total if total else 0.0
            rows.append({"Actual": actual, "Predicted": pred, "Students": int(cm[i, j]),
                         "Share": share, "Label": f"{share:.0%} of actual",
                         "Result": "Correct" if i == j else "Mistake"})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=CLASS_ORDER, title="Predicted outcome",
                axis=alt.Axis(orient="top", labelAngle=0, labelColor=C["text"], labelFontSize=13)),
        y=alt.Y("Actual:N", sort=CLASS_ORDER, title="Actual outcome",
                axis=alt.Axis(labelColor=C["text"], labelFontSize=13)),
        tooltip=["Actual:N", "Predicted:N", "Result:N", "Students:Q",
                 alt.Tooltip("Share:Q", format=".1%", title="Share of actual row")],
    )
    rect = dict(cornerRadius=8, stroke=C["bg"], strokeWidth=4)
    correct = base.transform_filter(alt.datum.Result == "Correct").mark_rect(**rect).encode(
        color=alt.Color("Share:Q", legend=None, scale=alt.Scale(domain=[0, 1], range=["#123a37", "#14907f"])))
    wrong = base.transform_filter(alt.datum.Result == "Mistake").mark_rect(**rect).encode(
        color=alt.Color("Share:Q", legend=None, scale=alt.Scale(domain=[0, 0.6], range=["#1d2238", "#a23a55"])))
    count = base.mark_text(fontSize=18, fontWeight=700, dy=-7, color=C["text"]).encode(text="Students:Q")
    share = base.mark_text(fontSize=11, dy=12, color="#d4d4d8").encode(text="Label:N")
    return (correct + wrong + count + share).resolve_scale(color="independent").properties(height=320)


def donut(counts):
    total = int(sum(counts.values()))
    df = pd.DataFrame({"Outcome": list(counts), "Students": list(counts.values())})
    df["Share"] = df["Students"] / max(total, 1)
    arc = alt.Chart(df).mark_arc(innerRadius=72, outerRadius=112, stroke=C["bg"], strokeWidth=3).encode(
        theta=alt.Theta("Students:Q", stack=True),
        order=alt.Order("Outcome:N", sort="ascending"),
        color=alt.Color("Outcome:N", title=None, scale=alt.Scale(
            domain=CLASS_ORDER, range=[OUTCOME_COLORS[c] for c in CLASS_ORDER])),
        tooltip=["Outcome:N", "Students:Q", alt.Tooltip("Share:Q", format=".1%")],
    )
    center = alt.Chart(pd.DataFrame({"t": [f"{total:,}"], "s": ["students"]}))
    big = center.mark_text(fontSize=26, fontWeight=700, dy=-8, color=C["text"]).encode(text="t:N")
    small = center.mark_text(fontSize=12, dy=16, color=C["muted"]).encode(text="s:N")
    return (arc + big + small).properties(height=270)


def roc(roc_data):
    rows = []
    for cls, (fpr, tpr) in roc_data.items():
        rows += [{"Outcome": cls, "False positive rate": f, "True positive rate": t}
                 for f, t in zip(fpr, tpr)]
    df = pd.DataFrame(rows)
    lines = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(values=[0, .25, .5, .75, 1])),
        y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(values=[0, .25, .5, .75, 1])),
        color=alt.Color("Outcome:N", title=None, scale=alt.Scale(
            domain=list(roc_data), range=[OUTCOME_COLORS[c] for c in roc_data])),
        tooltip=["Outcome:N", alt.Tooltip("False positive rate:Q", format=".3f"),
                 alt.Tooltip("True positive rate:Q", format=".3f")],
    )
    chance = alt.Chart(pd.DataFrame({"False positive rate": [0, 1], "True positive rate": [0, 1]})).mark_line(
        strokeDash=[5, 5], color=C["muted"], opacity=0.6).encode(x="False positive rate:Q", y="True positive rate:Q")
    return (chance + lines).properties(height=320)


def probability_bars(probs):
    df = pd.DataFrame({"Outcome": CLASS_ORDER, "Probability": probs})
    base = alt.Chart(df).encode(
        y=alt.Y("Outcome:N", sort=CLASS_ORDER, title=None, axis=alt.Axis(labelColor=C["text"], labelFontSize=13)),
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=["Outcome:N", alt.Tooltip("Probability:Q", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=5, height=22).encode(
        color=alt.Color("Outcome:N", legend=None, scale=alt.Scale(
            domain=CLASS_ORDER, range=[OUTCOME_COLORS[c] for c in CLASS_ORDER])))
    text = base.mark_text(align="left", dx=6, color=C["text"], fontWeight=600).encode(
        text=alt.Text("Probability:Q", format=".1%"))
    return (bars + text).properties(height=150)


def ranked_bars(series, value_title, color=None, fmt=".3f", signed=False):
    df = series.rename(value_title).rename_axis("Feature").reset_index()
    if signed:
        df["Direction"] = np.where(df[value_title] >= 0, "Towards this outcome", "Away from this outcome")
        color_enc = alt.Color("Direction:N", title=None, scale=alt.Scale(
            domain=["Towards this outcome", "Away from this outcome"], range=[C["violet"], C["cyan"]]))
    else:
        color_enc = alt.value(color or C["violet"])
    order = df.reindex(df[value_title].abs().sort_values(ascending=False).index)["Feature"].tolist()
    base = alt.Chart(df).encode(
        y=alt.Y("Feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=320, labelColor=C["text"])),
        x=alt.X(f"{value_title}:Q", title=value_title),
        tooltip=["Feature:N", alt.Tooltip(f"{value_title}:Q", format=fmt)],
    )
    return base.mark_bar(cornerRadiusEnd=4, height=16).encode(color=color_enc).properties(
        height=max(180, 26 * len(df)))


def class_distribution(counts):
    df = pd.DataFrame({"Outcome": list(counts.index), "Students": list(counts.values)})
    base = alt.Chart(df).encode(
        x=alt.X("Outcome:N", sort=CLASS_ORDER, title=None, axis=alt.Axis(labelAngle=0, labelColor=C["text"])),
        y=alt.Y("Students:Q"), tooltip=["Outcome:N", "Students:Q"])
    bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, width=56).encode(
        color=alt.Color("Outcome:N", legend=None, scale=alt.Scale(
            domain=CLASS_ORDER, range=[OUTCOME_COLORS[c] for c in CLASS_ORDER])))
    text = base.mark_text(dy=-9, color=C["text"], fontWeight=600).encode(text="Students:Q")
    return (bars + text).properties(height=240)
