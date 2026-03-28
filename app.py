"""
app.py
ThreatScout — Streamlit Dashboard

Run with:  streamlit run app.py
"""

import time
import random
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# -- project imports ----------------------------------------------------------
from simulation.agents    import create_employees
from simulation.behaviour import update_behaviour, inject_insider
from core.anomaly         import build_and_train_detector
from core.risk            import update_risk_scores, get_alerts

from core.fuzzy import (
    anomaly_low, anomaly_medium, anomaly_high,
    deviation_low, deviation_medium, deviation_high
)
# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="ThreatScout",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.5px; }

  .risk-high   { color: #ff4b4b; font-weight: 700; }
  .risk-medium { color: #ffa500; font-weight: 700; }
  .risk-low    { color: #00cc88; font-weight: 700; }

  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
  }

  .alert-box {
    background: rgba(255, 75, 75, 0.12);
    border-left: 4px solid #ff4b4b;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14px;
  }

  .explanation-box {
    background: rgba(0, 204, 136, 0.06);
    border-left: 4px solid #00cc88;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.6;
  }

  div[data-testid="stSidebar"] { background: #0d0f14; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session state initialisation
# =============================================================================
def init_state():
    if "employees" not in st.session_state:
        employees = create_employees(n=15)
        st.session_state.employees   = employees
        st.session_state.step        = 0
        st.session_state.insider     = None
        st.session_state.history     = []   # list of risk_results per step
        st.session_state.running     = False

        # Train anomaly detector on clean initial behaviour
        update_behaviour(employees, step=0)
        detector = build_and_train_detector(employees)
        st.session_state.detector    = detector

init_state()

employees = st.session_state.employees
detector  = st.session_state.detector


# =============================================================================
# Sidebar controls
# =============================================================================
with st.sidebar:
    st.markdown("## 🛡️ ThreatScout")
    st.caption("Explainable Insider Threat Detection")
    st.divider()

    if st.button("▶ Advance One Step", use_container_width=True):
        st.session_state.step += 1
        step = st.session_state.step
        update_behaviour(employees, step=step)
        results = update_risk_scores(employees, detector)
        st.session_state.history.append(results)

    if st.button("💉 Inject Insider", use_container_width=True,
                 disabled=st.session_state.insider is not None):
        insider = inject_insider(employees)
        st.session_state.insider = insider
        st.success(f"Insider injected: {insider.id} ({insider.role})")

    if st.button("🔄 Reset Simulation", use_container_width=True):
        for key in ["employees", "step", "insider", "history", "detector", "running"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.markdown(f"**Step:** {st.session_state.step}")
    if st.session_state.insider:
        ins = st.session_state.insider
        st.markdown(
            f"**Insider:** <span class='risk-high'>{ins.id} ({ins.role})</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("**Insider:** None")


# =============================================================================
# Header
# =============================================================================
st.markdown("# 🛡️ ThreatScout")
st.caption("Explainable Hybrid Insider Threat Detection · Fuzzy Behavioral Fingerprinting")
st.divider()


# =============================================================================
# If no steps run yet — show onboarding
# =============================================================================
if not st.session_state.history:
    st.info("⬅️ Click **Advance One Step** in the sidebar to start the simulation.")
    st.stop()

# Latest risk results
latest = st.session_state.history[-1]
alerts = get_alerts(latest)


# =============================================================================
# KPI row
# =============================================================================
col1, col2, col3, col4 = st.columns(4)
total      = len(latest)
high_risk  = sum(1 for r in latest if r["risk_label"] == "High")
med_risk   = sum(1 for r in latest if r["risk_label"] == "Medium")
avg_score  = sum(r["final_score"] for r in latest) / total

with col1:
    st.metric("Employees Monitored", total)
with col2:
    st.metric("🔴 High Risk",   high_risk,  delta=None)
with col3:
    st.metric("🟠 Medium Risk", med_risk,   delta=None)
with col4:
    st.metric("Avg Risk Score", f"{avg_score:.2f}")

st.divider()


# =============================================================================
# Alerts banner
# =============================================================================
if alerts:
    st.markdown("### 🚨 Active Alerts")
    for a in alerts:
        st.markdown(
            f"<div class='alert-box'>⚠️ <b>{a['employee_id']}</b> ({a['role']}) — "
            f"Risk: <b>{a['risk_label']}</b> · Score: {a['final_score']:.2f}</div>",
            unsafe_allow_html=True,
        )
    st.write("")


# =============================================================================
# Main layout: Employee table  |  Selected employee detail
# =============================================================================
left, right = st.columns([1, 1.6], gap="large")


# ---- LEFT: Employee risk table --------------------------------------------
with left:
    st.markdown("### 👥 Employee Risk Board")

    # Build a display DataFrame
    rows = []
    for r in sorted(latest, key=lambda x: x["final_score"], reverse=True):
        label = r["risk_label"]
        rows.append({
            "ID":       r["employee_id"],
            "Role":     r["role"],
            "Risk":     label,
            "Score":    r["final_score"],
            "Anomaly":  r["anomaly_score"],
            "Deviation":r["deviation_score"],
        })
    df = pd.DataFrame(rows)

    def colour_risk(val):
        colours = {"High": "color: #ff4b4b", "Medium": "color: #ffa500", "Low": "color: #00cc88"}
        return colours.get(val, "")

    st.dataframe(
        df.style.applymap(colour_risk, subset=["Risk"])
                .format({"Score": "{:.3f}", "Anomaly": "{:.3f}", "Deviation": "{:.3f}"}),
        use_container_width=True,
        height=420,
    )

    # Employee selector for detail view
    emp_ids = [r["employee_id"] for r in latest]
    selected_id = st.selectbox("Select employee for detail view", emp_ids,
                                key="selected_emp")


# ---- RIGHT: Detail panel ---------------------------------------------------
with right:
    selected = next((r for r in latest if r["employee_id"] == selected_id), None)

    if selected:
        st.markdown(f"### 🔍 {selected['employee_id']} · {selected['role']}")

        label_colour = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
        st.markdown(
            f"Risk: <span class='{label_colour[selected['risk_label']]}'>"
            f"{selected['risk_label']}</span> · Score: **{selected['final_score']:.3f}**",
            unsafe_allow_html=True,
        )
        st.write("")

        # ---- Radar chart ---------------------------------------------------
        features = ["after_hours_login", "file_access", "usb_usage", "job_sites"]
        labels   = ["After-Hours Login", "File Access", "USB Usage", "Job Sites"]

        SERIES = [
            ("Baseline",  "#4488ff", "rgba(68,136,255,0.12)",  "baseline"),
            ("Current",   "#00cc88", "rgba(0,204,136,0.12)",   "current"),
            ("Deviation", "#ff4b4b", "rgba(255,75,75,0.12)",   "deviation"),
        ]

        fig_radar = go.Figure()
        for series_name, line_colour, fill_colour, key in SERIES:
            vals = [selected[key][f] for f in features]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=fill_colour,
                line=dict(color=line_colour, width=2),
                name=series_name,
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="rgba(255,255,255,0.1)",
                                linecolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                                 linecolor="rgba(255,255,255,0.1)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=10, b=40, l=40, r=40),
            height=300,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ---- Explanation panel -------------------------------------------
        st.markdown("#### Fuzzy Explanation")
        explanation = selected["explanation"]

        # Remove markdown
        explanation = explanation.replace("**", "").replace("*", "")

        # Break into lines more cleanly
        explanation = explanation.replace(". ", ".<br><br>")

        # Highlight key parts
        explanation = explanation.replace("Risk assessed as", "<b>Risk assessed as</b>")
        explanation = explanation.replace("Dominant rule:", "<br><b>Dominant rule:</b>")

        st.markdown(
            f"<div class='explanation-box'>{explanation}</div>",
            unsafe_allow_html=True,
        )
        st.write("")
                # ---- Membership Function Visualization -------------------------
        st.markdown("#### Membership Functions")

        import numpy as np

        x = np.linspace(0, 1, 200)

        # ---- Anomaly Membership Plot
        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(go.Scatter(x=x, y=[anomaly_low(v) for v in x], name="Low"))
        fig_anomaly.add_trace(go.Scatter(x=x, y=[anomaly_medium(v) for v in x], name="Medium"))
        fig_anomaly.add_trace(go.Scatter(x=x, y=[anomaly_high(v) for v in x], name="High"))

        fig_anomaly.update_layout(
            title="Anomaly Score Membership",
            xaxis_title="Anomaly Score",
            yaxis_title="Membership Degree",
            height=250,
            margin=dict(t=40, b=20)
        )
        fig_anomaly.add_vline(
            x=selected["anomaly_score"],
            line_dash="dash",
            annotation_text="Current",
            annotation_position="top"
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

        # ---- Deviation Membership Plot
        fig_deviation = go.Figure()
        fig_deviation.add_trace(go.Scatter(x=x, y=[deviation_low(v) for v in x], name="Low"))
        fig_deviation.add_trace(go.Scatter(x=x, y=[deviation_medium(v) for v in x], name="Medium"))
        fig_deviation.add_trace(go.Scatter(x=x, y=[deviation_high(v) for v in x], name="High"))

        fig_deviation.update_layout(
            title="Deviation Score Membership",
            xaxis_title="Deviation Score",
            yaxis_title="Membership Degree",
            height=250,
            margin=dict(t=40, b=20)
        )
        fig_deviation.add_vline(
            x=selected["deviation_score"],
            line_dash="dash",
            annotation_text="Current",
            annotation_position="top"
        )
        st.plotly_chart(fig_deviation, use_container_width=True)
        # ---- Fired rules -------------------------------------------------
        if selected["fired_rules"]:
            st.markdown("#### Fired Rules")
            for rule in sorted(selected["fired_rules"],
                               key=lambda x: x["strength"], reverse=True):
                bar_w = int(rule["strength"] * 100)
                colour = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00cc88"}[rule["label"]]
                st.markdown(
                    f"<div style='font-size:12px; margin-bottom:6px;'>"
                    f"<b>{rule['rule']}</b> · strength {rule['strength']:.2f}<br>"
                    f"<div style='height:4px;background:rgba(255,255,255,0.1);border-radius:2px;'>"
                    f"<div style='width:{bar_w}%;height:4px;background:{colour};border-radius:2px;'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )


# =============================================================================
# Risk over time — line chart
# =============================================================================
st.divider()
st.markdown("### 📈 Risk Score History")

if len(st.session_state.history) > 1:
    # Build a step × employee matrix
    timeline_data = {}
    for step_idx, step_results in enumerate(st.session_state.history):
        for r in step_results:
            eid = r["employee_id"]
            if eid not in timeline_data:
                timeline_data[eid] = []
            timeline_data[eid].append({"step": step_idx + 1,
                                        "score": r["final_score"],
                                        "label": r["risk_label"],
                                        "role": r["role"]})

    rows = []
    for eid, entries in timeline_data.items():
        for e in entries:
            rows.append({"Employee": eid, "Step": e["step"],
                         "Score": e["score"], "Role": e["role"]})

    df_time = pd.DataFrame(rows)
    insider_id = st.session_state.insider.id if st.session_state.insider else None

    fig_line = px.line(
        df_time, x="Step", y="Score", color="Employee",
        line_dash="Employee",
        color_discrete_map={insider_id: "#ff4b4b"} if insider_id else {},
        labels={"Score": "Risk Score", "Step": "Simulation Step"},
        height=320,
    )
    fig_line.update_traces(line_width=1.5, opacity=0.7)
    if insider_id:
        fig_line.update_traces(
            selector=dict(name=insider_id),
            line_width=3, opacity=1.0,
        )

    fig_line.add_hline(y=0.65, line_dash="dot", line_color="#ff4b4b",
                       annotation_text="Alert threshold", annotation_position="right")
    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", range=[0, 1.05]),
        legend=dict(orientation="v", font_size=10),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.caption("Advance more steps to see the risk timeline.")


# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption("ThreatScout · Explainable Hybrid Insider Threat Detection · "
           "Isolation Forest + Fuzzy Behavioral Fingerprinting")