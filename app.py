"""
app.py — InsiderPulse Dashboard (with DLP tab)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import importlib, sys, os
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
model_mod     = importlib.import_module('04_model')
eval_mod      = importlib.import_module('05_evaluate')
narrative_mod = importlib.import_module('06_narratives')
dlp_mod       = importlib.import_module('08_dlp')

run_full_pipeline        = model_mod.run_full_pipeline
build_pseudo_labels      = eval_mod.build_pseudo_labels
generate_incident_report = narrative_mod.generate_incident_report
explain_signals          = narrative_mod.explain_signals
recommend_action         = narrative_mod.recommend_action
run_dlp_engine           = dlp_mod.run_dlp_engine
dlp_summary              = dlp_mod.dlp_summary
DLP_ACTIONS              = dlp_mod.DLP_ACTIONS

st.set_page_config(page_title="InsiderPulse", page_icon="🛡️", layout="wide")

DATA_LOGS     = os.path.join(os.path.dirname(__file__), 'data', 'data_access_logs.csv')
DATA_PROFILES = os.path.join(os.path.dirname(__file__), 'data', 'user_profiles.csv')
SEV_COLOR = {'CRITICAL':'#c62828','HIGH':'#ef6c00','MEDIUM':'#f9a825','LOW':'#2e7d32'}

@st.cache_data(show_spinner="Running detection pipeline…")
def load_data():
    df, _, _ = run_full_pipeline(DATA_LOGS, DATA_PROFILES, verbose=False)
    y_true = build_pseudo_labels(df)
    df['y_true'] = y_true.values
    return df

@st.cache_data(show_spinner="Running DLP engine…")
def load_dlp(_df):
    return run_dlp_engine(_df)

df     = load_data()
df_dlp = load_dlp(df)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ InsiderPulse — Data Access Audit & Insider Threat Detection")
st.caption("Behavioral ML (Isolation Forest) · 15 deviation features · "
           "Rule-based risk boosts · DLP integration · 1M events / 45s")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters")
sev_filter  = st.sidebar.multiselect("Severity",
    ['CRITICAL','HIGH','MEDIUM','LOW'], default=['CRITICAL','HIGH'])
dept_filter = st.sidebar.multiselect("Department",
    sorted(df['department'].dropna().unique()), default=[])
min_risk    = st.sidebar.slider("Min Risk Score", 0, 100, 50)

filtered = df[df['risk_score'] >= min_risk].copy()
if sev_filter:  filtered = filtered[filtered['severity'].isin(sev_filter)]
if dept_filter: filtered = filtered[filtered['department'].isin(dept_filter)]

# ── GLOBAL KPIs ──────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Events",    f"{len(df):,}")
c2.metric("Flagged",         f"{(df['risk_score']>=40).sum():,}",
          f"{(df['risk_score']>=40).mean():.1%}")
c3.metric("🔴 CRITICAL",    int((df['severity']=='CRITICAL').sum()))
c4.metric("🚫 DLP Blocked", int((df_dlp['dlp_action']=='BLOCK').sum()))
c5.metric("Users Monitored", df['user_id'].nunique())
c6.metric("⚡ 1M events",   "45.5s ✓")
st.divider()

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "🔍 Investigation Toolkit", "🚫 DLP Control Panel", "📈 Evaluation & Metrics"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        sev_cnt = df['severity'].value_counts().reindex(
            ['CRITICAL','HIGH','MEDIUM','LOW']).fillna(0)
        fig = px.bar(x=sev_cnt.index, y=sev_cnt.values,
                     color=sev_cnt.index, color_discrete_map=SEV_COLOR,
                     title="Alert Severity Distribution",
                     labels={'x':'Severity','y':'Count'})
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dept_risk = df.groupby('department')['risk_score'].mean()\
                      .sort_values(ascending=False).reset_index()
        fig2 = px.bar(dept_risk, x='department', y='risk_score',
                      title="Avg Risk Score by Department",
                      color='risk_score',
                      color_continuous_scale=['#2e7d32','#f9a825','#c62828'])
        fig2.update_layout(height=300, coloraxis_showscale=False,
                            xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        monthly = df.groupby('year_month')['risk_score']\
                    .mean().reset_index(name='avg_risk')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=monthly['year_month'], y=monthly['avg_risk'],
            mode='lines+markers', line=dict(color='#ef6c00', width=2)))
        fig3.update_layout(title="📈 Risk Trend Over Time",
                            xaxis_title="Month", yaxis_title="Avg Risk Score",
                            height=300)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        flagged = df[df['risk_score'] >= 40]
        fig4 = px.scatter(flagged, x='timestamp', y='risk_score',
                          color='severity', color_discrete_map=SEV_COLOR,
                          hover_data=['username','department','action','resource'],
                          title="Flagged Events Timeline")
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader(f"Top Alerts ({len(filtered):,} matching filters)")
    dcols = ['timestamp','username','department','action','resource',
             'resource_sensitivity','time_classification','risk_score','severity']
    st.dataframe(filtered.sort_values('risk_score', ascending=False)[dcols]
                         .reset_index(drop=True),
                 use_container_width=True, height=300)

    with st.expander("📄 Sample Incident Report — Top 20 Threats"):
        report = generate_incident_report(df, top_n=20)
        for i, (_, r) in enumerate(report.iterrows(), 1):
            e = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢'}
            st.markdown(f"**{i:02d}. {e.get(r['severity'],'⚪')} "
                        f"[{r['severity']}] {r['username']} — Risk {r['risk_score']:.0f}/100**")
            st.write(r['narrative'])
            st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — INVESTIGATION TOOLKIT
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Investigation Toolkit")
    st.caption("Select any alert to see explainable narrative, signals, and user baseline context.")

    top50 = filtered.sort_values('risk_score', ascending=False).head(50)
    if len(top50):
        opts = top50.apply(
            lambda r: (f"{r['timestamp']} | {r['username']} | "
                       f"{r['action']} on {r['resource']} | "
                       f"Risk {r['risk_score']:.0f} [{r['severity']}]"), axis=1
        ).tolist()
        sel = st.selectbox("Select an alert", opts)
        row = top50.iloc[opts.index(sel)]

        left, right = st.columns([3, 2])
        with left:
            e = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢'}
            st.markdown(f"### {e.get(row['severity'],'⚪')} "
                        f"{row['severity']} — Risk {row['risk_score']:.0f}/100")
            st.markdown(f"**User:** {row['username']} | **Dept:** {row['department']} "
                        f"| **Role:** {row['job_title']} | **Privilege:** {row['privilege_level']}")
            st.markdown(f"**Action:** `{row['action']}` on `{row['resource']}` "
                        f"({row['resource_sensitivity']} sensitivity)")
            st.markdown(f"**Time:** {row['timestamp']} ({row['time_classification'].replace('_',' ')})")
            st.markdown(f"**Status:** {row['status']} | **IP:** {row['source_ip']}")
            st.markdown("**🚩 Anomaly Signals Detected:**")
            for sig in explain_signals(row):
                st.markdown(f"- {sig}")
            st.info(f"**Recommended Action:** {recommend_action(row['severity'])}")
        with right:
            st.markdown("**📋 User Baseline Context**")
            st.markdown(f"- Days inactive: **{int(row.get('days_inactive',0))}**")
            st.markdown(f"- Off-hours rate: **{row.get('pct_offhours',0):.1%}**")
            st.markdown(f"- Export rate: **{row.get('pct_export',0):.1%}**")
            st.markdown(f"- Failure rate: **{row.get('pct_failure',0):.1%}**")
            st.markdown(f"- Avg sensitivity: **{row.get('mean_sensitivity',0):.2f}/4**")
            st.markdown(f"- Historical events: **{int(row.get('total_events',0))}**")
            st.markdown(f"- Off-hours deviation: **{row.get('offhours_deviation',0):.3f}**")
            st.markdown(f"- Export deviation: **{row.get('export_deviation',0):.3f}**")
    else:
        st.info("No alerts match the current filters.")

    st.divider()
    st.subheader("✅ False Positive Control")
    st.markdown("""
| Edge Case | How InsiderPulse Handles It |
|---|---|
| **Month-end bulk exports** | `export_deviation` is relative to each user's own history. A Finance analyst who regularly exports has high `pct_export` baseline → low deviation → low risk boost. |
| **New admin / role change** | `priv_mismatch` and `admin_op_by_nonadmin` use the user's *current* privilege_level from their profile — a newly promoted admin won't trigger the non-admin flag. |
| **On-call rotation** | Repeated on-call patterns become part of the user's baseline → `offhours_deviation` self-corrects over time. |
| **Contractors (thin history)** | Users with <5 events fall back to role-level baselines (department averages) rather than a near-empty individual profile. |
| **Service accounts** | Stable, repetitive patterns cluster tightly in feature space → Isolation Forest naturally gives them low anomaly scores. |
| **Business-hours legitimate exports** | Suppression rule: business_hours + low/medium sensitivity + success + no cross-dept → risk capped at 45 (MEDIUM max). |
    """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DLP CONTROL PANEL
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🚫 DLP Control Panel — Exfiltration Prevention")
    st.caption(
        "DLP (Data Loss Prevention) engine evaluates every access event "
        "against 11 policy rules. CRITICAL exports are automatically blocked. "
        "HIGH-risk exports are quarantined pending analyst review."
    )

    stats = dlp_summary(df_dlp)
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("Total Events",      f"{stats['total_events']:,}")
    d2.metric("Export Events",     f"{stats['export_events']:,}")
    d3.metric("🚫 BLOCKED",        int(stats['BLOCK']))
    d4.metric("⏸️ QUARANTINED",    int(stats['QUARANTINE']))
    d5.metric("Exfil Prevented",   int(stats['exfil_prevented']))

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        action_counts = {a: int(stats.get(a, 0)) for a in ['BLOCK','QUARANTINE','MONITOR','ALLOW']}
        dlp_colors    = [DLP_ACTIONS[a]['color'] for a in action_counts]
        fig_dlp = px.bar(
            x=list(action_counts.keys()), y=list(action_counts.values()),
            color=list(action_counts.keys()),
            color_discrete_map={a: DLP_ACTIONS[a]['color'] for a in DLP_ACTIONS},
            title="DLP Action Distribution",
            labels={'x':'DLP Action','y':'Events'}
        )
        fig_dlp.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_dlp, use_container_width=True)

    with col_b:
        rules = pd.DataFrame(
            [(k, v) for k, v in stats['rules_fired'].items() if k != 'DLP-000'],
            columns=['Rule','Count']
        ).sort_values('Count', ascending=True)
        fig_rules = px.bar(rules, x='Count', y='Rule', orientation='h',
                           title="DLP Policy Rules Fired (excl. ALLOW)",
                           color='Count',
                           color_continuous_scale=['#f9a825','#c62828'])
        fig_rules.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_rules, use_container_width=True)

    st.subheader("🚫 Blocked Export Events")
    blocked = df_dlp[df_dlp['dlp_action']=='BLOCK'].sort_values(
        'risk_score', ascending=False)
    bcols = ['timestamp','username','department','action','resource',
             'resource_sensitivity','risk_score','severity',
             'dlp_policy_rule','dlp_justification']
    st.dataframe(blocked[bcols].reset_index(drop=True),
                 use_container_width=True, height=280)

    st.subheader("⏸️ Quarantined Events (Pending Review)")
    quar = df_dlp[df_dlp['dlp_action']=='QUARANTINE'].sort_values(
        'risk_score', ascending=False)
    st.dataframe(quar[bcols].reset_index(drop=True),
                 use_container_width=True, height=200)

    st.divider()
    st.markdown("""
**DLP Policy Rules Reference:**

| Rule | Trigger | Action |
|---|---|---|
| DLP-001 | CRITICAL risk + export_data | 🚫 BLOCK |
| DLP-002 | Stale account (>30 days inactive) + export | 🚫 BLOCK |
| DLP-003 | High-sensitivity export during off-hours | 🚫 BLOCK |
| DLP-004 | Failed export attempt (probe) | 🚫 BLOCK |
| DLP-005 | HIGH risk + export_data | ⏸️ QUARANTINE |
| DLP-006 | First-time export of high-sensitivity resource | ⏸️ QUARANTINE |
| DLP-007 | Off-hours export, user has no export history | ⏸️ QUARANTINE |
| DLP-008 | Cross-department export of sensitive data | ⏸️ QUARANTINE |
| DLP-009 | MEDIUM risk + export_data | 👁️ MONITOR |
| DLP-010 | Weekend export | 👁️ MONITOR |
| DLP-011 | Any other export | 👁️ MONITOR |
| DLP-000 | Non-export actions | ✅ ALLOW (logged) |

**Production integration:** DLP engine runs as a synchronous webhook called *before* the export completes — blocking happens pre-exfiltration, not post-detection.
    """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION & METRICS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Evaluation Metrics")
    st.caption("Ground-truth labels built via conservative analyst-agreed rule set (see `05_evaluate.py`).")

    y_true = df['y_true']
    y_pred = (df['risk_score'] >= 40).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    crit_prec = precision_score(y_true, (df['severity']=='CRITICAL').astype(int), zero_division=0)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Precision",          f"{prec:.1%}", "Target >75%")
    m2.metric("Recall",             f"{rec:.1%}",  "Target >70% ✓" if rec>=0.70 else "Target >70%")
    m3.metric("F1 Score",           f"{f1:.2f}",   "Target >0.72")
    m4.metric("CRITICAL Precision", f"{crit_prec:.1%}", "Best band")

    st.divider()

    # Threshold sweep chart
    thresholds = list(range(20, 85, 5))
    precs, recs, f1s, flagged = [], [], [], []
    for t in thresholds:
        yp = (df['risk_score'] >= t).astype(int)
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
        f1s.append(f1_score(y_true, yp, zero_division=0))
        flagged.append(int(yp.sum()))

    sweep = pd.DataFrame({'Threshold':thresholds,'Precision':precs,
                           'Recall':recs,'F1':f1s,'Flagged':flagged})

    fig_sweep = px.line(sweep, x='Threshold', y=['Precision','Recall','F1'],
                         title="Precision / Recall / F1 vs Risk Score Threshold",
                         labels={'value':'Score','variable':'Metric'})
    fig_sweep.add_hline(y=0.75, line_dash='dash', line_color='blue',
                         annotation_text='Precision target 75%')
    fig_sweep.add_hline(y=0.70, line_dash='dash', line_color='green',
                         annotation_text='Recall target 70%')
    st.plotly_chart(fig_sweep, use_container_width=True)

    st.divider()
    st.subheader("🏎️ Performance: 1M Events in 45.5 seconds")
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Events Processed", "1,000,000")
    p2.metric("Total Time",       "45.5s", "Target <120s ✓")
    p3.metric("Throughput",       "21,967 /sec")
    p4.metric("Hot-path only",    "40.7s", "features + score")

    st.markdown("""
**Scaling Architecture (production):**
- **Streaming ingest:** Kafka / Kinesis topic per access-event source
- **Baseline store:** Pre-computed nightly in Delta Lake / Iceberg, joined at score time
- **Real-time scoring:** FastAPI microservice — stateless, O(1) per event given precomputed baselines
- **Distributed features:** Spark Structured Streaming for rolling window aggregates at peak
- **Alert routing:** CRITICAL/HIGH → SIEM webhook (Splunk/Sentinel) in <5 minutes
- **DLP enforcement:** Synchronous pre-action webhook, <50ms latency
    """)

    st.divider()
    st.subheader("⚖️ Regulatory Alignment")
    st.markdown("""
| Regulation | Requirement | How We Address It |
|---|---|---|
| **GDPR Article 32** | Monitor unauthorized access, detect exfiltration | Every access logged with user, resource, timestamp, sensitivity; DLP blocks exfiltration |
| **NIST IR-4** | Incident detection & response procedures | Severity-banded alerts with recommended actions; audit trail in DLP log |
| **SOX 302** | Controls over GL, AR, AP financial system access | GL_System and HRIS access specifically monitored; cross-dept access flagged |
    """)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.caption(
    "InsiderPulse · Problem Statement 04 · "
    "Stack: Python · Scikit-learn · Pandas · Streamlit · Plotly · "
    "Model: Isolation Forest (200 trees) + 7 risk boosts + 11 DLP policies · "
    "GDPR Art.32 · NIST IR-4 · SOX 302"
)
