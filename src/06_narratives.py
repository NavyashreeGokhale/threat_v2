"""
06_narratives.py — Investigation Toolkit + Incident Report
===========================================================
Deliverable: "Investigation toolkit" + "Sample incident report (10-15 threats)"
Rubric:      Risk Scoring (25 pts) — "explanations clear"

Every flagged event gets a plain-English narrative explaining exactly
WHY it was flagged, designed for a non-technical security analyst.

Architecture note:
  Template-based for hackathon reliability.
  build_llm_prompt() generates a ready-to-use prompt for Claude/GPT —
  swap build_narrative() for an API call in production.
"""

import pandas as pd
import importlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
model_mod = importlib.import_module('04_model')
run_full_pipeline = model_mod.run_full_pipeline


# ── Signal explanations ───────────────────────────────────────────────────────

def explain_signals(row: pd.Series) -> list[str]:
    """Return list of plain-English reasons this event was flagged."""
    reasons = []

    # Time-based signals
    if row['is_offhours'] == 1:
        tc = row['time_classification'].replace('_', ' ')
        dev = row.get('offhours_deviation', 0)
        if dev > 0.5:
            reasons.append(
                f"Accessed during {tc} — this user only works off-hours "
                f"{(1-dev)*100:.0f}% of the time, making this highly unusual"
            )
        else:
            reasons.append(f"Access during {tc}")

    # Export signals
    if row['is_export'] == 1:
        dev = row.get('export_deviation', 0)
        if dev > 0.5:
            reasons.append(
                f"Data export by a user who rarely exports "
                f"(historical export rate: {(1-dev)*100:.0f}%)"
            )
        else:
            reasons.append("Data export action")

    # Sensitivity signals
    sens = row.get('resource_sensitivity', '')
    if sens in ('high', 'restricted'):
        reasons.append(
            f"Target resource '{row['resource']}' is classified {sens}-sensitivity"
        )
    above = row.get('sensitivity_above_base', 0)
    if above > 0.5:
        reasons.append(
            f"Accessing data {above:.1f} sensitivity levels above this "
            f"user's typical access level"
        )

    # First-time access
    if row.get('is_first_time_resource', 0) == 1:
        reasons.append(
            f"First time this user has ever accessed '{row['resource']}'"
        )

    # Privilege signals
    if row.get('priv_mismatch', 0) == 1:
        reasons.append(
            f"Privilege mismatch: '{row['privilege_level']}'-tier user "
            f"accessing {sens}-sensitivity data"
        )

    if row.get('admin_op_by_nonadmin', 0) == 1:
        reasons.append(
            f"Administrative operation performed by non-admin account "
            f"(privilege: {row['privilege_level']})"
        )

    # Cross-department
    if row.get('cross_dept_access', 0) == 1:
        reasons.append(
            f"Cross-department access: {row['department']} user "
            f"accessing '{row['resource']}', which is outside their "
            f"department's typical systems"
        )

    # Stale account
    if row.get('stale_account_active', 0) == 1:
        inactive = int(row.get('days_inactive', 0))
        reasons.append(
            f"Account was inactive for {inactive} days before this access "
            f"(dormant account reactivation risk)"
        )

    # Failed access
    if row.get('is_failure', 0) == 1:
        reasons.append(
            "Access attempt failed — possible credential probing or "
            "unauthorized access attempt"
        )

    if not reasons:
        reasons.append(
            "Statistical deviation from established behavioral pattern "
            "(Isolation Forest model detected unusual combination of features)"
        )

    return reasons


def recommend_action(severity: str) -> str:
    actions = {
        'CRITICAL': 'BLOCK + IMMEDIATE INVESTIGATION + escalate to CISO',
        'HIGH':     'INVESTIGATE within 4 hours + verify with manager',
        'MEDIUM':   'REVIEW in next analyst triage cycle (within 24 hrs)',
        'LOW':      'MONITOR — log for audit trail, no immediate action',
    }
    return actions.get(severity, 'MONITOR')


def build_narrative(row: pd.Series) -> str:
    """Single-paragraph plain-English incident narrative."""
    signals = explain_signals(row)
    signals_text = '; '.join(signals)
    return (
        f"{row['username']} ({row['department']}, {row['job_title']}, "
        f"privilege: {row['privilege_level']}) performed '{row['action']}' "
        f"on '{row['resource']}' ({row.get('resource_sensitivity','?')} sensitivity) "
        f"at {row['timestamp']}. "
        f"Risk Score: {row['risk_score']:.0f}/100 — {row['severity']}. "
        f"Anomaly signals: {signals_text}. "
        f"Recommended action: {recommend_action(row['severity'])}."
    )


def build_llm_prompt(row: pd.Series) -> str:
    """
    Production upgrade path — drop this prompt into Claude/GPT to get
    richer narratives.  The template-based narrative above implements
    the same logic deterministically for demo reliability.
    """
    signals = explain_signals(row)
    return (
        "You are a senior cybersecurity analyst. Write a concise 2-3 sentence "
        "incident narrative for the following data access event, explaining the "
        "risk to a non-technical manager. End with a clear recommended action.\n\n"
        f"User: {row['username']} | Dept: {row['department']} | "
        f"Privilege: {row['privilege_level']} | Inactive days: {row.get('days_inactive',0)}\n"
        f"Event: {row['action']} on {row['resource']} "
        f"({row.get('resource_sensitivity','?')} sensitivity)\n"
        f"Time: {row['timestamp']} ({row.get('time_classification','?').replace('_',' ')})\n"
        f"Status: {row['status']}\n"
        f"Risk score: {row['risk_score']:.0f}/100 | Severity: {row['severity']}\n"
        f"Anomaly signals detected:\n" +
        '\n'.join(f"  - {s}" for s in signals)
    )


def generate_incident_report(df_scored: pd.DataFrame,
                              top_n: int = 20) -> pd.DataFrame:
    """Generate incident report for top-N highest-risk events."""
    top = df_scored.sort_values('risk_score', ascending=False).head(top_n).copy()
    top['narrative']      = top.apply(build_narrative, axis=1)
    top['recommendation'] = top['severity'].apply(recommend_action)
    top['llm_prompt']     = top.apply(build_llm_prompt, axis=1)
    return top


def print_incident_report(report: pd.DataFrame, n: int = 15):
    print("=" * 70)
    print("  SAMPLE INCIDENT REPORT — TOP", n, "HIGHEST RISK EVENTS")
    print("=" * 70)
    for i, (_, row) in enumerate(report.head(n).iterrows(), 1):
        sev_bar = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
        print(f"\nAlert {i:02d} {sev_bar.get(row['severity'], '⚪')}  "
              f"[{row['severity']}] Risk {row['risk_score']:.0f}/100")
        print(f"  User    : {row['username']} ({row['department']}, {row['privilege_level']})")
        print(f"  Action  : {row['action']} on {row['resource']} "
              f"({row.get('resource_sensitivity','?')} sensitivity)")
        print(f"  Time    : {row['timestamp']} ({row.get('time_classification','?').replace('_',' ')})")
        print(f"  Signals :")
        for s in explain_signals(row):
            print(f"    • {s}")
        print(f"  Action  : {row['recommendation']}")
        print("-" * 70)


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_scored, _, _ = run_full_pipeline(
        '../data/data_access_logs.csv',
        '../data/user_profiles.csv',
        verbose=False
    )

    report = generate_incident_report(df_scored, top_n=20)
    print_incident_report(report, n=15)

    save_cols = [c for c in report.columns
                 if c not in ('resource_set', '_resource_set', 'llm_prompt')]
    report[save_cols].to_csv('../output/06_incident_report.csv', index=False)

    # Save LLM prompts separately
    report[['user_id', 'username', 'timestamp', 'risk_score', 'severity',
            'llm_prompt']].to_csv('../output/06_llm_prompts.csv', index=False)

    print("\nSaved: output/06_incident_report.csv")
    print("Saved: output/06_llm_prompts.csv")
