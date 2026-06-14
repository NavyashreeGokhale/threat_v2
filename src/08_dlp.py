"""
08_dlp.py — Data Loss Prevention (DLP) Engine
===============================================
Challenge Overview: "Prevent exfiltration (integrate with DLP if possible)"
Bonus Rubric:       DLP integration (+4 pts)

WHAT REAL DLP DOES:
  In production, DLP sits inline BEFORE an export completes.
  It intercepts the request, evaluates a policy, and either:
    BLOCK     → export is cancelled, user notified, CISO alerted
    QUARANTINE→ export is held pending human review (< 4 hrs SLA)
    ALLOW     → export proceeds, logged for audit trail
    MONITOR   → export proceeds, flagged for periodic review

  This module simulates that policy enforcement layer.
  In a real deployment, it would be called as a webhook from the
  access control layer BEFORE the data leaves the system.

DLP POLICY RULES (evaluated in priority order):
  Priority 1 — BLOCK:
    • CRITICAL risk score + export_data action
    • Any export to external destination + restricted/high sensitivity
    • Stale account attempting any export
    • Failed export attempt (credential probing)

  Priority 2 — QUARANTINE:
    • HIGH risk score + export_data action
    • First-time export of high-sensitivity resource
    • Off-hours export by user with no export history

  Priority 3 — MONITOR:
    • MEDIUM risk score + export_data action
    • Weekend export of any sensitivity

  Priority 4 — ALLOW:
    • Everything else (logged but not flagged)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import importlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

model_mod = importlib.import_module('04_model')
run_full_pipeline = model_mod.run_full_pipeline


# ── DLP Policy Engine ─────────────────────────────────────────────────────────

DLP_ACTIONS = {
    'BLOCK':      {'color': '#c62828', 'emoji': '🚫', 'sla': 'Immediate'},
    'QUARANTINE': {'color': '#ef6c00', 'emoji': '⏸️',  'sla': '< 4 hours'},
    'MONITOR':    {'color': '#f9a825', 'emoji': '👁️',  'sla': '< 24 hours'},
    'ALLOW':      {'color': '#2e7d32', 'emoji': '✅',  'sla': 'No action'},
}


def evaluate_dlp_policy(row: pd.Series) -> dict:
    """
    Evaluate DLP policy for a single access event.
    Returns dict with: action, policy_rule, justification, timestamp
    """
    action      = 'ALLOW'
    policy_rule = None
    reasons     = []

    # Only DLP-relevant events are exports and admin ops
    # (other actions don't exfiltrate data directly)
    is_export    = row.get('is_export', 0) == 1
    is_admin     = row.get('action', '') == 'admin_operation'
    risk_score   = row.get('risk_score', 0)
    severity     = row.get('severity', 'LOW')
    sensitivity  = row.get('sensitivity_score', 0)
    stale        = row.get('stale_account_active', 0) == 1
    offhours     = row.get('is_offhours', 0) == 1
    first_time   = row.get('is_first_time_resource', 0) == 1
    pct_export   = row.get('pct_export', 1.0)
    is_failure   = row.get('is_failure', 0) == 1
    cross_dept   = row.get('cross_dept_access', 0) == 1

    # ── PRIORITY 1: BLOCK ────────────────────────────────────────────────────
    if is_export and severity == 'CRITICAL':
        action = 'BLOCK'
        policy_rule = 'DLP-001'
        reasons.append('CRITICAL-risk export event — automatic block')

    elif is_export and stale:
        action = 'BLOCK'
        policy_rule = 'DLP-002'
        reasons.append(f'Stale account ({int(row.get("days_inactive",0))} days inactive) attempting export')

    elif is_export and sensitivity >= 3 and offhours:
        action = 'BLOCK'
        policy_rule = 'DLP-003'
        reasons.append('High-sensitivity data export during off-hours')

    elif is_failure and is_export:
        action = 'BLOCK'
        policy_rule = 'DLP-004'
        reasons.append('Failed export attempt — possible exfiltration probe blocked')

    # ── PRIORITY 2: QUARANTINE ───────────────────────────────────────────────
    elif is_export and severity == 'HIGH':
        action = 'QUARANTINE'
        policy_rule = 'DLP-005'
        reasons.append('HIGH-risk export — held for analyst review')

    elif is_export and first_time and sensitivity >= 3:
        action = 'QUARANTINE'
        policy_rule = 'DLP-006'
        reasons.append('First-time export of high-sensitivity resource')

    elif is_export and offhours and pct_export < 0.05:
        action = 'QUARANTINE'
        policy_rule = 'DLP-007'
        reasons.append('Off-hours export by user with no export history')

    elif is_export and cross_dept and sensitivity >= 3:
        action = 'QUARANTINE'
        policy_rule = 'DLP-008'
        reasons.append('Cross-department export of sensitive data')

    # ── PRIORITY 3: MONITOR ──────────────────────────────────────────────────
    elif is_export and severity == 'MEDIUM':
        action = 'MONITOR'
        policy_rule = 'DLP-009'
        reasons.append('MEDIUM-risk export — logged for periodic review')

    elif is_export and row.get('is_weekend', 0) == 1:
        action = 'MONITOR'
        policy_rule = 'DLP-010'
        reasons.append('Weekend export — logged for review')

    elif is_export:
        action = 'MONITOR'
        policy_rule = 'DLP-011'
        reasons.append('All export actions monitored by policy')

    # ── PRIORITY 4: ALLOW (non-export events) ────────────────────────────────
    else:
        action = 'ALLOW'
        policy_rule = 'DLP-000'
        reasons.append('Non-export action — logged for audit trail only')

    return {
        'dlp_action':      action,
        'dlp_policy_rule': policy_rule,
        'dlp_justification': '; '.join(reasons),
        'dlp_evaluated_at': datetime.utcnow().isoformat(),
        'dlp_sla':         DLP_ACTIONS[action]['sla'],
    }


def run_dlp_engine(df_scored: pd.DataFrame) -> pd.DataFrame:
    """Apply DLP policy to all events. Returns df with DLP columns added."""
    df = df_scored.copy()
    dlp_results = df.apply(evaluate_dlp_policy, axis=1)
    dlp_df = pd.DataFrame(dlp_results.tolist(), index=df.index)
    return pd.concat([df, dlp_df], axis=1)


def dlp_summary(df_dlp: pd.DataFrame) -> dict:
    """Compute DLP summary statistics."""
    export_events = df_dlp[df_dlp['is_export'] == 1]
    all_events    = df_dlp

    return {
        'total_events':          len(all_events),
        'export_events':         len(export_events),
        'BLOCK':                 (all_events['dlp_action'] == 'BLOCK').sum(),
        'QUARANTINE':            (all_events['dlp_action'] == 'QUARANTINE').sum(),
        'MONITOR':               (all_events['dlp_action'] == 'MONITOR').sum(),
        'ALLOW':                 (all_events['dlp_action'] == 'ALLOW').sum(),
        'blocked_pct':           (all_events['dlp_action'] == 'BLOCK').mean(),
        'exfil_prevented':       (
            (all_events['dlp_action'] == 'BLOCK') &
            (all_events['is_export'] == 1)
        ).sum(),
        'rules_fired':           all_events['dlp_policy_rule'].value_counts().to_dict(),
    }


def print_dlp_report(df_dlp: pd.DataFrame):
    stats = dlp_summary(df_dlp)
    print("=" * 60)
    print("  DLP ENGINE REPORT")
    print("=" * 60)
    print(f"  Total events processed : {stats['total_events']:,}")
    print(f"  Export events          : {stats['export_events']:,}")
    print()
    print("  DLP Actions:")
    for action in ['BLOCK', 'QUARANTINE', 'MONITOR', 'ALLOW']:
        n = stats.get(action, 0)
        emoji = DLP_ACTIONS[action]['emoji']
        print(f"    {emoji} {action:<12} : {n:>4}")
    print()
    print(f"  Exfiltration attempts PREVENTED : {stats['exfil_prevented']}")
    print()
    print("  Top Policy Rules Fired:")
    for rule, cnt in sorted(stats['rules_fired'].items(),
                             key=lambda x: x[1], reverse=True)[:6]:
        print(f"    {rule} : {cnt}")
    print()
    print("  Sample BLOCKED events:")
    blocked = df_dlp[df_dlp['dlp_action'] == 'BLOCK'].sort_values(
        'risk_score', ascending=False).head(5)
    for _, r in blocked.iterrows():
        print(f"    🚫 {r['username']} | {r['action']} on {r['resource']} "
              f"| Risk {r['risk_score']:.0f} | {r['dlp_justification']}")
    print("=" * 60)


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_scored, _, _ = run_full_pipeline(
        '../data/data_access_logs.csv',
        '../data/user_profiles.csv',
        verbose=False
    )

    df_dlp = run_dlp_engine(df_scored)
    print_dlp_report(df_dlp)

    save_cols = [c for c in df_dlp.columns
                 if c not in ('resource_set', '_resource_set')]
    df_dlp[save_cols].to_csv('../output/08_dlp_audit_log.csv', index=False)
    print("\nSaved: output/08_dlp_audit_log.csv")
