"""
03_features.py — Feature Engineering
======================================
Rubric: Detection Accuracy (30 pts) — "feature engineering is key"

WHY THIS EXISTS:
  Raw log columns (action='export_data', time='night') are not enough.
  A machine learning model needs DEVIATION signals — not "did this happen?"
  but "is this unusual FOR THIS USER compared to their own history?"

  Example:
    Raw:  action=export_data, time=night
    Good: export_deviation=0.95 (user almost never exports)
          offhours_deviation=0.88 (user almost never works nights)
          sensitivity_above_baseline=1.5 (accessing far more sensitive data than usual)

  This transforms 9 raw columns into 15 targeted behavioral features.

15 FEATURES EXPLAINED:
  Deviation features (compare event to user's own baseline):
  1.  offhours_deviation      — off-hours × (1 − user's typical % off-hours)
  2.  export_deviation        — export × (1 − user's typical % exports)
  3.  failure_deviation       — failure × (1 − user's typical failure rate)
  4.  sensitivity_above_base  — how much MORE sensitive than user normally accesses

  First-time / novelty features:
  5.  is_first_time_resource  — has user EVER accessed this resource before?
                                (computed time-causally, in timestamp order)

  Cross-context features:
  6.  cross_dept_access       — resource not typically used by user's department
  7.  priv_mismatch           — user-tier accessing high-sensitivity data
  8.  admin_op_by_nonadmin    — admin_operation by non-admin/power-user

  Direct risk signals:
  9.  sensitivity_score       — raw sensitivity encoding (1-4)
  10. time_risk_score         — raw time risk encoding (0-3)
  11. action_risk_score       — raw action risk encoding (1-4)
  12. is_offhours             — binary: night or unusual_hours
  13. is_export               — binary: export_data action
  14. is_failure              — binary: status = failure
  15. stale_account_active    — account inactive >30 days but still logging in
"""

import pandas as pd
import numpy as np
import importlib, sys, os
sys.path.insert(0, os.path.dirname(__file__))

ingest_mod   = importlib.import_module('01_ingest')
baseline_mod = importlib.import_module('02_baseline')

ingest                = ingest_mod.ingest
build_user_baselines  = baseline_mod.build_user_baselines
build_role_baselines  = baseline_mod.build_role_baselines

# Department → expected resources mapping (domain knowledge)
DEPT_RESOURCE_MAP = {
    'HR':          {'HRIS', 'File_Share', 'Email_Archive'},
    'Finance':     {'GL_System', 'BI_Tool', 'File_Share', 'Email_Archive'},
    'Engineering': {'PROD_DB', 'Admin_Console', 'Data_Lake', 'File_Share'},
    'IT':          {'Admin_Console', 'SIEM', 'PROD_DB', 'File_Share'},
    'Sales':       {'Customer_Vault', 'BI_Tool', 'Email_Archive', 'File_Share'},
    'Marketing':   {'BI_Tool', 'Customer_Vault', 'Email_Archive', 'File_Share'},
    'Security':    {'SIEM', 'Admin_Console', 'Data_Lake', 'File_Share'},
    'Compliance':  {'Data_Lake', 'BI_Tool', 'HRIS', 'GL_System', 'File_Share'},
    'Legal':       {'File_Share', 'Email_Archive', 'Data_Lake'},
    'Operations':  {'PROD_DB', 'Data_Lake', 'BI_Tool', 'File_Share'},
    'Support':     {'Customer_Vault', 'File_Share', 'Email_Archive'},
    'Executive':   {'BI_Tool', 'GL_System', 'HRIS', 'File_Share', 'Email_Archive'},
}

# The 15 features the model will train on
FEATURE_COLS = [
    'sensitivity_score',
    'time_risk_score',
    'action_risk_score',
    'is_offhours',
    'is_export',
    'is_failure',
    'offhours_deviation',
    'export_deviation',
    'failure_deviation',
    'sensitivity_above_base',
    'is_first_time_resource',
    'cross_dept_access',
    'priv_mismatch',
    'admin_op_by_nonadmin',
    'stale_account_active',
]


def _compute_first_time_resource(df: pd.DataFrame) -> pd.Series:
    """
    Time-causal first-time-resource flag.
    Walks events in chronological order and marks 1 the first time
    a user touches each resource.  Later occurrences = 0.
    This is crucial — we must NOT look at future events to judge past ones.
    """
    df_sorted  = df.sort_values('timestamp')
    seen       = {}
    flags      = []
    for _, row in df_sorted.iterrows():
        uid, res = row['user_id'], row['resource']
        user_seen = seen.setdefault(uid, set())
        flags.append(1 if res not in user_seen else 0)
        user_seen.add(res)
    result = pd.Series(flags, index=df_sorted.index)
    return result.reindex(df.index)   # restore original row order


def engineer_features(logs: pd.DataFrame,
                       profiles: pd.DataFrame,
                       user_base: pd.DataFrame,
                       role_base: pd.DataFrame) -> pd.DataFrame:
    df = logs.copy()

    # ── Merge user baseline ───────────────────────────────────────────────────
    base_cols = ['user_id', 'pct_offhours', 'pct_export', 'pct_failure',
                 'mean_sensitivity', 'max_sensitivity', 'total_events']
    df = df.merge(user_base[base_cols], on='user_id', how='left')

    # ── Merge profile metadata ────────────────────────────────────────────────
    prof_cols = ['user_id', 'department', 'job_title', 'privilege_level',
                 'days_inactive', 'stale_account']
    df = df.merge(profiles[prof_cols], on='user_id', how='left')

    # ── Fallback: if user has <5 events use role baseline ────────────────────
    df = df.merge(role_base, on='privilege_level', how='left')
    thin = df['total_events'].fillna(0) < 5
    df.loc[thin, 'pct_offhours']     = df.loc[thin, 'role_pct_offhours']
    df.loc[thin, 'pct_export']       = df.loc[thin, 'role_pct_export']
    df.loc[thin, 'mean_sensitivity'] = df.loc[thin, 'role_mean_sens']
    df.loc[thin, 'pct_failure']      = df.loc[thin, 'role_pct_failure']

    # ── Feature 1: offhours_deviation ────────────────────────────────────────
    # "How surprising is an off-hours event given this user's history?"
    df['offhours_deviation'] = (
        df['is_offhours'] * (1 - df['pct_offhours'].fillna(0.25))
    ).round(4)

    # ── Feature 2: export_deviation ──────────────────────────────────────────
    df['export_deviation'] = (
        df['is_export'] * (1 - df['pct_export'].fillna(0.15))
    ).round(4)

    # ── Feature 3: failure_deviation ─────────────────────────────────────────
    df['failure_deviation'] = (
        df['is_failure'] * (1 - df['pct_failure'].fillna(0.05))
    ).round(4)

    # ── Feature 4: sensitivity_above_base ────────────────────────────────────
    # Positive only — we care when this event is MORE sensitive than usual
    df['sensitivity_above_base'] = (
        df['sensitivity_score'] - df['mean_sensitivity'].fillna(2)
    ).clip(lower=0).round(4)

    # ── Feature 5: is_first_time_resource (time-causal) ──────────────────────
    df['is_first_time_resource'] = _compute_first_time_resource(df)

    # ── Feature 6: cross_dept_access ─────────────────────────────────────────
    def _cross_dept(row):
        allowed = DEPT_RESOURCE_MAP.get(row['department'], set())
        return 0 if (not allowed or row['resource'] in allowed) else 1
    df['cross_dept_access'] = df.apply(_cross_dept, axis=1)

    # ── Feature 7: priv_mismatch ──────────────────────────────────────────────
    # "user"-tier privilege accessing high or restricted data
    df['priv_mismatch'] = (
        (df['privilege_level'] == 'user') &
        (df['sensitivity_score'] >= 3)
    ).astype(int)

    # ── Feature 8: admin_op_by_nonadmin ──────────────────────────────────────
    df['admin_op_by_nonadmin'] = (
        (df['action'] == 'admin_operation') &
        (~df['privilege_level'].isin(['admin', 'power-user']))
    ).astype(int)

    # ── Feature 15: stale_account_active ─────────────────────────────────────
    df['stale_account_active'] = df['stale_account'].fillna(0).astype(int)

    return df


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logs, profiles = ingest('../data/data_access_logs.csv',
                            '../data/user_profiles.csv', verbose=False)
    user_base = build_user_baselines(logs, profiles)
    role_base = build_role_baselines(user_base)
    df = engineer_features(logs, profiles, user_base, role_base)

    print("=" * 55)
    print("  FEATURE ENGINEERING SUMMARY")
    print("=" * 55)
    print(f"  Events with features   : {len(df)}")
    print(f"  Feature columns        : {len(FEATURE_COLS)}")
    print()
    print("  Feature means (signal strength):")
    for c in FEATURE_COLS:
        print(f"    {c:<30} {df[c].mean():.4f}")
    print()
    print("  Non-zero rates (how often each fires):")
    for c in FEATURE_COLS:
        rate = (df[c] > 0).mean()
        print(f"    {c:<30} {rate:.1%}")
    print("=" * 55)

    save_cols = [c for c in df.columns
                 if c not in ('resource_set', '_resource_set')]
    df[save_cols].to_csv('../output/03_features.csv', index=False)
    print("\nSaved: output/03_features.csv")
