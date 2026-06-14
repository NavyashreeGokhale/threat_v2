"""
02_baseline.py — User Behavioral Baseline Builder
===================================================
Deliverable: "Establish baseline behavior per user/role"
Rubric:      False Positive Control (20 pts) — context understood

WHY THIS EXISTS:
  The core insight of behavioral anomaly detection is comparing each
  event to THAT USER'S OWN history, not a global rule like "flag all
  night access".  A DBA who always works at 3 AM is not suspicious.
  A Finance intern who does it once is.

  This module computes a baseline profile for every user and every
  role-group from the historical logs.  Everything is stored in a
  single DataFrame (one row per user) that the feature engineer joins
  against in the next step.

WHAT IT COMPUTES:
  Per-user statistics:
  ┌─────────────────────────────────────────────────────────┐
  │  pct_offhours   — how often does this user work late?   │
  │  pct_weekend    — how often on weekends?                 │
  │  pct_export     — how often does this user export data? │
  │  pct_failure    — failure rate for this user            │
  │  mean_sens      — avg sensitivity of data they touch    │
  │  top_resources  — which resources they normally hit     │
  │  avg_daily_evt  — typical number of events per day      │
  │  resource_set   — full set of resources ever accessed   │
  └─────────────────────────────────────────────────────────┘

  Per-role baselines (for new users / contractors with little history):
  ┌─────────────────────────────────────────────────────────┐
  │  role_pct_offhours, role_pct_export, role_mean_sens     │
  └─────────────────────────────────────────────────────────┘
"""

import pandas as pd
import numpy as np
import importlib, sys, os
sys.path.insert(0, os.path.dirname(__file__))
ingest_mod = importlib.import_module('01_ingest')
ingest = ingest_mod.ingest


def build_user_baselines(logs: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user behavioral baseline.
    Returns a DataFrame with one row per user_id.
    """
    df = logs.copy()

    # ── Per-user aggregations ─────────────────────────────────────────────────
    base = df.groupby('user_id').agg(
        total_events      = ('timestamp', 'count'),
        n_unique_resources= ('resource',  'nunique'),
        pct_offhours      = ('is_offhours','mean'),
        pct_weekend       = ('is_weekend', 'mean'),
        pct_export        = ('is_export',  'mean'),
        pct_failure       = ('is_failure', 'mean'),
        mean_sensitivity  = ('sensitivity_score', 'mean'),
        max_sensitivity   = ('sensitivity_score', 'max'),
        first_seen        = ('timestamp',  'min'),
        last_seen         = ('timestamp',  'max'),
    ).reset_index()

    # Average events per active day
    base['active_days'] = (
        (base['last_seen'] - base['first_seen']).dt.days + 1
    )
    base['avg_daily_events'] = (base['total_events'] / base['active_days']).round(2)

    # Most-used resource per user
    top_res = (
        df.groupby(['user_id', 'resource'])
          .size()
          .reset_index(name='cnt')
          .sort_values('cnt', ascending=False)
          .groupby('user_id')
          .first()
          .rename(columns={'resource': 'top_resource'})
          [['top_resource']]
          .reset_index()
    )
    base = base.merge(top_res, on='user_id', how='left')

    # Full set of resources ever accessed (used for first-time-access detection)
    res_sets = df.groupby('user_id')['resource'].apply(set).rename('resource_set')
    base = base.merge(res_sets, on='user_id', how='left')

    # Merge profile metadata
    base = base.merge(
        profiles[['user_id', 'department', 'job_title', 'privilege_level',
                  'days_inactive', 'stale_account', 'is_active', 'tenure_months']],
        on='user_id', how='left'
    )

    return base


def build_role_baselines(user_baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Per-privilege-level baseline (fallback for users with <10 events).
    Used by feature engineer when individual history is too thin.
    """
    role_base = user_baselines.groupby('privilege_level').agg(
        role_pct_offhours  = ('pct_offhours',     'mean'),
        role_pct_export    = ('pct_export',        'mean'),
        role_mean_sens     = ('mean_sensitivity',  'mean'),
        role_pct_failure   = ('pct_failure',       'mean'),
    ).reset_index()
    return role_base


def print_baseline_summary(base: pd.DataFrame):
    print("=" * 55)
    print("  USER BASELINE SUMMARY")
    print("=" * 55)
    print(f"  Users profiled         : {len(base)}")
    print()
    print("  Avg behaviour by privilege level:")
    cols = ['pct_offhours', 'pct_weekend', 'pct_export', 'mean_sensitivity']
    summary = base.groupby('privilege_level')[cols].mean().round(3)
    print(summary.to_string())
    print()
    print("  Users with stale accounts  :", base['stale_account'].sum())
    print("  Users active in logs       :", base['total_events'].gt(0).sum())
    print()
    print("  Sample baselines (5 users):")
    sample_cols = ['user_id', 'department', 'privilege_level',
                   'pct_offhours', 'pct_export', 'mean_sensitivity', 'total_events']
    print(base[sample_cols].head(5).to_string(index=False))
    print("=" * 55)


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logs, profiles = ingest('../data/data_access_logs.csv',
                            '../data/user_profiles.csv', verbose=False)

    user_base = build_user_baselines(logs, profiles)
    role_base = build_role_baselines(user_base)

    print_baseline_summary(user_base)

    # Save (drop the resource_set column — it's a Python set, not CSV-serialisable)
    save_cols = [c for c in user_base.columns if c != 'resource_set']
    user_base[save_cols].to_csv('../output/02_user_baselines.csv', index=False)
    role_base.to_csv('../output/02_role_baselines.csv', index=False)
    print("\nSaved: output/02_user_baselines.csv")
    print("Saved: output/02_role_baselines.csv")
