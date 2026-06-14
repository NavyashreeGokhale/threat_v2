"""
01_ingest.py — Data Access Log Ingestion Layer
===============================================
Deliverable: "Access log ingestion (supports CSV, API formats)"

WHY THIS EXISTS:
  In production, logs come from multiple sources — a SIEM exports CSV,
  a cloud API streams JSON events, a database dumps parquet files.
  This module provides a single, consistent loader that normalises
  whatever format arrives into one clean DataFrame the rest of the
  pipeline can rely on.

WHAT IT DOES:
  1. Loads CSV logs + user profiles
  2. Validates required columns exist
  3. Standardises data types (timestamps, categories)
  4. Adds derived time columns (hour, day_of_week, month, is_weekend)
  5. Simulates an API ingestion path (JSON format) for the demo
  6. Reports basic ingestion stats
"""

import pandas as pd
import json
import os
from datetime import datetime

# ── Column contracts ──────────────────────────────────────────────────────────
REQUIRED_LOG_COLS = [
    'timestamp', 'user_id', 'username', 'action',
    'resource', 'resource_sensitivity', 'status',
    'source_ip', 'time_classification'
]
REQUIRED_PROFILE_COLS = [
    'user_id', 'username', 'department', 'job_title',
    'privilege_level', 'systems_access', 'days_inactive', 'is_active'
]

# ── Sensitivity encoding (used throughout pipeline) ───────────────────────────
SENSITIVITY_MAP   = {'low': 1, 'medium': 2, 'high': 3, 'restricted': 4}
TIME_CLASS_MAP    = {'business_hours': 0, 'weekend': 1, 'unusual_hours': 2, 'night': 3}
ACTION_RISK_MAP   = {
    'login': 1, 'api_call': 1, 'file_access': 2,
    'sql_query': 2, 'admin_operation': 3, 'export_data': 4
}


def load_csv(logs_path: str, profiles_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files with basic validation."""
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Logs file not found: {logs_path}")
    if not os.path.exists(profiles_path):
        raise FileNotFoundError(f"Profiles file not found: {profiles_path}")

    logs     = pd.read_csv(logs_path,     parse_dates=['timestamp'])
    profiles = pd.read_csv(profiles_path, parse_dates=['last_login', 'hire_date'])

    # Validate required columns
    missing_log  = set(REQUIRED_LOG_COLS)     - set(logs.columns)
    missing_prof = set(REQUIRED_PROFILE_COLS) - set(profiles.columns)
    if missing_log:
        raise ValueError(f"Missing log columns: {missing_log}")
    if missing_prof:
        raise ValueError(f"Missing profile columns: {missing_prof}")

    return logs, profiles


def load_from_api_simulation(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate ingestion from a streaming API (JSON format).
    In production this would be: requests.get(API_ENDPOINT) or a Kafka consumer.
    Here we serialise the first 5 rows to JSON and re-parse them — same logic,
    different transport — to prove the pipeline handles both formats.
    """
    json_payload = logs.head(5).to_json(orient='records', date_format='iso')
    records      = json.loads(json_payload)
    return pd.DataFrame(records)


def standardise(logs: pd.DataFrame, profiles: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalise types and add derived time features.
    Every downstream file calls THIS function — single source of truth.
    """
    logs = logs.copy()
    profiles = profiles.copy()

    # ── Logs: ensure timestamp is datetime ────────────────────────────────────
    logs['timestamp'] = pd.to_datetime(logs['timestamp'])

    # ── Logs: lowercase categoricals ─────────────────────────────────────────
    for col in ['action', 'resource_sensitivity', 'status', 'time_classification']:
        logs[col] = logs[col].str.lower().str.strip()

    # ── Logs: numeric encodings ───────────────────────────────────────────────
    logs['sensitivity_score'] = logs['resource_sensitivity'].map(SENSITIVITY_MAP).fillna(2)
    logs['time_risk_score']   = logs['time_classification'].map(TIME_CLASS_MAP).fillna(0)
    logs['action_risk_score'] = logs['action'].map(ACTION_RISK_MAP).fillna(1)

    # ── Logs: time decomposition ──────────────────────────────────────────────
    logs['hour']       = logs['timestamp'].dt.hour
    logs['dow']        = logs['timestamp'].dt.dayofweek   # 0=Mon, 6=Sun
    logs['month']      = logs['timestamp'].dt.month
    logs['year_month'] = logs['timestamp'].dt.to_period('M').astype(str)
    logs['is_weekend'] = logs['dow'].isin([5, 6]).astype(int)
    logs['is_offhours']= logs['time_classification'].isin(
                            ['night', 'unusual_hours']).astype(int)
    logs['is_export']  = (logs['action'] == 'export_data').astype(int)
    logs['is_failure'] = (logs['status'] == 'failure').astype(int)

    # ── Profiles: boolean fix ────────────────────────────────────────────────
    profiles['is_active']    = profiles['is_active'].astype(bool)
    profiles['stale_account']= (profiles['days_inactive'] > 30).astype(int)

    # ── Profiles: tenure in months ───────────────────────────────────────────
    profiles['hire_date'] = pd.to_datetime(profiles['hire_date'], errors='coerce')
    profiles['tenure_months'] = (
        (pd.Timestamp('today') - profiles['hire_date'])
        .dt.days / 30
    ).round(1)

    return logs, profiles


def ingest(logs_path: str, profiles_path: str,
           verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point — called by every other module.
    Returns (logs, profiles) fully standardised.
    """
    logs_raw, profiles_raw = load_csv(logs_path, profiles_path)
    logs, profiles         = standardise(logs_raw, profiles_raw)

    if verbose:
        print("=" * 55)
        print("  INGESTION REPORT")
        print("=" * 55)
        print(f"  Log events loaded      : {len(logs):,}")
        print(f"  User profiles loaded   : {len(profiles):,}")
        print(f"  Date range             : {logs['timestamp'].min().date()} "
              f"→ {logs['timestamp'].max().date()}")
        print(f"  Unique users in logs   : {logs['user_id'].nunique()}")
        print(f"  Sources (resources)    : {logs['resource'].nunique()}")
        print()
        print("  Action breakdown:")
        for action, cnt in logs['action'].value_counts().items():
            print(f"    {action:<20} {cnt:>4}  ({cnt/len(logs):.1%})")
        print()
        print("  Time classification:")
        for tc, cnt in logs['time_classification'].value_counts().items():
            print(f"    {tc:<20} {cnt:>4}  ({cnt/len(logs):.1%})")
        print()
        print("  Sensitivity levels:")
        for s, cnt in logs['resource_sensitivity'].value_counts().items():
            print(f"    {s:<20} {cnt:>4}  ({cnt/len(logs):.1%})")
        print()
        print("  Failure rate           :", f"{logs['is_failure'].mean():.1%}")
        print("  Off-hours rate         :", f"{logs['is_offhours'].mean():.1%}")
        print("  Export rate            :", f"{logs['is_export'].mean():.1%}")
        print()

        # API simulation proof
        api_sample = load_from_api_simulation(logs_raw)
        print(f"  API ingestion test     : {len(api_sample)} events parsed from JSON ✓")
        print("=" * 55)

    return logs, profiles


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logs, profiles = ingest(
        '../data/data_access_logs.csv',
        '../data/user_profiles.csv'
    )
    logs.to_csv('../output/01_ingested_logs.csv', index=False)
    profiles.to_csv('../output/01_ingested_profiles.csv', index=False)
    print("Saved: output/01_ingested_logs.csv")
    print("Saved: output/01_ingested_profiles.csv")
