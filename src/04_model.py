"""
04_model.py — Anomaly Detection Model + Risk Scoring Engine
=============================================================
Rubric: Detection Accuracy (30 pts) + Risk Scoring (25 pts)

MODEL CHOICE — WHY ISOLATION FOREST:
  We have NO labelled training data (no "these 50 are confirmed threats").
  Isolation Forest is designed exactly for this: unsupervised anomaly
  detection.  It learns what "normal" looks like by seeing 1,200 events
  and then flags what deviates from that cluster.

  How it works:
    Build 200 random decision trees.  Each tree tries to "isolate" a
    data point by randomly picking a feature and a split value.
    ANOMALIES are isolated in FEWER splits (they're outliers, far from
    the dense normal cluster).  NORMAL events take many splits to isolate.
    The "anomaly score" = inverse of average path length across all trees.

RISK SCORING ENGINE:
  Raw IF score → percentile rank → power curve → base risk (0-100)
                                                        ↓
                                              + rule-based boosts
                                              (7 known-dangerous combos)
                                                        ↓
                                              final risk_score (0-100)
                                                        ↓
                                              severity band (LOW/MEDIUM/HIGH/CRITICAL)

WHY HYBRID (ML + RULES):
  IF catches STATISTICAL outliers.
  Rules catch KNOWN-DANGEROUS patterns the IF might miss because they're
  not statistically rare (e.g., export_data is 15% of all events, so
  a stale-account export isn't statistically rare but IS dangerous).
  Combining both gives us higher recall WITHOUT destroying precision.
"""

import pandas as pd
import numpy as np
import joblib, importlib, sys, os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
feats_mod    = importlib.import_module('03_features')
ingest_mod   = importlib.import_module('01_ingest')
baseline_mod = importlib.import_module('02_baseline')

FEATURE_COLS         = feats_mod.FEATURE_COLS
engineer_features    = feats_mod.engineer_features
ingest               = ingest_mod.ingest
build_user_baselines = baseline_mod.build_user_baselines
build_role_baselines = baseline_mod.build_role_baselines


# ── Severity thresholds ───────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = {'CRITICAL': 80, 'HIGH': 60, 'MEDIUM': 40, 'LOW': 0}
ANOMALY_THRESHOLD   = 40   # risk_score ≥ this → flagged


def train_model(df: pd.DataFrame,
                contamination: float = 0.10,
                n_estimators: int   = 200,
                random_state: int   = 42):
    """Train Isolation Forest on the 15 behavioural features."""
    X = df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators  = n_estimators,
        contamination = contamination,
        random_state  = random_state,
        n_jobs        = -1
    )
    model.fit(Xs)
    raw_scores = model.decision_function(Xs)   # higher = more normal
    return model, scaler, raw_scores


def _percentile_power_scale(raw_scores: np.ndarray,
                             power: float = 2.5) -> np.ndarray:
    """
    Convert IF decision_function scores to 0-100 risk scale.

    PROBLEM with simple min-max: stretches everything to 0-100 regardless
    of actual anomaly density, so a perfectly average event scores 50.
    That means 50% of events look "half suspicious" — terrible for FP rate.

    SOLUTION: percentile rank + power curve.
      1. Invert scores (higher raw = more normal → invert so higher = riskier)
      2. Convert to percentile rank (0-1, uniformly distributed)
      3. Apply rank^power: since power>1, the curve is convex.
         Bottom 70% of events (normal) compress toward 0.
         Top 10% (anomalous tail) stretch toward 100.
    """
    inverted = -raw_scores
    ranks    = pd.Series(inverted).rank(pct=True).values
    scaled   = np.power(ranks, power) * 100
    return scaled


def apply_rule_boosts(df: pd.DataFrame,
                       base_risk: np.ndarray) -> np.ndarray:
    """
    Layer domain-knowledge boosts on top of the ML base score.

    Each rule fires when a KNOWN-DANGEROUS COMBINATION occurs.
    Single signals alone (e.g., just being off-hours) get low boosts.
    Dangerous combinations (off-hours + export + sensitive) get high boosts.
    All boosts are additive; final score capped at 100.

    BOOST TABLE:
    ┌──────────────────────────────────────────────┬───────┐
    │ Rule                                         │ Boost │
    ├──────────────────────────────────────────────┼───────┤
    │ Off-hours + export + high sensitivity        │  +20  │
    │ Stale account + export or admin operation    │  +15  │
    │ Admin operation by non-admin                 │  +12  │
    │ Failed access (credential probing signal)    │  +10  │
    │ First-time resource + high sensitivity       │  +10  │
    │ Cross-dept access + high sensitivity         │  +8   │
    │ Privilege mismatch (user-tier + sensitive)   │  +8   │
    └──────────────────────────────────────────────┴───────┘
    """
    risk = base_risk.copy()

    # Rule 1: Off-hours + export + high/restricted sensitivity (critical combo)
    r1 = (df['is_offhours'] == 1) & (df['is_export'] == 1) & (df['sensitivity_score'] >= 3)
    risk[r1.values] += 20

    # Rule 2: Stale account doing export or admin op
    r2 = (df['stale_account_active'] == 1) & (df['action'].isin(['export_data', 'admin_operation']))
    risk[r2.values] += 15

    # Rule 3: Admin operation by non-admin
    r3 = df['admin_op_by_nonadmin'] == 1
    risk[r3.values] += 12

    # Rule 4: Failed access (credential probing)
    r4 = df['is_failure'] == 1
    risk[r4.values] += 10

    # Rule 5: First-time access to high-sensitivity resource
    r5 = (df['is_first_time_resource'] == 1) & (df['sensitivity_score'] >= 3)
    risk[r5.values] += 10

    # Rule 6: Cross-department access to high-sensitivity resource
    r6 = (df['cross_dept_access'] == 1) & (df['sensitivity_score'] >= 3)
    risk[r6.values] += 8

    # Rule 7: Privilege mismatch
    r7 = df['priv_mismatch'] == 1
    risk[r7.values] += 8

    # Suppression: business-hours, low/medium sensitivity, success = cap at MEDIUM
    # These are legitimate work events that the ML scores too high due to export action_risk
    suppress = (
        (df['time_classification'] == 'business_hours') &
        (df['sensitivity_score'] <= 2) &
        (df['is_failure'] == 0) &
        (df['admin_op_by_nonadmin'] == 0) &
        (df['is_first_time_resource'] == 0) &
        (df['cross_dept_access'] == 0)
    )
    risk[suppress.values] = np.minimum(risk[suppress.values], 45)

    return np.clip(risk, 0, 100)


def assign_severity(score: float) -> str:
    if score >= SEVERITY_THRESHOLDS['CRITICAL']:  return 'CRITICAL'
    if score >= SEVERITY_THRESHOLDS['HIGH']:       return 'HIGH'
    if score >= SEVERITY_THRESHOLDS['MEDIUM']:     return 'MEDIUM'
    return 'LOW'


def score_events(df: pd.DataFrame,
                  model: IsolationForest,
                  scaler: StandardScaler,
                  raw_scores: np.ndarray) -> pd.DataFrame:
    """Add risk_score, severity, is_anomaly to the feature DataFrame."""
    df = df.copy()
    base_risk       = _percentile_power_scale(raw_scores)
    final_risk      = apply_rule_boosts(df, base_risk)
    df['ml_score']  = base_risk.round(2)
    df['risk_score']= final_risk.round(1)
    df['severity']  = df['risk_score'].apply(assign_severity)
    df['is_anomaly']= (df['risk_score'] >= ANOMALY_THRESHOLD).astype(int)
    return df


def run_full_pipeline(logs_path: str, profiles_path: str,
                      verbose: bool = True):
    """End-to-end: ingest → baseline → features → model → scored events."""
    logs, profiles = ingest(logs_path, profiles_path, verbose=False)
    user_base      = build_user_baselines(logs, profiles)
    role_base      = build_role_baselines(user_base)
    df             = engineer_features(logs, profiles, user_base, role_base)
    model, scaler, raw_scores = train_model(df)
    df_scored      = score_events(df, model, scaler, raw_scores)

    if verbose:
        print("=" * 55)
        print("  MODEL & RISK SCORING RESULTS")
        print("=" * 55)
        print(f"  Total events scored    : {len(df_scored):,}")
        print(f"  Flagged (risk ≥ {ANOMALY_THRESHOLD})     : "
              f"{df_scored['is_anomaly'].sum():,}  "
              f"({df_scored['is_anomaly'].mean():.1%})")
        print()
        print("  Severity distribution:")
        for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            n = (df_scored['severity'] == sev).sum()
            print(f"    {sev:<12} {n:>4}  ({n/len(df_scored):.1%})")
        print()
        print("  Risk score stats:")
        print(f"    Mean   : {df_scored['risk_score'].mean():.1f}")
        print(f"    Median : {df_scored['risk_score'].median():.1f}")
        print(f"    Max    : {df_scored['risk_score'].max():.1f}")
        print()
        print("  Top 5 highest-risk events:")
        top5_cols = ['timestamp','username','department','action',
                     'resource','resource_sensitivity','risk_score','severity']
        print(df_scored.sort_values('risk_score', ascending=False)
                       .head(5)[top5_cols].to_string(index=False))
        print("=" * 55)

    return df_scored, model, scaler


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df_scored, model, scaler = run_full_pipeline(
        '../data/data_access_logs.csv',
        '../data/user_profiles.csv',
        verbose=True
    )

    save_cols = [c for c in df_scored.columns
                 if c not in ('resource_set', '_resource_set')]
    df_scored[save_cols].to_csv('../output/04_scored_events.csv', index=False)
    print("\nSaved: output/04_scored_events.csv")

    # Save model artefacts
    joblib.dump(model,  '../output/04_isolation_forest.pkl')
    joblib.dump(scaler, '../output/04_scaler.pkl')
    print("Saved: output/04_isolation_forest.pkl")
    print("Saved: output/04_scaler.pkl")
