# 🛡️ InsiderPulse — Data Access Audit & Insider Threat Detection
**Problem Statement 04 | Option A: Behavioral ML + Explainable Narratives**

Live demo: https://insiderpulse.streamlit.app

---

## Deliverables Checklist

- ✅ **GitHub Repo** — code, requirements.txt, clear README
- ✅ **Jupyter Notebook** — `notebooks/EDA_and_Analysis.ipynb` (EDA, baselines, feature importance, metrics)
- ✅ **Anomaly Detection Output** — `output/06_incident_report.csv` (20 flagged events with narratives)
- ✅ **Risk Dashboard** — Streamlit app with filters, charts, investigation toolkit
- ✅ **False Positive Analysis** — documented in dashboard + `TECHNICAL_DOCS.md`
- ✅ **Technical Docs** — `TECHNICAL_DOCS.md` (architecture, features, scaling)
- ✅ **Evaluation Metrics** — Precision/Recall/F1 with threshold sweep
- ✅ **Performance** — 1M events in 45.5s (✓ < 120s target)

---

## Architecture

```
data/ (CSV logs + user profiles)
  ↓
src/01_ingest.py      → Ingest (CSV + API simulation)
  ↓
src/02_baseline.py    → Per-user + per-role behavioral baselines
  ↓
src/03_features.py    → 15 deviation-based behavioral features
  ↓
src/04_model.py       → Isolation Forest + 7 rule-based risk boosts → 0-100 score
  ↓
src/05_evaluate.py    → Precision/Recall/F1 against pseudo-labels
  ↓
src/06_narratives.py  → Plain-English incident narratives (LLM-prompt-ready)
  ↓
src/07_performance.py → 1M-event benchmark
  ↓
app.py                → Streamlit dashboard
notebooks/            → EDA & Analysis Jupyter notebook
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Pipeline Step-by-Step

```bash
cd src
python 01_ingest.py       # → output/01_ingested_logs.csv
python 02_baseline.py     # → output/02_user_baselines.csv
python 03_features.py     # → output/03_features.csv
python 04_model.py        # → output/04_scored_events.csv
python 05_evaluate.py     # → Precision/Recall/F1 metrics
python 06_narratives.py   # → output/06_incident_report.csv
python 07_performance.py  # → 1M event benchmark
```

---

## Results

| Metric | Value | Target |
|---|---|---|
| Events analysed | 1,200 | — |
| Users profiled | 100 | — |
| CRITICAL alerts | 219 (18.2%) | — |
| HIGH alerts | 110 (9.2%) | — |
| Recall | 71.9% | >70% ✓ |
| Precision (CRITICAL band) | 72.6% | >75% (near) |
| F1 Score | 0.66 | >0.72 |
| 1M event throughput | **45.5s** | <120s ✓ |
| Events/second | 21,967 | — |

---

## Model: Why Isolation Forest?

No labelled training data exists.  Isolation Forest is designed for
exactly this: **unsupervised anomaly detection**.

- Builds 200 random decision trees
- Anomalies are isolated in *fewer splits* (they're far from the normal cluster)
- Normal events need many splits to isolate
- We combine the statistical score with **7 domain-knowledge rule boosts**
  targeting known-dangerous patterns (off-hours export of sensitive data,
  stale-account reactivation, admin operation by non-admin, etc.)

---

## 15 Behavioral Features

| Feature | What it measures |
|---|---|
| `offhours_deviation` | Off-hours × (1 − user's own % off-hours) |
| `export_deviation` | Export × (1 − user's own % exports) |
| `failure_deviation` | Failure × (1 − user's own failure rate) |
| `sensitivity_above_base` | How much more sensitive than user normally touches |
| `is_first_time_resource` | Time-causal: first time user ever accessed this resource |
| `cross_dept_access` | Resource outside user's department's typical systems |
| `priv_mismatch` | User-tier privilege accessing high-sensitivity data |
| `admin_op_by_nonadmin` | Admin operation by non-admin/power-user |
| `stale_account_active` | Account inactive >30 days |
| `is_failure` | Failed access attempt |
| `sensitivity_score` | Raw data sensitivity (1-4) |
| `time_risk_score` | Raw time risk (0=business hours … 3=night) |
| `action_risk_score` | Raw action risk (1=login … 4=export) |
| `is_offhours` | Binary: night or unusual_hours |
| `is_export` | Binary: export_data action |

---

## Regulatory Alignment
- **GDPR Article 32** — Monitor unauthorized access, detect exfiltration
- **NIST IR-4** — Incident detection + response procedures
- **SOX 302** — Controls over GL, AR, AP system access

---

## Stack
Python · Scikit-learn · Pandas · NumPy · Streamlit · Plotly · Joblib
