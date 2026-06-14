# 🛡️ InsiderPulse — Data Access Audit & Insider Threat Detection
**Problem Statement 04 | Option A: Behavioral ML + Explainable Narratives + DLP Integration**

🔗 **Live Demo:** https://insider-threat-v2.streamlit.app/

---

## Deliverables Checklist

- ✅ **GitHub Repo** — code, `requirements.txt`, clear README
- ✅ **Jupyter Notebook** — `notebooks/EDA_and_Analysis.ipynb` (EDA, baselines, feature importance, metrics)
- ✅ **Anomaly Detection Output** — `output/06_incident_report.csv` (20 flagged events with full narratives)
- ✅ **Risk Dashboard** — 4-tab Streamlit app (Dashboard · Investigation · DLP · Evaluation)
- ✅ **False Positive Analysis** — 6 edge cases documented in app Tab 2 + `TECHNICAL_DOCS.md`
- ✅ **Technical Docs** — `TECHNICAL_DOCS.md` (architecture, feature engineering, scaling to 1M+)
- ✅ **Evaluation Metrics** — Precision / Recall / F1 with full threshold sweep (Tab 4)
- ✅ **DLP Integration** — `src/08_dlp.py` — 11 policy rules, BLOCK / QUARANTINE / MONITOR / ALLOW
- ✅ **Performance** — 1M events in **45.5s** ✓ (target < 120s)

---

## Architecture

```
data/
├── data_access_logs.csv          raw access events (1,200 × 365 days)
└── user_profiles.csv             user metadata (100 users)

src/
├── 01_ingest.py      → Ingest layer  — CSV + API/JSON simulation, validation, standardisation
├── 02_baseline.py    → Baselines     — per-user & per-role behavioral statistics
├── 03_features.py    → Features      — 15 deviation-based behavioral features
├── 04_model.py       → Model         — Isolation Forest + 7 rule-based risk boosts → 0-100 score
├── 05_evaluate.py    → Evaluation    — Precision / Recall / F1 against pseudo-labels
├── 06_narratives.py  → Narratives    — plain-English incident narratives (LLM-prompt-ready)
├── 07_performance.py → Benchmark     — 1M-event throughput test
└── 08_dlp.py         → DLP Engine    — 11 policy rules, exfiltration prevention

app.py                → 4-tab Streamlit dashboard
notebooks/
└── EDA_and_Analysis.ipynb        baseline analysis, feature importance, model evaluation
output/
├── 01_ingested_logs.csv
├── 02_user_baselines.csv
├── 02_role_baselines.csv
├── 03_features.csv
├── 04_scored_events.csv
├── 05_evaluated_events.csv
├── 06_incident_report.csv        20 flagged events with full narratives
├── 06_llm_prompts.csv            LLM-ready prompts for each alert
├── 07_performance.csv            benchmark results
└── 08_dlp_audit_log.csv          full DLP audit trail
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
python 01_ingest.py        # → output/01_ingested_logs.csv
python 02_baseline.py      # → output/02_user_baselines.csv
python 03_features.py      # → output/03_features.csv
python 04_model.py         # → output/04_scored_events.csv
python 05_evaluate.py      # → Precision / Recall / F1 printed to console
python 06_narratives.py    # → output/06_incident_report.csv
python 07_performance.py   # → 1M event benchmark
python 08_dlp.py           # → output/08_dlp_audit_log.csv
```

---

## Results

| Metric | Value | Target |
|---|---|---|
| Events analysed | 1,200 | — |
| Users profiled | 100 | — |
| CRITICAL alerts | 219 (18.2%) | — |
| HIGH alerts | 110 (9.2%) | — |
| Recall | **71.9%** | >70% ✓ |
| Precision (CRITICAL band) | **72.6%** | >75% (near target) |
| F1 Score | 0.66 | >0.72 |
| 1M event throughput | **45.5s** | <120s ✓ |
| Throughput | 21,967 events/sec | — |
| DLP — Blocked exports | **123** | — |
| DLP — Quarantined | **19** | — |

---

## Model: Why Isolation Forest?

No labelled training data exists. Isolation Forest is designed for exactly this: **unsupervised anomaly detection**.

- Builds 200 random decision trees
- Anomalies are isolated in **fewer splits** — they're far from the dense normal cluster
- Normal events need many splits to isolate
- Raw IF scores → percentile rank → power curve (`rank^2.5`) → base risk 0-100
- Combined with **7 domain-knowledge rule boosts** targeting known-dangerous patterns

### 7 Rule-Based Risk Boosts

| Rule | Boost |
|---|---|
| Off-hours + export + high/restricted sensitivity | +20 |
| Stale account + export or admin operation | +15 |
| Admin operation by non-admin | +12 |
| Failed access attempt (credential probing) | +10 |
| First-time access to high-sensitivity resource | +10 |
| Cross-department access to high-sensitivity resource | +8 |
| Privilege mismatch (user-tier on sensitive data) | +8 |

---

## 15 Behavioral Features

All features measure **deviation from that user's own baseline**, not global thresholds.

| Feature | What it measures |
|---|---|
| `offhours_deviation` | Off-hours event × (1 − user's historical % off-hours) |
| `export_deviation` | Export event × (1 − user's historical % exports) |
| `failure_deviation` | Failure × (1 − user's historical failure rate) |
| `sensitivity_above_base` | How much MORE sensitive than user normally accesses |
| `is_first_time_resource` | Time-causal first-ever access to this resource |
| `cross_dept_access` | Resource outside user's department's expected systems |
| `priv_mismatch` | User-tier privilege accessing high-sensitivity data |
| `admin_op_by_nonadmin` | Admin operation by non-admin / power-user |
| `stale_account_active` | Account inactive >30 days |
| `is_failure` | Failed access attempt |
| `sensitivity_score` | Raw sensitivity encoding (low=1 … restricted=4) |
| `time_risk_score` | Raw time risk (business_hours=0 … night=3) |
| `action_risk_score` | Raw action risk (login=1 … export_data=4) |
| `is_offhours` | Binary: night or unusual_hours |
| `is_export` | Binary: export_data action |

---

## DLP Integration

`src/08_dlp.py` implements a full **Data Loss Prevention policy engine** with 11 rules:

| Rule | Trigger | Action |
|---|---|---|
| DLP-001 | CRITICAL risk + export_data | 🚫 BLOCK |
| DLP-002 | Stale account + export | 🚫 BLOCK |
| DLP-003 | High-sensitivity export, off-hours | 🚫 BLOCK |
| DLP-004 | Failed export attempt | 🚫 BLOCK |
| DLP-005 | HIGH risk + export_data | ⏸️ QUARANTINE |
| DLP-006 | First-time export of high-sensitivity resource | ⏸️ QUARANTINE |
| DLP-007 | Off-hours export, no export history | ⏸️ QUARANTINE |
| DLP-008 | Cross-department sensitive export | ⏸️ QUARANTINE |
| DLP-009 | MEDIUM risk + export | 👁️ MONITOR |
| DLP-010 | Weekend export | 👁️ MONITOR |
| DLP-011 | Any other export | 👁️ MONITOR |

In production the DLP engine runs as a **synchronous pre-action webhook** — BLOCK happens before data leaves the system, not after.

---

## False Positive Control

| Edge Case | How It's Handled |
|---|---|
| Month-end bulk exports | `export_deviation` is relative to each user's own history — regular exporters score low |
| New admin / role change | Uses current `privilege_level` from profile — newly promoted admins don't trigger non-admin flag |
| On-call rotation | Repeated patterns become part of baseline → `offhours_deviation` self-corrects |
| Contractors (thin history) | Role-level baseline fallback for users with <5 events |
| Service accounts | Stable repetitive patterns cluster tightly → IF gives low anomaly scores naturally |
| Business-hours legitimate exports | Suppression rule caps risk at 45 (MEDIUM max) for low-risk combinations |

---

## Scaling to 1M+ Daily Events

| Stage | Approach |
|---|---|
| Ingest | Kafka / Kinesis streaming topic per source |
| Baselines | Pre-computed nightly in Delta Lake, joined at score time (not recomputed per event) |
| Feature compute | Spark Structured Streaming for rolling window aggregates |
| Scoring | FastAPI microservice — stateless, O(1) per event |
| Alerting | CRITICAL/HIGH → SIEM webhook (Splunk / Sentinel) in <5 minutes |
| DLP enforcement | Synchronous pre-action webhook, <50ms latency |

**Benchmark:** 1M events processed in **45.5s** (21,967 events/sec) on a single machine.

---

## Regulatory Alignment

| Regulation | Requirement | Coverage |
|---|---|---|
| GDPR Article 32 | Monitor unauthorized access, detect exfiltration | Every access logged; DLP blocks exfiltration pre-transfer |
| NIST IR-4 | Incident detection & response procedures | Severity-banded alerts with recommended actions + DLP audit trail |
| SOX 302 | Controls over GL, AR, AP financial system access | GL_System and HRIS access monitored; cross-dept access flagged |

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| 📊 Dashboard | Severity distribution, dept risk, trend chart, flagged events table, incident report |
| 🔍 Investigation Toolkit | Alert selector, anomaly signals, user baseline context, FP edge-case table |
| 🚫 DLP Control Panel | Blocked/quarantined counts, policy rule breakdown, audit tables |
| 📈 Evaluation & Metrics | Precision/Recall/F1, threshold sweep chart, performance benchmark, regulatory alignment |

---

## Stack

Python · Scikit-learn · Pandas · NumPy · Streamlit · Plotly · Joblib · Jupyter
