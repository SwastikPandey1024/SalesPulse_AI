# SalesPulse AI — Operator Experience & ML Observability Guide

## 1. Overview & Operational Philosophy

SalesPulse AI Version 3.0 provides a unified, production-grade ML forecasting, observability, and data quality dashboard built on Streamlit.

The operator experience is structured around 6 dedicated workspaces:
1. **Sales Overview:** Executive business KPIs, historical daily trends, and interactive business "What-If" scenario simulation.
2. **Forecast Lab:** Multi-horizon recursive forecasting (7, 14, 30 days) with candidate model selection, empirical or calibrated prediction interval methods, and persistent audit logging with run ID provenance.
3. **Forecast Performance:** Out-of-sample realization tracking comparing historical forecast runs against reconciled ground-truth actuals, calculating MAE, RMSE, WAPE, Bias, and Interval Coverage.
4. **Data Quality:** Ingestion verification enforcing data contracts, isolating quarantined records, and maintaining an immutable persistent audit trail (`DatasetAuditLog`) with zero customer PII retention.
5. **Anomaly Monitor:** Explainable statistical time-series anomaly detection with scope, type, severity, and status filtering, overlay visualization, evidence cards, and operator triage resolution workflows.
6. **System Health:** Operational readiness diagnostics covering SQLite WAL database status, schema version verification (`v2`), model registry candidate inventory, and cache management.

---

## 2. Workspace Navigation & Workflows

### 2.1 Sales Overview Workspace
- **Executive KPIs:** Live summaries of Total Realized Revenue, Transaction Volume, Average Order Value (AOV), and Top Product Category.
- **Continuous Daily Trend:** Time-series plot of reindexed daily sales paired with a 7-day rolling moving average.
- **Business Scenario Simulator:** Operator levers for Promotional Campaign Boost (0–50%), Price Adjustment (-20% to +20%), and Seasonal Multiplier (0.5x to 1.5x), delivering instant revenue impact deltas and baseline comparisons.
- **Model Benchmarks & Explainability:** Expandable walk-forward rolling-origin cross-validation metrics matrix and XGBoost feature importance rankings.

### 2.2 Forecast Lab Workspace
- **Workflow:**
  1. Select Forecast Horizon (7, 14, or 30 days).
  2. Select Active Candidate Model (`XGBoost`, `RandomForest`, or `Ensemble`).
  3. Select Prediction Interval Method: `heuristic_sqrt_step` (legacy empirical square-root horizon expansion) or `calibrated_oof_residual` (calibrated out-of-fold residual bounds).
  4. Specify Sales Scope (`TOTAL_STORE` or custom identifier).
  5. Check `Persist Run to Audit Storage` to enable delayed realization tracking.
  6. Adjust temporal feature inputs or use pre-populated recent lag values.
  7. Click `Generate Multi-Horizon Forecast`.
- **Output:**
  - Provenance Banner with unique Run ID (`UUIDv4`), origin timestamp, scope, model, and interval method.
  - Interactive Plotly chart with prediction interval uncertainty band.
  - Formatted data table and instant CSV download button.
  - Actionable business recommendations (demand trajectory and 7-day moving average momentum).

### 2.3 Forecast Performance Workspace
- **Delayed Realization Protocol:**
  - Forecast items remain in state `Pending Realization (Awaiting Actuals)` until corresponding ground-truth actual sales are ingested.
  - **Missing actuals are NEVER treated as zero.** They are excluded from error calculations to prevent artificial distortion of MAE, RMSE, WAPE, or Bias.
  - **No Historical Guarantee:** Historical evaluation metrics reflect realized past accuracy and do not guarantee future performance.
- **Metrics Calculated:**
  - **MAE (Mean Absolute Error):** $\frac{1}{N} \sum |actual - predicted|$ across realized steps.
  - **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{N} \sum (actual - predicted)^2}$ across realized steps.
  - **WAPE (Weighted Absolute Percentage Error):** $\frac{\sum |actual - predicted|}{\sum actual} \times 100\%$.
  - **Mean Bias (%):** $\frac{\sum (predicted - actual)}{\sum actual} \times 100\%$.
  - **Interval Coverage Rate (%):** Percentage of realized steps where $lower\_bound \le actual \le upper\_bound$.
- **Operator Actuals Tool:** Form to ingest actual daily sales for a calendar date, automatically triggering cross-run reconciliation for all matching forecast items.

### 2.4 Data Quality Workspace
- **Ingestion Protocol:** Accepts CSV or Excel tabular records. Safe ingestion automatically detects dataset grain (`TRANSACTION`, `DAILY_AGGREGATED`, `FEATURE_BATCH`).
- **Data Freshness & Ingestion Monitoring (Phase 4A):**
  - **Trustworthy Signals:** Evaluates last successful ingestion timestamp, latest attempt outcome, calendar observation freshness (staleness vs threshold), validation failure/warning counts, and consecutive failure escalations.
  - **Deterministic Status States:**
    - `HEALTHY`: Observations within staleness threshold, latest ingestion passed, zero consecutive failures.
    - `WARNING`: Dataset stale ($> \text{threshold}$ days) or latest validation has non-blocking warnings.
    - `CRITICAL`: Severe staleness ($> 2 \times \text{threshold}$), consecutive failures $\ge 2$, or latest validation failed.
    - `UNKNOWN`: Database unavailable or uninitialized baseline; never falsely reports healthy when data is missing.
  - **Operator Action:** Manual "🔄 Run Health Check" button evaluates state on demand without expensive reruns on every interaction.
  - **Audit History:** Expandable log displaying historical monitoring check records, status transitions, and actionable remediations.
- **Contract Enforcement:**
  - `PASSED`: All contracts met; records ready for downstream forecasting.
  - `WARNING`: Non-blocking review flags (e.g. freshness staleness, non-critical distribution anomalies).
  - `FAILED`: Critical violations (duplicate daily keys, invalid numeric sales, missing required feature columns); downstream forecasting is blocked.
- **Quarantine Isolation:** Defective rows are segregated into quarantine with sample row indices and actionable remediation advice.
- **Audit History:** Live table of recent validation runs querying `DatasetAuditLog`. Never stores raw tabular rows.

### 2.5 Anomaly Monitor Workspace
- **Statistical Detection Methodology:**
  - Baseline Center: 14-day rolling median ($C_t = \text{median}(S_{t-W}, \dots, S_{t-1})$).
  - Dispersion Scale: Scaled Median Absolute Deviation ($\text{MAD} \times 1.4826$), falling back to IQR scale or deterministic minimum if dispersion is zero.
  - Robust Z-Score: $z_t = \frac{x_t - C_t}{\text{scale}}$.
  - Classification:
    - `SALES_SURGE`: $z_t \ge k_{surge}$ ($+2.5$).
    - `SALES_DROP`: $z_t \le -k_{drop}$ ($-2.5$).
    - `FORECAST_ERROR_ANOMALY`: Extreme realized forecast error.
- **Severity Levels:**
  - `LOW`: $2.0 \le |z| < 2.5$
  - `MEDIUM`: $2.5 \le |z| < 3.5$
  - `HIGH`: $3.5 \le |z| < 5.0$
  - `CRITICAL`: $|z| \ge 5.0$
- **SEASONALITY LIMITATION NOTICE:**
  > **Important Operator Notice:** The 14-day rolling median baseline pools weekdays and weekends together without day-of-week stratification. Expected intra-week cyclical variations (e.g. Saturday retail surges or mid-week lulls) may register elevated z-scores (~1.8–2.5). Detected anomalies represent statistical signals for operational investigation, not confirmed root causes or automatically seasonally-adjusted facts.
- **Operator Triage Workflow:** Operators can select any active anomaly, supply an operational rationale (e.g. "Confirmed flash promotion" or "Warehouse power failure"), and mark the anomaly as `RESOLVED`. The audit log records the resolution timestamp and rationale immutably.

### 2.6 System Health Workspace
- **Diagnostic Probes:**
  - **Database Readiness:** Checks SQLite WAL connection and verifies active schema version matches application `SCHEMA_VERSION=2`.
  - **Data Freshness & Ingestion SLA:** Displays overall status, calendar staleness days relative to SLA, consecutive failures, and scheduling readiness.
  - **Candidate Registry:** Enumerates active ML models (`XGBoost`, `RandomForest`, `Ensemble`).
  - **Service Architecture:** Probes optional FastAPI daemon (`http://127.0.0.1:8000/ready`). When offline, confirms local self-contained domain mode is operating independently.
  - **Persistence Inventory:** Live counts of persisted forecast runs, anomalies, validation logs, and monitoring checks.
  - **Resource Caches:** Purges `@st.cache_resource` and `@st.cache_data` on demand.

---

## 3. Local Startup & Execution Instructions

### Running Streamlit Locally
```powershell
# In project root
streamlit run app.py
```
The application executes natively using local domain services (`ForecastingService`, `ValidationService`, `AnalyticsService`, `DataMonitoringService`, `AuditRepository`, `ModelRegistry`). A running FastAPI server is **not required**.

### Running Optional FastAPI Daemon
```powershell
uvicorn salespulse.api.app:app --host 127.0.0.1 --port 8000 --reload
```

### Scheduling Boundary & Background Workers
The monitoring service is deliberately decoupled from Streamlit reruns. In Phase 4A, no background daemon is spawned by the frontend. Instead, external schedulers (e.g., cron, Airflow DAG, systemd timer) invoke the programmatic entrypoint:
```python
from salespulse.monitoring.service import run_scheduled_monitoring_check
from salespulse.monitoring.schemas import MonitoringConfig

# Standard cron-scheduled run
verdict = run_scheduled_monitoring_check(
    sales_scope="TOTAL_STORE",
    config=MonitoringConfig(staleness_threshold_days=30, consecutive_failure_threshold=2)
)
print(f"Monitoring check complete: {verdict.overall_status} (Freshness: {verdict.freshness.freshness_state})")
```

### Running Test Suites
```powershell
# Run data freshness and ingestion monitoring tests
pytest tests/test_monitoring.py -v

# Run dashboard integration tests
pytest tests/test_dashboard.py -v

# Run full project regression test suite
pytest -v

# Verify compilation
python -m py_compile app.py
```

---

## 4. Known Limitations & Deferred Capabilities

1. **Continuous Scheduler Daemon:** Phase 4A establishes the decoupled monitoring service and cron-ready entrypoint (`run_scheduled_monitoring_check`), but deliberately avoids launching an uncontrolled always-running background thread inside Streamlit. An external orchestrator or background worker process will be introduced when production deployment infrastructure is finalized.
2. **Intra-Week Seasonality:** Automatic weekday/weekend stratification is deferred to Phase 4B. Current anomaly alerts require operator discernment for known cyclical days.
3. **Online Learning & Retraining:** Model weights are fixed and read-only. Retraining and automated hyperparameter tuning are managed externally.
4. **Multi-Tenant Permissions:** Role-based access control (RBAC) and user authentication are deferred; all operators currently share administrative visibility.
