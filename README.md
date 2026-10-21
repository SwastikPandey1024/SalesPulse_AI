<div align="center">

# 📊 SalesPulse AI v3.0 Production
### *Autonomous Sales Analytics, Multi-Horizon ML Forecasting, Observability & Executive Decision-Support*

[![Release v3.0](https://img.shields.io/badge/Release-v3.0.0_Production-4F46E5?style=for-the-badge&logo=github)](https://github.com/SwastikPandey1024/SalesPulse_AI)
[![Tests Passing](https://img.shields.io/badge/Tests-209_Passed-10B981?style=for-the-badge&logo=pytest)](https://github.com/SwastikPandey1024/SalesPulse_AI)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI_v0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit_v1.43-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![SQLite WAL](https://img.shields.io/badge/Storage-SQLite_WAL_v2-003B57?style=for-the-badge&logo=sqlite)](https://sqlite.org/)
[![License CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

<br/>

**[📖 Architecture Specs](docs/PRD.md)** • **[📡 Observability Guide](docs/OPERATOR_EXPERIENCE_AND_OBSERVABILITY.md)** • **[🖼️ Visual Walkthrough](docs/DASHBOARD_WALKTHROUGH.md)** • **[📐 Contracts](docs/INTEGRATION_CONTRACTS.md)** • **[🌐 Deployment](DEPLOYMENT.md)**

</div>

---

## 🏛️ Executive Product Mission

**SalesPulse AI** is an enterprise-grade sales decision-support platform designed for retail operators, financial planners, and commercial directors. Rather than functioning as a superficial machine-learning showcase, SalesPulse AI combines deterministic commercial accounting, recursive time-series forecasting, statistical anomaly detection, data freshness and distribution drift monitoring, and an auditable business recommendation engine to answer:

1. **What is happening in the business?** Executive Gross vs. Net Sales, returns impact, Average Order Value (AOV), Average Selling Price (ASP), and multi-period trends.
2. **Why is it happening?** Root-cause decomposition: cancellation rates, portfolio SKU dependency, weekend dispatch latency, and geographic concentration.
3. **What is likely to happen next?** Recursive 7-, 14-, and 30-day sales forecasting with empirical or calibrated prediction intervals and seasonal-naive baselines.
4. **Which risks and opportunities deserve management attention?** Quantified, prioritized recommendations ranked by commercial urgency and evidence confidence.
5. **What actions should leadership take?** Operational directives with quantifiable financial impact, explicit assumptions, and follow-up success metrics.
6. **How reliable are the data and predictions?** Comprehensive data validation, date-gap auditing, out-of-sample walk-forward cross-validation, and ground-truth realization tracking.
7. **Is the ML system healthy, and when should it be retrained?** Formal MLOps governance matrix distinguishing feature drift (PSI/TVD) from realized performance degradation (WAPE), with strict champion-challenger promotion gates and zero silent overwrites.

---

## 🌟 Four Core Business Capabilities

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             SALESPULSE AI v3.0                              │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ A. Understand the    │ B. Anticipate the    │ C. Recommend Actions          │
│    Business          │    Future            │    (Decision Support)         │
│ • Net & Gross Sales  │ • Recursive Forecast │ • Margin Erosion Alert        │
│ • Realized Returns   │ • Walk-Forward OOF   │ • Portfolio Concentration     │
│ • AOV & ASP          │ • Prediction Bands   │ • Geographic Over-reliance    │
│ • Product & Geo Mix  │ • Seasonal Naive     │ • Dispatch Latency Smoothing  │
├──────────────────────┴──────────────────────┴───────────────────────────────┤
│ D. Operate the System Reliably (MLOps & Governance)                         │
│ • Ingestion Validation & Freshness SLA   • Feature Drift (PSI & TVD)        │
│ • Anomaly Detection (14d Median/MAD)     • Retraining Governance Protocol   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Interactive Dashboard Preview & Operator Workspaces

SalesPulse AI features six dedicated, purpose-built workspaces in `app.py`. For complete high-resolution walkthroughs, see the **[🖼️ Dashboard Feature Walkthrough Guide](docs/DASHBOARD_WALKTHROUGH.md)**.

> [!NOTE]
> **Runtime & Demonstration Disclaimer:** The screenshots below are illustrative examples captured from the baseline evaluation environment running historical UCI Online Retail II demonstration data. Displayed KPI values, dates, and monitoring statuses represent captured snapshots and may differ from your current local runtime or newly ingested datasets.

### 1. 📊 Sales Overview & Decision-Support Radar
- **Commercial KPIs:** Net Realized Revenue, Gross Sales, Returns & Cancellations, Invoices/Orders, Average Order Value (AOV), Average Selling Price (ASP), Units Dispatched, and Return Rate (%).
- **Time-Series Analysis:** Daily Net Sales with 7-day and 30-day moving average trendlines.
- **Seasonality & Growth:** Month-over-Month (MoM) revenue growth tables and seasonality charts.
- **Contribution Breakdown:** Top revenue-generating products and country revenue share donut charts.
- **Interactive Scenario Simulator:** Simulate promotional lift, price elasticity, and seasonal shifts.

[![Sales Overview Workspace](media/sales-overview.png)](media/sales-overview.png)
*Figure 1: Executive KPI banner, date range filtering, gross vs. net revenue breakdown, and dataset provenance badge. (Captured illustrative run; displayed figures may vary from live runtime.)*

[![Executive Decision Support Radar](media/decision-intelligence-radar.png)](media/decision-intelligence-radar.png)
*Figure 2: Executive Decision-Support Radar with 6 prioritized, evidence-backed commercial directives categorized by risk and impact. (Captured illustrative run; displayed figures may vary from live runtime.)*

---

### 2. 🔮 Production Forecast Lab
- **Multi-Horizon Forecasting:** Recursive 7-day, 14-day, and 30-day rollout projections.
- **Candidate Architectures:** XGBoost, Random Forest, and Ensemble blend via verified `ModelRegistry`.
- **Prediction Uncertainty:** Choose between `Heuristic Sqrt-Step` ($\hat{y}_t \pm 1.96 \cdot \sigma_{\text{res}} \sqrt{h}$) and `Calibrated OOF Residuals` (empirical quantiles).
- **Baseline Comparison:** Benchmark ML projections against the Seasonal Naive ($y_{t+h} = y_{t+h-7}$) recurrence standard.
- **Audit Persistence:** Automatically persists run configuration, origin date, and step-level bounds to SQLite storage with unique UUIDv4 run IDs.

[![Production Forecast Lab](media/forecast-lab.png)](media/forecast-lab.png)
*Figure 3: Multi-horizon recursive forecast generation (30-day projection) with candidate model selection, expanding uncertainty intervals, and run provenance logging. (Captured illustrative run; displayed figures may vary from live runtime.)*

---

### 3. 🎯 Forecast Performance & Reconciliation
- **Section A: Historical Out-of-Sample Backtesting:** 13 chronological rolling origins with strict target embargo. Displays MAE, RMSE, WAPE (Micro & Macro), Forecast Bias, and Interval Coverage comparing ML candidates against Seasonal-Naive.
- **Section B: Ground-Truth Realized Performance:** Reconciles past forecasts against newly observed actuals stored in SQLite database.
- **Pending Realization Safeguard:** Future forecast steps without ground truth remain explicitly marked as **Pending Realization** and are **never** filled with zero.

[![Forecast Performance Backtest Benchmarks](media/forecast-performance.png)](media/forecast-performance.png)
*Figure 4: Historical out-of-sample backtest benchmarks across 13 rolling origins comparing Candidate ML models against the Seasonal-Naive baseline. (Captured illustrative run; displayed figures may vary from live runtime.)*

[![Ground-Truth Forecast Reconciliation](media/forecast-reconciliation.png)](media/forecast-reconciliation.png)
*Figure 5: Ground-truth reconciliation protocol matching predictions against observed actuals with explicit 'Pending Realization' guards. (Captured illustrative run; displayed figures may vary from live runtime.)*

---

### 4. 🛡️ Data Quality & Ingestion Operations
- **Contract Ingestion:** Ingest raw CSV/Excel workbooks conforming to strict schema contracts.
- **Accounting Reconciliation:** Audits accepted rows, quarantined records, cancellations, and duplicate lines.
- **Data Freshness SLA:** Monitors days since last observation against configurable cadence thresholds (Daily, Hourly, Weekly).
- **Dataset Drift Monitoring:** Evaluates feature-level Population Stability Index (PSI) and Total Variation Distance (TVD) against promoted reference baselines.

[![Data Quality Workspace](media/data-quality.png)](media/data-quality.png)
*Figure 6: Data quality workspace enforcing schema contracts, auditing dataset provenance/checksums, and isolating quarantined records. (Captured illustrative run; displayed figures may vary from live runtime.)*

---

### 5. ⚠️ Explainable Time-Series Anomaly Monitor
- **Statistical Engine:** 14-day rolling median baseline with Median Absolute Deviation (MAD) spread.
- **Alert Typology:** Categorizes findings into `SALES_SURGE`, `SALES_DROP`, and `FORECAST_ERROR_ANOMALY`.
- **Operator Triage:** Workflow to review, investigate, and mark anomalies as `ACTIVE` or `RESOLVED` with auditable resolution reasons.
- **Seasonality Disclosure:** Prominent notice disclosing that intra-week variations may elevate z-scores, requiring operator contextual verification.

[![Explainable Anomaly Monitor](media/anomaly-monitor.png)](media/anomaly-monitor.png)
*Figure 7: Time-series statistical anomaly monitor displaying surges, drops, and triage status across the full historical sales series. (Captured illustrative run; displayed figures may vary from live runtime.)*

---

### 6. ⚙️ System Health & MLOps Governance
- **Database Diagnostics:** SQLite WAL mode status, schema version (`v2`), and audit log row counts.
- **Model Inventory:** Verified candidate architectures, feature vector contracts (13 features), and explicit non-fallback policies.
- **Honest Health Reporting:** Evaluates demo data truthfully, reporting `NOT APPLICABLE (HISTORICAL DEMO ARCHIVE)` instead of fabricating false green health checks.
- **MLOps Retraining Governance:** Formal protocol outlining Data Sufficiency ($\ge 90$ days), Degradation vs. Drift policy, and Champion-Challenger promotion gates.

[![System Health & MLOps Governance](media/system-health.png)](media/system-health.png)
*Figure 8: 4-dimensional ingestion SLA health cards (truthful 'NOT APPLICABLE' demo state), verified ML model inventory, and retraining governance policy. (Captured illustrative run; displayed figures may vary from live runtime.)*

---

## 📊 Dataset Provenance & Accounting Standard

SalesPulse AI ships with two verified demonstration datasets and supports arbitrary user uploads:

### 1. UCI Online Retail II (Primary Demo - UK, £ GBP)
- **Source:** UCI Machine Learning Repository ([DOI: 10.24432/C5BW33](https://doi.org/10.24432/C5BW33)).
- **Citation:** Chen, D. (2012). Online Retail II.
- **Licensing:** Distributed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Scope & Temporal Span:** Dec 1, 2009 – Dec 9, 2011 (739 Continuous Calendar Days).
- **Raw Volume:** 1,067,371 raw transaction rows across two annual workbook sheets.

#### Accounting Reconciliation Standard:
$$\text{Net Sales} = \text{Gross Sales} - \text{Realized Returns and Cancellations}$$

| Metric | Accounting Value | Technical Implementation |
| :--- | :--- | :--- |
| **Gross Sales Volume** | £21,032,607.72 | Sum of all positive line items ($Quantity > 0, Price > 0$) |
| **Returns & Cancellations** | £1,531,298.54 | Exact negative amount from cancelled invoices ($InvoiceNo \text{ starts with 'C'}$) |
| **Net Realized Revenue** | £19,501,309.18 | Gross Sales minus Returns |
| **Gross Units Dispatched** | 10,751,209 Units | Positive ordered quantities |
| **Unique Orders / Invoices** | 40,192 Invoices | Valid, non-cancelled commercial transactions |
| **Average Order Value (AOV)** | £485.20 / Order | Net Revenue divided by Unique Invoices |
| **Average Selling Price (ASP)** | £1.81 / Unit | Net Revenue divided by Units Dispatched (£1.96 Gross ASP) |
| **Return Rate** | 7.28% | Cumulative returns as a percentage of gross sales |

### 2. US Superstore (Secondary Demo - US, $ USD)
- **Scope:** 9,994 retail orders spanning 2014–2017 across Furniture, Office Supplies, and Technology.
- **Currency:** US Dollar ($ USD).
- **Isolation Policy:** Superstore data and UK Online Retail data maintain separate namespaces and are **never** mixed into a single sales series.

---

## 🛡️ Data Transparency & Known Operational Boundaries

### Data & Demo Mode Transparency
SalesPulse AI explicitly distinguishes between static demonstration archives and live ingestion pipelines:
- **Historical Demo State:** When operating with the included UCI Online Retail II dataset (spanning 2009–2011), the monitoring engine designates the Freshness SLA as `NOT APPLICABLE (HISTORICAL DEMO ARCHIVE)` rather than producing false green health alerts.
- **Live Ingestion State:** Live cadence monitoring, freshness SLAs, and drift evaluation dynamically activate when users upload current tabular records or configure scheduled pipeline workers.

### Documented Technical Boundaries
- **Time-Series Seasonality Limitation:** The Anomaly Monitor uses a 14-day rolling median with MAD dispersion. It does *not* automatically stratify recurring intra-week patterns (e.g. Saturday retail surges vs. weekday lulls). Elevated z-scores (~1.8–2.5) on recurring high-volume days are statistical signals requiring operational domain triage, not confirmed defects.
- **Compounding Forecast Uncertainty:** Recursive multi-step projections compound error variance over forward horizons. Prediction uncertainty bounds realistically expand with horizon step ($\sqrt{h}$); 30-day projection intervals are naturally wider than 7-day intervals.
- **Database Boundary:** The standard distribution is configured for local SQLite in Write-Ahead Logging (WAL) mode for lightweight single-node performance. Multi-tenant or high-concurrency environments should connect to an external PostgreSQL instance via SQLAlchemy URL configuration.

---

## 🧠 MLOps Retraining Governance & Rollback Protocol

A central principle of SalesPulse AI is that **data drift alone does not warrant automatic model retraining**:

```text
[New Ground Truth Data Ingested]
               │
               ▼
   [Data Sufficiency Check] ──(N < 90 Days)──► [HOLD: Awaiting History]
               │ (N ≥ 90 Days)
               ▼
     [Performance Check] ────(WAPE Shift ≤ 15%)─► [HEALTHY: Maintain Champion]
               │ (WAPE Shift > 15%)
               ▼
   [Candidate Model Training]
               │
               ▼
[Walk-Forward OOF Validation] ─(Candidate MAE ≥ Champion)─► [REJECT CHALLENGER]
               │ (Candidate Outperforms on MAE & WAPE across all horizons)
               ▼
    [Manual Approval Gate]
               │
               ▼
 [Promote Challenger to Production]
  (Archive previous champion SHA-256; automated rollback enabled)
```

---

## 🛠️ Installation & Execution

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/SwastikPandey1024/SalesPulse_AI.git
cd MLminorStreamlit

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 4. Run the Optional FastAPI Daemon
```bash
uvicorn salespulse.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.

---

## 🧪 Scientific Validation & Test Suite

The SalesPulse AI platform is backed by a rigorous test suite of **209 automated tests** covering domain contracts, time-series parsing, metric parity, drift detection, anomaly detection, persistence idempotency, monitoring status contracts, and recommendation rules.

```bash
# Verify Python syntax and compilation
python -m py_compile app.py

# Execute the full pytest test suite
pytest -q
```

### Verified Test Breakdown:
- `tests/test_observability_contract.py`: 11 passed (Monitoring contracts, 4D health synthesis, state precedence hierarchy, SQLite disconnection safety).
- `tests/test_analytics.py`: 24 passed (Surge/Drop detection, MAD zero handling, Seasonality disclosure, DecisionSupportEngine rules).
- `tests/test_drift.py`: 16 passed (PSI thresholds, TVD calculation, baseline profiling, sample sufficiency).
- `tests/test_monitoring.py`: 19 passed (Freshness SLA, CadenceEnum parsing, consecutive failures, sanitization).
- `tests/test_storage.py`: 16 passed (SQLite WAL initialization, schema v1->v2 migration, rollback, idempotent logging).
- `tests/test_services.py`: 8 passed (Numerical parity, prediction interval methods, ModelRegistry isolation).
- `tests/test_api.py`: 18 passed (FastAPI endpoints, error schemas, input validation).
- `tests/test_contracts.py`: 18 passed (RFC JSON compliance, grain detection, timestamp boundaries).
- `tests/test_features.py`: 4 passed (13-feature vector contracts, target leakage prevention).
- `tests/test_ingestion.py`: 7 passed (Tabular CSV/Excel readers, corrupted encoding handling).
- `tests/test_metrics.py`: 4 passed (MAE, RMSE, WAPE zero-division safety, coverage).
- `tests/test_time_series.py`: 6 passed (Daily aggregation, missing date zero-filling policy).
- `tests/test_dashboard.py`: 5 passed (Dashboard service initialization, database offline fallback).
- `tests/test_acceptance_remediation.py`: 30 passed (End-to-end acceptance across all 6 workspaces).

**Total: 209 passed, 0 failures.**

---

## 📄 License & Attribution

- **Application Software:** Licensed under the [MIT License](LICENSE).
- **UCI Online Retail II Dataset:** Distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.
- **Third-Party & Sample Data:** Sample datasets (e.g., US Superstore) and pre-trained demonstration artifacts are governed by their respective source licenses and terms as specified in [LICENSE](LICENSE).

---

<div align="center">
<strong>SalesPulse AI v3.0 Production</strong> • Built with rigorous engineering for real business operators.
</div>
