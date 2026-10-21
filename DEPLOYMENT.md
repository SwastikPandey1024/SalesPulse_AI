# 🌐 SalesPulse AI v3.0 — Deployment & Hosting Guide

- **Canonical Live Application:** [https://salespulseai.streamlit.app/](https://salespulseai.streamlit.app/)
- **Source Repository:** [https://github.com/SwastikPandey1024/SalesPulse_AI](https://github.com/SwastikPandey1024/SalesPulse_AI)
- **Hosting Provider:** [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- **Deployment Branch:** `main`
- **Primary Entrypoint:** `app.py`

---

## 1. Architecture & Deployment Topology Overview

SalesPulse AI Version 3.0 provides a dual-surface architecture:
1. **Frontend Operator Interface (`app.py`):** An interactive multi-workspace Streamlit dashboard deployed on **Streamlit Community Cloud**. It operates self-contained domain services, in-memory caching, and local SQLite audit logging.
2. **Optional REST API Service (`salespulse.api.app`):** A production-minded FastAPI service exposing RFC-compliant `/api/v1` endpoints for data contract validation, multi-horizon forecasts, and statistical anomaly detection.

> [!IMPORTANT]
> **API Hosting Boundary:** The public URL `https://salespulseai.streamlit.app/` hosts the Streamlit dashboard application. Streamlit Community Cloud runs `app.py` as a single web process; it does **not** expose or proxy the optional FastAPI background service (`uvicorn salespulse.api.app:app`). When deployed on Community Cloud, `app.py` automatically runs in local self-contained domain mode, utilizing internal domain services directly. For teams requiring the headless REST API, deploy the FastAPI application separately on a dedicated cloud service (e.g. AWS ECS, Google Cloud Run, Render, or an internal Kubernetes cluster).

---

## 2. Streamlit Community Cloud Deployment

The live application is deployed on Streamlit Community Cloud directly linked to the GitHub repository.

### 2.1 Repository & Application Configuration
- **Repository:** `SwastikPandey1024/SalesPulse_AI`
- **Branch:** `main`
- **Main file path:** `app.py`
- **App URL:** `https://salespulseai.streamlit.app/`
- **Python Runtime:** Python 3.11+
- **Configuration:** Custom UI theme defined in `.streamlit/config.toml` (dark mode palette, custom fonts, streamlined toolbar).

### 2.2 Dependency Resolution & Build Process
Streamlit Community Cloud automatically provisions a containerized environment based on the repository's [`requirements.txt`](requirements.txt):
- Core Data & Analytics: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `plotly`, `joblib`, `openpyxl`
- Persistence & API Frameworks: `sqlalchemy>=2.0.0`, `fastapi>=0.115.0`, `pydantic>=2.8.0`, `httpx>=0.27.0`, `python-multipart>=0.0.9`, `uvicorn>=0.30.0`

Official Streamlit Community Cloud documentation:
- [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Share your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)

### 2.3 Continuous Deployment & Update Workflow
Streamlit Community Cloud monitors the connected GitHub repository:
1. **Automated Sync:** Whenever commits are pushed to the `main` branch, Streamlit Cloud detects changes and initiates an automatic rebuild and hot reload.
2. **Safe Git Staging Practice:** Always stage specific, reviewed files rather than blanket commands:
   ```bash
   # Stage only targeted changes
   git add app.py salespulse/ README.md
   git commit -m "feat: improve forecast caching and update documentation"
   git push origin main
   ```
   *(Never execute `git add .` or `git add -A` without inspecting git status to prevent staging untracked data or secrets.)*
3. **Log & Status Inspection:**
   - Access deployment logs via the **"Manage App"** drawer located in the bottom-right corner of the live application at `https://salespulseai.streamlit.app/`.
   - Real-time build logs display package installation, system dependencies, container initialization, and application output.

---

## 3. Post-Deployment Smoke-Test Checklist

Following any deployment or container rebuild, execute this verification protocol across all six workspaces:

### ✅ Workspace 1: Sales Overview
- [ ] Application loads without Python traceback or red exception banners.
- [ ] Executive KPI cards display valid formatted numbers (£8.06M net revenue across full UCI span).
- [ ] Continuous Daily Trend chart renders with 7-day and 30-day moving averages.
- [ ] Decision Support Radar populates prioritized directives with quantifiable financial impact.
- [ ] "What-If" Scenario Simulator sliders adjust simulated revenue deltas reactively.

### ✅ Workspace 2: Forecast Lab
- [ ] Radio selector supports 7-Day, 14-Day, and 30-Day horizon adjustments.
- [ ] Candidate model selector allows switching between `XGBoost`, `RandomForest`, and composite `Ensemble`.
- [ ] Clicking **Generate Multi-Horizon Forecast** produces trajectory line and prediction uncertainty envelope.
- [ ] Provenance banner populates an immutable Run ID (`UUIDv4`) and origin timestamp.

### ✅ Workspace 3: Forecast Performance & Reconciliation
- [ ] Section A displays the 13-origin walk-forward CV benchmarks table comparing ML models against the Seasonal-Naive baseline.
- [ ] Section B (Ground-Truth Forecast Reconciliation) loads past persisted runs.
- [ ] Unobserved steps are explicitly guarded as `Pending Realization (Awaiting Actuals)` and never filled with zero.

### ✅ Workspace 4: Data Quality & Ingestion Operations
- [ ] Official dataset provenance citation (UCI Online Retail II; DOI: 10.24432/C5BW33) and SHA-256 checksums are visible.
- [ ] Audited counts show 1,041,670 accepted rows, 19,494 cancellations, and 6,207 quarantined price adjustments.
- [ ] Data Freshness SLA truthfully reports `NOT APPLICABLE (HISTORICAL DEMO ARCHIVE)` for static baseline data.

### ✅ Workspace 5: Explainable Anomaly Monitor
- [ ] Full series scatter/line chart renders statistical surge and drop markers.
- [ ] Seasonality notice banner is clearly visible, disclosing that the 14-day rolling median pools intra-week patterns.
- [ ] Finding details table displays robust z-score lineage and allows status triage (`ACTIVE` vs `RESOLVED`).

### ✅ Workspace 6: System Health & MLOps Control Center
- [ ] Database readiness probe verifies SQLite WAL mode and schema version `v2`.
- [ ] Candidate registry enumerates active verified model bundles (`model.pkl` and `rf_model.pkl`).
- [ ] Service architecture card indicates self-contained domain mode when optional FastAPI daemon is offline.
- [ ] MLOps Retraining Governance protocol displays the 3-stage promotion policy.

---

## 4. Security, Secrets & Data Privacy Guidelines

1. **Public Sandbox Privacy Warning:**
   The instance at `https://salespulseai.streamlit.app/` is a public demonstration sandbox. **Users and operators must never upload confidential financial sheets, PII (names, emails, phone numbers), or proprietary trade secrets to the hosted demo.**
2. **No Secrets in Code or Git:**
   - Database credentials, API tokens, and secret keys must **never** be committed to Git.
   - The repository tracks `.env.example` as a template. Local secrets reside in unversioned `.env` files (excluded by `.gitignore`).
3. **Streamlit Secrets Management:**
   When connecting to external enterprise databases (e.g. PostgreSQL), configure secrets via the Streamlit Community Cloud App Settings dashboard (**Settings -> Secrets**):
   ```toml
   # Streamlit Community Cloud Secrets configuration (TOML)
   [database]
   url = "postgresql://user:password@db.example.internal:5432/salespulse"
   ```
4. **Sanitized Error Reporting:**
   The application and API enforce error-sanitization wrappers (`sanitize_error_detail`) ensuring that local filesystem paths, environment variables, and stack traces are never exposed to browser clients.

---

## 5. Hosting Limitations & Production Operational Boundaries

| Dimension | Streamlit Community Cloud | Self-Hosted / Enterprise |
| :--- | :--- | :--- |
| **Compute / Memory** | Shared container (~1 GB RAM limit; sleeps after inactivity) | Dedicated vCPU and scalable memory allocation |
| **Storage Persistence** | Ephemeral container disk; local SQLite database resets upon container reboot or re-deploy | Persistent cloud volume or managed relational database (PostgreSQL) |
| **Network & Access** | Publicly accessible via HTTPS (`salespulseai.streamlit.app`) | Configurable behind corporate VPN, SSO, mTLS, or reverse proxy |
| **FastAPI Microservice** | Offline (Streamlit Cloud serves frontend only) | Fully containerized FastAPI daemon on port 8000 |
| **Data Scope** | Pre-loaded public demo datasets (UCI Online Retail II, US Superstore) | Private internal ERP/CRM transaction warehouses |

For high-concurrency enterprise workloads, package SalesPulse AI into a Docker container and configure `SALESPULSE_DATABASE_URL` pointing to an external PostgreSQL instance in Write-Ahead Logging mode.
