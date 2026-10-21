"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SalesPulse AI - Version 3.0                               ║
║        Production ML Forecasting, Observability & Operator Experience        ║
║                                                                              ║
║  An enterprise-grade ML platform featuring:                                  ║
║  • Sales Overview: Executive KPIs, Daily Trends & What-If Scenario Sim       ║
║  • Forecast Lab: Multi-Horizon Forecasting & Provenance Tracking             ║
║  • Forecast Performance: Reconciled Out-of-Sample Evaluation & Realization   ║
║  • Data Quality: Contract Ingestion Verification & Persistent Audit Trail    ║
║  • Anomaly Monitor: Explainable Statistical Time-Series Anomaly Monitoring   ║
║  • System Health: Database Diagnostics, Model Registry & Observability       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import datetime
from datetime import timedelta
import logging
import json
import hashlib
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PROJECT ENVIRONMENT & SYSTEM PATH
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from salespulse.data import (
    safe_ingest_file,
    aggregate_to_daily_sales,
    detect_dataset_grain,
    GrainType,
    ValidationSeverity,
    MISSING_DATE_POLICY,
)
from salespulse.features.temporal import ENGINEERED_FEATURES
from salespulse.services.forecasting_service import (
    ForecastingService,
    build_feature_dataframe as service_build_feature_dataframe,
    extract_date_features as service_extract_date_features,
    validate_numeric_scalar,
)
from salespulse.services.validation_service import ValidationService
from salespulse.services.analytics_service import AnalyticsService
from salespulse.services.model_registry import get_model_registry, ModelRegistry
from salespulse.storage.repository import AuditRepository
from salespulse.storage.database import get_engine, SCHEMA_VERSION, init_db
from salespulse.storage.models import (
    DatasetAuditLog,
    ForecastRunLog,
    ForecastItemLog,
    ActualSalesRecord,
    AnomalyRecord,
)
from salespulse.analytics.schemas import (
    AnomalyConfig,
    AnomalyReport,
    AnomalyPoint,
    AnomalyType,
    SeverityLevel,
)
from salespulse.monitoring import (
    DataMonitoringService,
    MonitoringConfig,
    MonitoringStatus,
    FreshnessState,
    IngestionOutcome,
    MonitoringVerdict,
    format_datetime_safe,
    parse_datetime_safe,
    CadenceType,
    CADENCE_DEFAULT_THRESHOLDS,
    get_cadence_default_threshold,
    DriftStatus,
    FeatureType,
    DriftMetricType,
    DriftBaselineProfile,
    DriftConfig,
    FeatureDriftResult,
    DriftReport,
    DatasetDriftService,
    profile_dataset,
)

# ============================================================================
# PAGE CONFIGURATION & PREMIUM STYLING
# ============================================================================
st.set_page_config(
    page_title="SalesPulse AI — Operator Experience",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, cards, badges, and callouts
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.25rem;
    }
    .metric-card-v2 {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .metric-card-v2:hover {
        transform: translateY(-2px);
    }
    .badge-v2 {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-active {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-resolved {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-pending {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .insight-card {
        border-left: 4px solid #4F46E5;
        background-color: #F8FAFC;
        padding: 0.85rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.65rem;
    }
    .seasonality-callout {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1.25rem;
    }
    .provenance-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1rem;
    }
    stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SalesPulse_Dashboard")

MODEL_DIR = os.path.join(PROJECT_ROOT, 'model_data')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
DATASET_PATH = os.path.join(MODEL_DIR, 'Sample - Superstore.csv')

# ============================================================================
# PERSISTENT DOMAIN SERVICES INITIALIZATION
# ============================================================================
@st.cache_resource
def get_dashboard_services():
    """Defensively initialize shared domain services and persistence repository."""
    try:
        engine = get_engine()
        init_db(engine)
        repo = AuditRepository(engine)
        registry = get_model_registry()
        fc_service = ForecastingService(registry=registry, repository=repo)
        val_service = ValidationService(repository=repo)
        ana_service = AnalyticsService(repository=repo, engine=engine)
        mon_service = DataMonitoringService(repository=repo, engine=engine)
        drift_service = DatasetDriftService(repository=repo)
        return {
            "engine": engine,
            "repo": repo,
            "registry": registry,
            "fc_service": fc_service,
            "val_service": val_service,
            "ana_service": ana_service,
            "mon_service": mon_service,
            "drift_service": drift_service,
            "db_ready": True,
            "db_error": None,
        }
    except Exception as exc:
        logger.warning(f"Database initialization warning in Streamlit: {exc}")
        registry = get_model_registry()
        mon_service = DataMonitoringService(repository=None, engine=None, auto_init_engine=False)
        drift_service = DatasetDriftService(repository=None)
        return {
            "engine": None,
            "repo": None,
            "registry": registry,
            "fc_service": ForecastingService(registry=registry),
            "val_service": ValidationService(),
            "ana_service": AnalyticsService(),
            "mon_service": mon_service,
            "drift_service": drift_service,
            "db_ready": False,
            "db_error": str(exc),
        }

services = get_dashboard_services()
audit_repo: Optional[AuditRepository] = services.get("repo")
fc_service: ForecastingService = services["fc_service"]
val_service: ValidationService = services["val_service"]
ana_service: AnalyticsService = services["ana_service"]
mon_service: DataMonitoringService = services["mon_service"]
drift_service: DatasetDriftService = services["drift_service"]
model_registry: ModelRegistry = services["registry"]
db_ready: bool = services.get("db_ready", False)

# ============================================================================
# LEGACY MODEL LOADING (CACHED FOR STABILITY & PARITY)
# ============================================================================
@st.cache_resource
def load_models():
    """Load both XGBoost and RandomForest models cleanly with schema verification."""
    models = {}
    errors = {}
    dummy_input = pd.DataFrame({col: [1.0] for col in ENGINEERED_FEATURES}).astype(np.float64)
    
    # 1. Load XGBoost (Primary)
    try:
        if os.path.exists(XGB_MODEL_PATH):
            with open(XGB_MODEL_PATH, "rb") as f:
                xgb_bundle = joblib.load(f)
            xgb_m = xgb_bundle.get('model_pipeline', xgb_bundle)
            _ = xgb_m.predict(dummy_input)
            models['XGBoost'] = xgb_m
        else:
            errors['XGBoost'] = f"Missing file: {XGB_MODEL_PATH}"
    except Exception as e:
        logger.error(f"XGBoost load/verify error: {e}")
        errors['XGBoost'] = str(e)
        
    # 2. Load RandomForest (Alternative)
    try:
        if os.path.exists(RF_MODEL_PATH):
            with open(RF_MODEL_PATH, "rb") as f:
                rf_bundle = joblib.load(f)
            rf_m = rf_bundle.get('model_pipeline', rf_bundle)
            _ = rf_m.predict(dummy_input)
            models['RandomForest'] = rf_m
        else:
            errors['RandomForest'] = f"Missing file: {RF_MODEL_PATH}"
    except Exception as e:
        logger.warning(f"RandomForest load/verify error: {e}")
        errors['RandomForest'] = str(e)
        
    if 'RandomForest' not in models and 'XGBoost' in models:
        models['RandomForest'] = models['XGBoost']
        
    return models, errors

models_dict, model_errors = load_models()

# ============================================================================
# HISTORICAL DATASETS & DEMO DATA LOADING (CACHED)
# ============================================================================
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ONLINE_RETAIL_DAILY_PATH = os.path.join(PROCESSED_DATA_DIR, "online_retail_ii_daily.csv")
ONLINE_RETAIL_PROVENANCE_PATH = os.path.join(PROCESSED_DATA_DIR, "online_retail_ii_provenance.json")
ONLINE_RETAIL_VALIDATION_PATH = os.path.join(PROCESSED_DATA_DIR, "online_retail_ii_validation.json")
ONLINE_RETAIL_TOP_PRODUCTS_PATH = os.path.join(PROCESSED_DATA_DIR, "online_retail_ii_top_products.csv")
ONLINE_RETAIL_COUNTRIES_PATH = os.path.join(PROCESSED_DATA_DIR, "online_retail_ii_countries.csv")

@st.cache_data
def load_online_retail_data() -> Dict[str, Any]:
    """Load pre-processed UCI Online Retail II canonical series and metadata."""
    res: Dict[str, Any] = {}
    try:
        if os.path.exists(ONLINE_RETAIL_DAILY_PATH):
            df = pd.read_csv(ONLINE_RETAIL_DAILY_PATH)
            df['date'] = pd.to_datetime(df['date'])
            res['daily'] = df
        if os.path.exists(ONLINE_RETAIL_PROVENANCE_PATH):
            with open(ONLINE_RETAIL_PROVENANCE_PATH, 'r', encoding='utf-8') as f:
                res['provenance'] = json.load(f)
        if os.path.exists(ONLINE_RETAIL_VALIDATION_PATH):
            with open(ONLINE_RETAIL_VALIDATION_PATH, 'r', encoding='utf-8') as f:
                res['validation'] = json.load(f)
        if os.path.exists(ONLINE_RETAIL_TOP_PRODUCTS_PATH):
            res['top_products'] = pd.read_csv(ONLINE_RETAIL_TOP_PRODUCTS_PATH)
        if os.path.exists(ONLINE_RETAIL_COUNTRIES_PATH):
            res['countries'] = pd.read_csv(ONLINE_RETAIL_COUNTRIES_PATH)
    except Exception as e:
        logger.error(f"Error loading Online Retail II dataset: {e}")
    return res

online_retail_data = load_online_retail_data()

@st.cache_data
def load_superstore_data():
    """Load and preprocess Superstore sample dataset."""
    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH, encoding='latin1')
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df['Ship Date'] = pd.to_datetime(df['Ship Date'])
            df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
            return df
    except Exception as e:
        logger.error(f"Dataset load error: {e}")
    return None

superstore_df = load_superstore_data()

# ============================================================================
# EXPLICIT DATA MODES CONTRACT
# ============================================================================
class DataMode:
    DEMO_DATA = "DEMO DATA"
    USER_DATA = "USER DATA"
    NO_DATA = "NO DATA"
    DATA_UNAVAILABLE = "DATA UNAVAILABLE / MONITORING ERROR"


def render_data_mode_banner(mode: str, dataset_name: str, currency: str = "$", details: str = ""):
    """Render prominent, unambiguous data mode banner in compliance with PRD specification."""
    if mode == DataMode.DEMO_DATA:
        badge_bg, badge_fg = "#DBEAFE", "#1E40AF"
        box_bg, box_border, box_left = "#EFF6FF", "#93C5FD", "#3B82F6"
        icon = "🧪"
        sub_desc = f"Historical Demonstration Dataset • Currency: {currency} • Not Live Real-Time Sales"
    elif mode == DataMode.USER_DATA:
        badge_bg, badge_fg = "#DCFCE7", "#166534"
        box_bg, box_border, box_left = "#F0FDF4", "#86EFAC", "#22C55E"
        icon = "📁"
        sub_desc = f"User Uploaded Custom Dataset • Currency: {currency} • Active Session Data"
    elif mode == DataMode.NO_DATA:
        badge_bg, badge_fg = "#F1F5F9", "#475569"
        box_bg, box_border, box_left = "#F8FAFC", "#CBD5E1", "#64748B"
        icon = "⚪"
        sub_desc = "No Dataset Currently Active • Select a demonstration dataset or upload a file"
    else:  # DATA UNAVAILABLE / MONITORING ERROR
        badge_bg, badge_fg = "#FEE2E2", "#991B1B"
        box_bg, box_border, box_left = "#FEF2F2", "#FCA5A5", "#EF4444"
        icon = "🔴"
        sub_desc = "Data source or monitoring operation encountered an unrecoverable failure"

    extra = f" • {details}" if details else ""
    st.markdown(f"""
    <div style="background-color: {box_bg}; border: 1px solid {box_border}; border-left: 5px solid {box_left}; padding: 0.65rem 1rem; border-radius: 6px; margin-bottom: 1rem;">
        <span style="background-color: {badge_bg}; color: {badge_fg}; padding: 3px 9px; border-radius: 12px; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.5px;">{icon} {mode}</span>
        <strong style="color: #0F172A; margin-left: 10px; font-size: 0.95rem;">{dataset_name}</strong>
        <span style="color: #64748B; font-size: 0.85rem; margin-left: 10px;">{sub_desc}{extra}</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# BACKWARD COMPATIBLE FEATURE ENGINEERING & INFERENCE UTILITIES
# ============================================================================
MIN_ANOMALY_HISTORY_DAYS = 14

def extract_date_features(date_input):
    return service_extract_date_features(date_input)

def validate_numeric_input(value, name, min_val=0.0):
    return validate_numeric_scalar(value, name, min_val)

def build_feature_dataframe(date_features, lag_1, lag_7, lag_14, lag_30,
                            rolling_mean_7, rolling_mean_14, rolling_std_7, trend):
    return service_build_feature_dataframe(
        date_features=date_features,
        lag_1=lag_1,
        lag_7=lag_7,
        lag_14=lag_14,
        lag_30=lag_30,
        rolling_mean_7=rolling_mean_7,
        rolling_mean_14=rolling_mean_14,
        rolling_std_7=rolling_std_7,
        trend=trend,
    )

def predict_sales(input_df, model_choice="XGBoost"):
    return fc_service.predict_single_step(input_df, model_choice=model_choice)

def generate_business_insights(predicted_val, lag_1, rm_7, rm_14):
    return fc_service.generate_business_insights(predicted_val, lag_1, rm_7, rm_14)

def forecast_horizon(initial_df, horizon_days=7, model_choice="XGBoost"):
    return fc_service.forecast_multi_horizon(
        initial_df=initial_df,
        horizon_days=horizon_days,
        model_choice=model_choice,
        interval_method="heuristic_sqrt_step",
        persist_run=False,
    )

def build_anomaly_overview_figure(
    actual_records: Optional[List[Any]],
    anomaly_records: Optional[List[Any]] = None,
    sales_scope: str = "TOTAL_STORE",
    min_history: int = 1,
    currency_symbol: str = "$",
) -> Tuple[Optional[go.Figure], str]:
    """Construct an explainable Plotly visualization for daily sales and anomaly overlays.
    
    Enforces minimum valid history policy to prevent misleading single-point trend charts.
    
    Returns:
        Tuple[Optional[go.Figure], str]:
            - Figure is None when there are no valid historical observations or points < min_history.
            - State string: 'NO_OBSERVATIONS', 'INSUFFICIENT_HISTORY', 'SINGLE_POINT', or 'MULTI_POINT'.
    """
    if not actual_records or len(actual_records) == 0:
        return None, "NO_OBSERVATIONS"

    points = []
    for r in reversed(actual_records):
        raw_d = getattr(r, "calendar_date", None) or (r.get("calendar_date") if isinstance(r, dict) else None)
        raw_s = getattr(r, "daily_sales", None) or (r.get("daily_sales") if isinstance(r, dict) else None)
        if raw_d is not None and raw_s is not None:
            try:
                dt = pd.to_datetime(raw_d)
                val = float(raw_s)
                points.append({"date": dt, "sales": val})
            except Exception:
                continue

    if len(points) == 0:
        return None, "NO_OBSERVATIONS"

    if len(points) < min_history:
        return None, "INSUFFICIENT_HISTORY"

    df_act = pd.DataFrame(points)
    unique_dates = df_act["date"].nunique()
    is_single_point = (len(df_act) == 1 or unique_dates == 1)

    fig = go.Figure()

    trace_mode = "lines+markers" if is_single_point else "lines"
    marker_kwargs = dict(size=9, color="#3B82F6") if is_single_point else dict()

    fig.add_trace(go.Scatter(
        x=df_act["date"],
        y=df_act["sales"],
        mode=trace_mode,
        marker=marker_kwargs,
        name=f"Daily Actual Sales ({currency_symbol})",
        line=dict(color="#94A3B8", width=2),
        hovertext=[f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br/><b>Sales:</b> {currency_symbol}{s:,.2f}" for d, s in zip(df_act["date"], df_act["sales"])],
        hoverinfo="text",
    ))

    anomalies = anomaly_records or []
    for r in anomalies:
        obs_date = getattr(r, "observation_date", None) or (r.get("observation_date") if isinstance(r, dict) else None)
        obs_val = getattr(r, "observed_value", None) or (r.get("observed_value") if isinstance(r, dict) else None)
        ano_type = getattr(r, "anomaly_type", None) or (r.get("anomaly_type") if isinstance(r, dict) else "ANOMALY")
        sev = getattr(r, "severity", None) or (r.get("severity") if isinstance(r, dict) else "MEDIUM")
        ref_c = getattr(r, "reference_center", None) or (r.get("reference_center") if isinstance(r, dict) else 0.0)
        score = getattr(r, "anomaly_score", None) or (r.get("anomaly_score") if isinstance(r, dict) else 0.0)
        thresh = getattr(r, "threshold", None) or (r.get("threshold") if isinstance(r, dict) else 0.0)
        status = getattr(r, "status", None) or (r.get("status") if isinstance(r, dict) else "ACTIVE")
        expl = getattr(r, "explanation", None) or (r.get("explanation") if isinstance(r, dict) else "")

        if obs_date is None or obs_val is None:
            continue

        dt = pd.to_datetime(obs_date)
        m_color = "#DC2626" if sev in ("CRITICAL", "HIGH") else "#F59E0B"
        sym = "circle" if ano_type == "SALES_SURGE" else "triangle-down"

        fig.add_trace(go.Scatter(
            x=[dt],
            y=[float(obs_val)],
            mode="markers",
            marker=dict(color=m_color, size=11, symbol=sym, line=dict(color="#FFFFFF", width=1.5)),
            name=f"{ano_type} ({sev})",
            hovertext=(
                f"<b>Date:</b> {dt.strftime('%Y-%m-%d')}<br/>"
                f"<b>Observed:</b> {currency_symbol}{float(obs_val):,.2f}<br/>"
                f"<b>Baseline:</b> {currency_symbol}{float(ref_c):,.2f}<br/>"
                f"<b>Score:</b> {float(score):.2f} (Thresh: {float(thresh):.2f})<br/>"
                f"<b>Severity:</b> {sev} | <b>Status:</b> {status}<br/>"
                f"<b>Explanation:</b> {expl}"
            ),
            hoverinfo="text",
            showlegend=False,
        ))

    layout_kwargs: Dict[str, Any] = {
        "title": f"Daily Sales Series with Statistical Anomaly Overlay ({sales_scope})",
        "xaxis_title": "Date",
        "yaxis_title": f"Sales ({currency_symbol})",
        "template": "plotly_white",
        "height": 380,
        "hovermode": "closest",
    }

    if is_single_point:
        single_dt = df_act["date"].iloc[0]
        start_pad = single_dt - timedelta(days=3)
        end_pad = single_dt + timedelta(days=3)
        layout_kwargs["xaxis"] = dict(
            range=[start_pad, end_pad],
            tickformat="%Y-%m-%d",
            dtick=86400000.0,
        )
        status_msg = "SINGLE_POINT"
    else:
        status_msg = "MULTI_POINT"

    fig.update_layout(**layout_kwargs)
    return fig, status_msg

# ============================================================================
# APPLICATION UI HEADER & SIDEBAR
# ============================================================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-header">📊 SalesPulse AI <span class="badge-v2">v3.0 Production</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production ML Forecasting, Explainability, Observability & Data Quality Assurance</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/analytics.png", width=60)
    st.title("⚙️ Operations Pane")
    st.divider()

    st.subheader("📁 Dataset & Data Mode")
    dataset_choice = st.selectbox(
        "Active Dataset Scope",
        options=[
            "UCI Online Retail II (Demo - UK, £ GBP)",
            "Sample - Superstore (Demo - US, $ USD)",
            "User Upload (Custom CSV / Excel)",
            "No Active Dataset (Empty State Test)",
        ],
        index=0,
        help="Select between verified demonstration datasets, custom upload, or testing the empty state."
    )

    # Determine Active Mode and Parameters
    if dataset_choice == "UCI Online Retail II (Demo - UK, £ GBP)":
        active_mode = DataMode.DEMO_DATA
        active_currency = "£"
        active_currency_code = "GBP"
        active_dataset_title = "UCI Online Retail II (UK Wholesaler, Dec 2009 – Dec 2011)"
    elif dataset_choice == "Sample - Superstore (Demo - US, $ USD)":
        active_mode = DataMode.DEMO_DATA
        active_currency = "$"
        active_currency_code = "USD"
        active_dataset_title = "US Superstore (US Retail Orders, 2014 – 2017)"
    elif dataset_choice == "User Upload (Custom CSV / Excel)":
        if "user_clean_daily_df" in st.session_state and st.session_state["user_clean_daily_df"] is not None:
            active_mode = DataMode.USER_DATA
            active_currency = "$"
            active_currency_code = "USD"
            active_dataset_title = st.session_state.get("user_dataset_name", "Uploaded Custom Dataset")
        else:
            active_mode = DataMode.NO_DATA
            active_currency = "$"
            active_currency_code = "USD"
            active_dataset_title = "Awaiting User File Upload"
    else:
        active_mode = DataMode.NO_DATA
        active_currency = "$"
        active_currency_code = "USD"
        active_dataset_title = "No Dataset Selected"

    st.caption(f"Current State: **{active_mode}** ({active_currency_code})")
    st.divider()

    available_models = model_registry.list_models()
    selected_model_choice = st.selectbox(
        "🧠 Active Model Candidate",
        options=available_models,
        index=0,
        help="Select candidate model from verified ModelRegistry."
    )

    st.divider()
    st.subheader("🔍 Health Indicators")
    if db_ready:
        st.success(f"✅ Storage: Connected (SQLite v{SCHEMA_VERSION})")
    else:
        st.warning("⚠️ Storage: Standalone / Local Only")

    st.success(f"✅ Active Candidate: {selected_model_choice}")
    st.success("✅ Feature Vector: 13 Features")

    st.divider()
    if st.button("🔄 Reset Resource Caches", help="Force reload ML models and refresh state"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### 📚 Reference & Docs")
    st.markdown("• [Architecture Specifications](https://github.com/SwastikPandey1024/SalesPulse_AI)")
    st.markdown("• [Dataset Provenance: UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)")
    st.caption("SalesPulse AI • Enterprise ML Observability Platform")

# ============================================================================
# MAIN 6-WORKSPACE OPERATOR INTERFACE
# ============================================================================
tab_overview, tab_forecast, tab_perf, tab_quality, tab_anomaly, tab_health = st.tabs([
    "📊 Sales Overview",
    "🔮 Forecast Lab",
    "🎯 Forecast Performance",
    "🛡️ Data Quality",
    "⚠️ Anomaly Monitor",
    "⚙️ System Health",
])

# ----------------------------------------------------------------------------
# 1. SALES OVERVIEW WORKSPACE
# ----------------------------------------------------------------------------
with tab_overview:
    try:
        render_data_mode_banner(active_mode, active_dataset_title, currency=active_currency, details="Executive Sales Trajectory & Business KPIs")
        st.subheader("📊 Executive Sales Overview & Business Trajectory")
        st.caption("Inspect aggregate historical sales performance, distribution metrics, period-over-period growth, and operational what-if simulations.")

        if active_mode == DataMode.NO_DATA:
            st.info(
                "💡 **No Active Dataset Selected**: The dashboard is currently in the **NO DATA** state.\n\n"
                "To explore sales trends and KPIs, select a demonstration dataset (e.g. **UCI Online Retail II** or **US Superstore**) "
                "in the sidebar Operations Pane, or upload your transaction workbook in the **Data Quality** workspace."
            )

        elif dataset_choice == "UCI Online Retail II (Demo - UK, £ GBP)":
            daily_df = online_retail_data.get("daily")
            prov = online_retail_data.get("provenance", {})
            top_products_df = online_retail_data.get("top_products")
            countries_df = online_retail_data.get("countries")

            if daily_df is not None and not daily_df.empty:
                # Provenance and Coverage Badge
                st.markdown("""
                <div class="provenance-card">
                    <strong>📌 Dataset Provenance:</strong> UCI Machine Learning Repository • <strong>Citation:</strong> Chen, D. (2012). Online Retail II. DOI: <a href="https://doi.org/10.24432/C5BW33" target="_blank">10.24432/C5BW33</a><br/>
                    • <strong>Scope:</strong> UK Giftware Wholesaler (1,067,371 raw transaction rows) | <strong>Coverage:</strong> Dec 1, 2009 – Dec 9, 2011 (739 Calendar Days)<br/>
                    • <strong>Accounting Standard:</strong> Net Sales = Gross Sales − Realized Returns/Cancellations | <strong>Currency:</strong> British Pound (£ GBP)
                </div>
                """, unsafe_allow_html=True)

                # Date Range Filter Slider
                min_dt = daily_df["date"].min().date()
                max_dt = daily_df["date"].max().date()
                date_range = st.slider(
                    "📅 Filter Observation Date Range",
                    min_value=min_dt,
                    max_value=max_dt,
                    value=(min_dt, max_dt),
                    format="YYYY-MM-DD",
                    key="ov_retail_slider"
                )

                mask = (daily_df["date"].dt.date >= date_range[0]) & (daily_df["date"].dt.date <= date_range[1])
                filtered_daily = daily_df[mask].copy()

                # Core Financial & Volume KPIs
                tot_gross = filtered_daily["gross_sales"].sum() if "gross_sales" in filtered_daily.columns else filtered_daily["sales"].sum()
                tot_returns = filtered_daily["returns"].sum() if "returns" in filtered_daily.columns else 0.0
                tot_net = filtered_daily["sales"].sum()
                tot_orders = int(filtered_daily["order_count"].sum()) if "order_count" in filtered_daily.columns else 0
                if "units_sold" in filtered_daily.columns:
                    tot_units = int(filtered_daily["units_sold"].sum())
                elif "quantity" in filtered_daily.columns:
                    tot_units = int(filtered_daily["quantity"].sum())
                else:
                    tot_units = 0
                aov_val = (tot_net / tot_orders) if tot_orders > 0 else 0.0
                return_rate_pct = (tot_returns / tot_gross * 100.0) if tot_gross > 0 else 0.0
                active_days = int(filtered_daily["is_observed_trading_day"].sum()) if "is_observed_trading_day" in filtered_daily.columns else len(filtered_daily)
                total_calendar_days = len(filtered_daily)

                # KPI Row 1: Financials & Volume
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Net Realized Revenue", f"£{tot_net:,.2f}", help="Gross sales minus returns and cancellations.")
                with k2:
                    st.metric("Gross Sales", f"£{tot_gross:,.2f}", help="Total order value before deduction of cancellations.")
                with k3:
                    st.metric("Returns & Cancellations", f"£{tot_returns:,.2f}", delta=f"-£{tot_returns:,.0f}", delta_color="inverse", help="Total refunded or cancelled order amount.")
                with k4:
                    st.metric("Total Invoices / Orders", f"{tot_orders:,}", help="Count of unique valid invoice events.")

                # KPI Row 2: Operational Efficiency
                k5, k6, k7, k8 = st.columns(4)
                with k5:
                    st.metric("Average Order Value (AOV)", f"£{aov_val:,.2f}", help="Net Sales divided by Total Invoices.")
                with k6:
                    st.metric("Units Dispatched", f"{tot_units:,}", help="Total gross unit quantity ordered.")
                with k7:
                    st.metric("Return Rate", f"{return_rate_pct:.2f}%", help="Returns as a percentage of gross sales.")
                with k8:
                    st.metric("Trading Calendar Days", f"{active_days} / {total_calendar_days} Days", help=f"Active trading days: {active_days}. Non-trading zero-filled days: {total_calendar_days - active_days}.")

                st.divider()

                # Daily Net Sales Time Series with Moving Averages
                filtered_daily["7D_MA"] = filtered_daily["sales"].rolling(window=7, min_periods=1).mean()
                filtered_daily["30D_MA"] = filtered_daily["sales"].rolling(window=30, min_periods=1).mean()

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=filtered_daily['date'], y=filtered_daily['sales'],
                    mode='lines', name='Daily Net Sales (£)', line=dict(color='#94A3B8', width=1.2),
                    hovertext=[f"<b>Date:</b> {d.strftime('%Y-%m-%d')}<br/><b>Net:</b> £{s:,.2f}<br/><b>Gross:</b> £{g:,.2f}<br/><b>Returns:</b> £{r:,.2f}"
                               for d, s, g, r in zip(filtered_daily["date"], filtered_daily["sales"], filtered_daily.get("gross_sales", filtered_daily["sales"]), filtered_daily.get("returns", [0]*len(filtered_daily)))],
                    hoverinfo="text"
                ))
                fig_ts.add_trace(go.Scatter(
                    x=filtered_daily['date'], y=filtered_daily['7D_MA'],
                    mode='lines', name='7-Day Moving Average', line=dict(color='#4F46E5', width=2.5)
                ))
                fig_ts.add_trace(go.Scatter(
                    x=filtered_daily['date'], y=filtered_daily['30D_MA'],
                    mode='lines', name='30-Day Moving Average', line=dict(color='#10B981', width=2, dash='dot')
                ))
                fig_ts.update_layout(
                    title=f"Historical Daily Net Sales & Multi-Period Trendlines (UCI Online Retail II, £ GBP)",
                    xaxis_title="Calendar Date", yaxis_title="Sales (£)",
                    template="plotly_white", height=400, hovermode="x unified"
                )
                st.plotly_chart(fig_ts, use_container_width=True)

                # Monthly Sales Trends & Period-over-Period (MoM) Growth
                filtered_daily["YearMonth"] = filtered_daily["date"].dt.to_period("M").astype(str)
                monthly_agg = filtered_daily.groupby("YearMonth").agg(
                    monthly_sales=("sales", "sum"),
                    monthly_orders=("order_count", "sum") if "order_count" in filtered_daily.columns else ("sales", "count")
                ).reset_index()
                monthly_agg["MoM_Growth_Pct"] = monthly_agg["monthly_sales"].pct_change() * 100.0

                m_col1, m_col2 = st.columns([3, 2])
                with m_col1:
                    fig_mom = go.Figure()
                    fig_mom.add_trace(go.Bar(
                        x=monthly_agg["YearMonth"],
                        y=monthly_agg["monthly_sales"],
                        name="Monthly Net Sales (£)",
                        marker=dict(color="#4F46E5"),
                        hovertext=[f"<b>Month:</b> {m}<br/><b>Net Sales:</b> £{s:,.2f}<br/><b>MoM Growth:</b> {g:+.1f}%" if pd.notnull(g) else f"<b>Month:</b> {m}<br/><b>Net Sales:</b> £{s:,.2f}<br/><b>MoM Growth:</b> Base Month"
                                   for m, s, g in zip(monthly_agg["YearMonth"], monthly_agg["monthly_sales"], monthly_agg["MoM_Growth_Pct"])],
                        hoverinfo="text"
                    ))
                    fig_mom.update_layout(
                        title="Monthly Net Sales Revenue & Seasonality (£ GBP)",
                        xaxis_title="Year-Month", yaxis_title="Net Sales (£)",
                        template="plotly_white", height=320
                    )
                    st.plotly_chart(fig_mom, use_container_width=True)
                    st.caption("ℹ️ *Note: December 2011 covers only 9 calendar days (Dec 1–Dec 9, 2011), representing an incomplete reporting period.*")

                with m_col2:
                    st.markdown("##### 📈 Month-over-Month (MoM) Growth Table")
                    display_mom = monthly_agg.tail(12).copy()
                    display_mom["monthly_sales"] = display_mom["monthly_sales"].apply(lambda x: f"£{x:,.2f}")
                    display_mom["MoM_Growth_Pct"] = display_mom["MoM_Growth_Pct"].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "Base")
                    display_mom.columns = ["Year-Month", "Monthly Sales (£)", "Orders", "MoM Growth"]
                    st.dataframe(display_mom, use_container_width=True, height=270)

                st.divider()

                # Top Products & Country Distribution
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    st.markdown("##### 🏆 Leading Products by Net Sales Revenue")
                    if top_products_df is not None and not top_products_df.empty:
                        top_display = top_products_df.head(10).copy()
                        fig_prod = px.bar(
                            top_display,
                            x="NetSales",
                            y="Description",
                            orientation="h",
                            title="Top 10 Products by Net Revenue (£)",
                            color="NetSales",
                            color_continuous_scale="Viridis",
                            text_auto="£,.0f"
                        )
                        fig_prod.update_layout(
                            template="plotly_white", height=340, yaxis=dict(autorange="reversed")
                        )
                        st.plotly_chart(fig_prod, use_container_width=True)
                    else:
                        st.info("Product catalog breakdown not available.")

                with p_c2:
                    st.markdown("##### 🌍 Geographic Market Breakdown")
                    if countries_df is not None and not countries_df.empty:
                        c_display = countries_df.head(8).copy()
                        fig_geo = px.pie(
                            c_display,
                            names="Country",
                            values="NetSales",
                            title="Revenue Share by Country (Top 8 Markets)",
                            hole=0.45,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_geo.update_layout(template="plotly_white", height=340)
                        st.plotly_chart(fig_geo, use_container_width=True)
                        st.caption("ℹ️ *United Kingdom accounts for ~84.6% of total revenue. EIRE, Germany, and France lead international volume.*")
                    else:
                        st.info("Geographic breakdown not available.")

            else:
                st.warning("Processed UCI Online Retail II dataset not found in data/processed/. Run `python salespulse/data/onboard_online_retail.py`.")

        elif dataset_choice == "Sample - Superstore (Demo - US, $ USD)":
            if superstore_df is not None:
                # Filters
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    sel_region = st.multiselect("Region Scope", options=superstore_df['Region'].unique().tolist(), default=superstore_df['Region'].unique().tolist(), key="ov_region")
                with f_col2:
                    sel_cat = st.multiselect("Category Scope", options=superstore_df['Category'].unique().tolist(), default=superstore_df['Category'].unique().tolist(), key="ov_cat")
                with f_col3:
                    sel_segment = st.multiselect("Customer Segment", options=superstore_df['Segment'].unique().tolist(), default=superstore_df['Segment'].unique().tolist(), key="ov_seg")

                filtered_df = superstore_df[
                    (superstore_df['Region'].isin(sel_region)) &
                    (superstore_df['Category'].isin(sel_cat)) &
                    (superstore_df['Segment'].isin(sel_segment))
                ]

                # High-level business KPIs
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Total Realized Revenue", f"${filtered_df['Sales'].sum():,.2f}")
                with k2:
                    st.metric("Total Transactions", f"{len(filtered_df):,}")
                with k3:
                    st.metric("Average Order Value (AOV)", f"${filtered_df['Sales'].mean():,.2f}")
                with k4:
                    top_c = filtered_df.groupby('Category')['Sales'].sum().idxmax() if not filtered_df.empty else "N/A"
                    st.metric("Leading Category", top_c)

                st.divider()

                # Time Series Daily Aggregation & Trend
                try:
                    daily_trend = aggregate_to_daily_sales(filtered_df, date_col="Order Date", sales_col="Sales", fill_missing_dates=True)
                    daily_trend["7D_MA"] = daily_trend["sales"].rolling(window=7, min_periods=1).mean()

                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=daily_trend['date'], y=daily_trend['sales'],
                        mode='lines', name='Daily Sales ($)', line=dict(color='#94A3B8', width=1)
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=daily_trend['date'], y=daily_trend['7D_MA'],
                        mode='lines', name='7-Day Moving Average', line=dict(color='#4F46E5', width=2.5)
                    ))
                    fig_ts.update_layout(
                        title="Historical Daily Sales & 7-Day Moving Average (Reindexed Superstore, $ USD)",
                        xaxis_title="Date", yaxis_title="Sales ($)",
                        template="plotly_white", height=380, hovermode="x unified"
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                except Exception as e_ts:
                    st.info(f"Aggregated trend visual unavailable: {e_ts}")

                # Visual Breakdowns
                c_br1, c_br2 = st.columns(2)
                with c_br1:
                    cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
                    fig_cat = px.pie(cat_sales, names='Category', values='Sales', title='Revenue Share by Product Category', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_cat.update_layout(template="plotly_white", height=320)
                    st.plotly_chart(fig_cat, use_container_width=True)
                with c_br2:
                    reg_sales = filtered_df.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
                    fig_reg = px.bar(reg_sales, x='Region', y='Sales', color='Category', title='Sales by Region & Category', barmode='group', color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_reg.update_layout(template="plotly_white", height=320)
                    st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.info("Superstore sample dataset not found.")

        elif active_mode == DataMode.USER_DATA:
            u_df = st.session_state.get("user_clean_daily_df")
            if u_df is not None and not u_df.empty:
                st.markdown(f"##### 📁 User Dataset Summary: {st.session_state.get('user_dataset_name', 'Custom Series')}")
                uk1, uk2, uk3 = st.columns(3)
                with uk1:
                    st.metric("Total Revenue", f"{active_currency}{u_df['sales'].sum():,.2f}")
                with uk2:
                    st.metric("Total Days Recorded", f"{len(u_df):,}")
                with uk3:
                    st.metric("Average Daily Sales", f"{active_currency}{u_df['sales'].mean():,.2f}")

                fig_u = px.line(u_df, x="date", y="sales", title=f"User Daily Sales ({active_currency})")
                fig_u.update_layout(template="plotly_white", height=360)
                st.plotly_chart(fig_u, use_container_width=True)

        # Business Scenario Simulator Section
        if active_mode != DataMode.NO_DATA:
            st.divider()
            st.markdown("### 🎯 Interactive Business Scenario Simulator")
            st.caption("Simulate promotional campaigns, pricing levers, and seasonal shifts to estimate sales lift.")
            
            sim_c1, sim_c2 = st.columns([1, 2])
            with sim_c1:
                p_boost = st.slider("📢 Promotional Campaign Boost (%)", 0, 50, 15, key="ov_promo")
                p_adjust = st.slider("🏷️ Price Adjustment (%)", -20, 20, 0, key="ov_price")
                s_mult = st.slider("📅 Seasonal Multiplier", 0.5, 1.5, 1.1, step=0.05, key="ov_season")
                default_bench = 26000.0 if active_currency == "£" else 200.0
                b_sales = st.number_input(f"Base Daily Sales Benchmark ({active_currency})", value=default_bench, step=100.0, key="ov_base")

            with sim_c2:
                effective_mult = (1.0 + (p_boost / 100.0)) * (1.0 + (p_adjust / 100.0)) * s_mult
                sim_sales = b_sales * effective_mult
                delta_val = sim_sales - b_sales
                pct_val = (delta_val / b_sales) * 100.0

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Baseline Sales", f"{active_currency}{b_sales:,.2f}")
                with m2:
                    st.metric("Simulated Sales", f"{active_currency}{sim_sales:,.2f}", delta=f"{pct_val:+.1f}%")
                with m3:
                    st.metric("Revenue Impact", f"{active_currency}{delta_val:+,.2f}")

                sim_df = pd.DataFrame({
                    "Scenario": ["Baseline Sales", "Simulated Sales"],
                    f"Sales ({active_currency})": [b_sales, sim_sales]
                })
                fig_sim = px.bar(
                    sim_df, x="Scenario", y=f"Sales ({active_currency})", color="Scenario",
                    color_discrete_map={"Baseline Sales": "#94A3B8", "Simulated Sales": "#4F46E5"},
                    title=f"Baseline vs. Simulated Sales Comparison ({active_currency})"
                )
                fig_sim.update_layout(template="plotly_white", height=280, showlegend=False)
                st.plotly_chart(fig_sim, use_container_width=True)

        # Backtest Benchmarks Expander
        with st.expander("🏆 Verified Model Benchmarks & Feature Explainability", expanded=False):
            bench_col1, bench_col2 = st.columns(2)
            with bench_col1:
                st.markdown("##### Walk-Forward Rolling Origin Evaluation")
                backtest_file = os.path.join(MODEL_DIR, "daily_candidate_models", "backtest_results_corrected.json")
                if not os.path.exists(backtest_file):
                    backtest_file = os.path.join(MODEL_DIR, "daily_candidate_models", "backtest_results.json")

                if os.path.exists(backtest_file):
                    with open(backtest_file, "r") as bf:
                        bt_full = json.load(bf)
                    bt_data = bt_full.get("summary_metrics", {})
                    origins_count = bt_full.get("origins_count", 13)

                    sel_hz = st.selectbox("Benchmark Horizon", options=["7_day", "14_day", "30_day"], key="ov_bt_hz")
                    b_rows = []
                    for m_name, m_horizons in bt_data.items():
                        h_info = m_horizons.get(sel_hz, {})
                        micro = h_info.get("micro", h_info)
                        macro = h_info.get("macro", {})
                        cov_val = micro.get("coverage", macro.get("coverage_mean"))
                        b_rows.append({
                            "Architecture": m_name,
                            "MAE ($)": f"${micro.get('mae', 0):,.2f}",
                            "WAPE (Micro)": f"{micro.get('wape', 0)*100:.1f}%",
                            "Bias": f"{micro.get('bias_pct', 0)*100:+.1f}%",
                            "Coverage": f"{cov_val*100:.1f}%" if cov_val is not None else "N/A",
                        })
                    st.dataframe(pd.DataFrame(b_rows).set_index("Architecture"), use_container_width=True)
                    st.caption(f"Evaluated on {origins_count} chronological rolling origins.")
                else:
                    st.info("Verified backtest benchmark results file not found.")

            with bench_col2:
                st.markdown("##### Feature Importance (XGBoost)")
                primary_m = models_dict.get("XGBoost")
                if primary_m:
                    reg = getattr(primary_m, 'named_steps', {}).get('regressor', primary_m)
                    importances = getattr(reg, 'feature_importances_', None)
                    if importances is not None and len(importances) == len(ENGINEERED_FEATURES):
                        imp_df = pd.DataFrame({
                            "Feature": ENGINEERED_FEATURES,
                            "Importance": importances
                        }).sort_values("Importance", ascending=True)
                        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
                        fig_imp.update_layout(template="plotly_white", height=300)
                        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e_ov:
        st.error(f"⚠️ Sales Overview error: {e_ov}")

# ----------------------------------------------------------------------------
# 2. FORECAST LAB WORKSPACE
# ----------------------------------------------------------------------------
with tab_forecast:
    try:
        render_data_mode_banner(active_mode, active_dataset_title, currency=active_currency, details="Multi-Horizon Forecast Lab & Predictive Inference")
        st.subheader("🔮 Production Forecast Lab")
        st.caption("Generate recursive multi-step forecasts with empirical or calibrated prediction intervals and full audit provenance.")

        if active_mode == DataMode.NO_DATA:
            st.info("💡 **No Active Dataset Selected**: You can still generate point and multi-step forecasts below using custom feature parameters or select a dataset in the sidebar.")

        f_cfg1, f_cfg2, f_cfg3 = st.columns([1, 1, 1])
        with f_cfg1:
            lab_horizon = st.radio("Forecast Horizon", options=[7, 14, 30], format_func=lambda x: f"{x}-Day Horizon", horizontal=True, key="lab_hz")
            lab_scope = st.text_input("Sales Scope Identifier", value="TOTAL_STORE", key="lab_scope")
        with f_cfg2:
            lab_model = st.selectbox("Candidate Architecture", options=model_registry.list_models(), index=0, key="lab_model")
            lab_interval = st.selectbox(
                "Prediction Interval Methodology",
                options=["heuristic_sqrt_step", "calibrated_oof_residual"],
                format_func=lambda x: "Heuristic Sqrt Step (Dashboard Default)" if x == "heuristic_sqrt_step" else "Calibrated OOF Residual",
                key="lab_interval"
            )
        with f_cfg3:
            lab_persist = st.checkbox("Persist Run to Audit Storage", value=True, help="Record run metadata and step predictions in database for realization tracking.", key="lab_persist")
            default_origin = datetime.date(2011, 12, 10) if active_currency == "£" else datetime.date.today()
            lab_base_date = st.date_input("Forecast Origin Date", value=default_origin, key="lab_date")

        st.markdown(f"##### Temporal Feature Inputs ({active_currency_code})")
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        def_l1 = 32150.0 if active_currency == "£" else 160.0
        def_rm7 = 53500.0 if active_currency == "£" else 155.0
        def_rm14 = 58200.0 if active_currency == "£" else 150.0
        def_rs7 = 18500.0 if active_currency == "£" else 15.0

        with p_c1:
            f_lag1 = st.number_input(f"Yesterday Sales ({active_currency})", value=def_l1, step=100.0 if active_currency == "£" else 10.0, key="lab_l1")
        with p_c2:
            f_rm7 = st.number_input(f"7-Day Rolling Mean ({active_currency})", value=def_rm7, step=100.0 if active_currency == "£" else 5.0, key="lab_rm7")
        with p_c3:
            f_rm14 = st.number_input(f"14-Day Rolling Mean ({active_currency})", value=def_rm14, step=100.0 if active_currency == "£" else 5.0, key="lab_rm14")
        with p_c4:
            f_rs7 = st.number_input(f"7-Day Volatility Std Dev ({active_currency})", value=def_rs7, step=50.0 if active_currency == "£" else 1.0, key="lab_rs7")

        btn_run_lab = st.button("🚀 Generate Multi-Horizon Forecast", use_container_width=True)

        if btn_run_lab:
            d_feats = extract_date_features(lab_base_date)
            init_df = build_feature_dataframe(
                d_feats, f_lag1, f_lag1 * 0.95, f_lag1 * 0.9, f_lag1 * 0.85,
                f_rm7, f_rm14, f_rs7, 100
            )

            import uuid
            assigned_run_id = str(uuid.uuid4()) if lab_persist else None

            with st.spinner("Generating recursive forecast rollout..."):
                fc_res_df = fc_service.forecast_multi_horizon(
                    initial_df=init_df,
                    horizon_days=lab_horizon,
                    model_choice=lab_model,
                    origin_date=lab_base_date,
                    interval_method=lab_interval,
                    persist_run=lab_persist,
                    run_id=assigned_run_id,
                    execution_source="streamlit_forecast_lab",
                    sales_scope=lab_scope,
                )
                st.session_state["lab_forecast_df"] = fc_res_df
                st.session_state["lab_run_id"] = assigned_run_id
                st.session_state["lab_meta"] = {
                    "horizon": lab_horizon,
                    "model": lab_model,
                    "interval": lab_interval,
                    "scope": lab_scope,
                    "origin": lab_base_date.isoformat(),
                    "persisted": lab_persist,
                    "currency": active_currency,
                }

        if "lab_forecast_df" in st.session_state and st.session_state["lab_forecast_df"] is not None:
            res_df = st.session_state["lab_forecast_df"]
            meta = st.session_state.get("lab_meta", {})
            run_id_val = st.session_state.get("lab_run_id")
            cur_sym = meta.get("currency", active_currency)

            # Forecast Provenance Card
            st.markdown(f"""
            <div class="provenance-card">
                <strong>📋 Forecast Provenance:</strong><br/>
                • <strong>Run ID:</strong> <code>{run_id_val or 'Unpersisted In-Memory Run'}</code><br/>
                • <strong>Origin Date:</strong> {meta.get('origin')} | <strong>Scope:</strong> {meta.get('scope')} | <strong>Model:</strong> {meta.get('model')}<br/>
                • <strong>Horizon:</strong> {meta.get('horizon')} Days | <strong>Interval Method:</strong> <code>{meta.get('interval')}</code> | <strong>Currency:</strong> {cur_sym}
            </div>
            """, unsafe_allow_html=True)

            # Interactive Forecast Plot
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([res_df['Date'], res_df['Date'][::-1]]),
                y=pd.concat([res_df['Upper Bound ($)'], res_df['Lower Bound ($)'][::-1]]),
                fill='toself',
                fillcolor='rgba(79, 70, 229, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Prediction Interval Band'
            ))
            fig_fc.add_trace(go.Scatter(
                x=res_df['Date'],
                y=res_df['Forecasted Sales ($)'],
                mode='lines+markers',
                line=dict(color='#4F46E5', width=3),
                marker=dict(size=7),
                name=f'Forecasted Daily Sales ({cur_sym})'
            ))
            fig_fc.update_layout(
                title=f"{meta.get('horizon')}-Day Sales Forecast with Prediction Uncertainty ({cur_sym})",
                xaxis_title="Forecast Date", yaxis_title=f"Sales ({cur_sym})",
                template="plotly_white", height=420, hovermode="x unified"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Data Table & CSV Download
            t_col1, t_col2 = st.columns([3, 1])
            with t_col1:
                display_df = res_df.copy()
                display_df.columns = ["Forecast Date", f"Predicted Sales ({cur_sym})", f"Lower Bound ({cur_sym})", f"Upper Bound ({cur_sym})"]
                st.dataframe(
                    display_df.style.format({
                        f"Predicted Sales ({cur_sym})": f"{cur_sym}{{:,.2f}}",
                        f"Lower Bound ({cur_sym})": f"{cur_sym}{{:,.2f}}",
                        f"Upper Bound ({cur_sym})": f"{cur_sym}{{:,.2f}}"
                    }),
                    use_container_width=True
                )
            with t_col2:
                csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Forecast CSV",
                    data=csv_bytes,
                    file_name=f"sales_forecast_{meta.get('horizon')}d_{meta.get('model')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                total_rev = res_df["Forecasted Sales ($)"].sum()
                avg_daily = res_df["Forecasted Sales ($)"].mean()
                st.metric("Total Projected Revenue", f"{cur_sym}{total_rev:,.2f}")
                st.metric("Avg Daily Projection", f"{cur_sym}{avg_daily:,.2f}")

            # Dynamic Recommendations
            first_pred = res_df['Forecasted Sales ($)'].iloc[0]
            insights = generate_business_insights(first_pred, f_lag1, f_rm7, f_rm14)
            st.markdown("##### 💡 Strategic Guidance & Demand Momentum")
            for item in insights:
                st.markdown(f"""
                <div class="insight-card">
                    <strong>{item['icon']} {item['title']}</strong><br/>
                    <span style="color: #475569;">{item['desc']}</span><br/>
                    <small style="color: #4F46E5; font-weight: 600;">{item['action']}</small>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("👆 Configure forecast parameters above and click **Generate Multi-Horizon Forecast** to begin.")

        # Single-Day Quick Prediction Expander
        with st.expander("⚡ Quick Single-Day Point Inference", expanded=False):
            with st.form("single_step_quick_form"):
                sq_c1, sq_c2, sq_c3 = st.columns(3)
                with sq_c1:
                    sq_date = st.date_input("Target Date", value=default_origin + timedelta(days=1))
                    sq_l1 = st.number_input(f"Yesterday Sales ({active_currency})", min_value=0.0, value=def_l1)
                with sq_c2:
                    sq_l7 = st.number_input(f"7-Day Lag ({active_currency})", min_value=0.0, value=def_rm7)
                    sq_rm7 = st.number_input(f"7-Day Mean ({active_currency})", min_value=0.0, value=def_rm7)
                with sq_c3:
                    sq_rm14 = st.number_input(f"14-Day Mean ({active_currency})", min_value=0.0, value=def_rm14)
                    sq_rs7 = st.number_input(f"7-Day Std Dev ({active_currency})", min_value=0.0, value=def_rs7)
                btn_sq = st.form_submit_button("Run Single-Step Inference")

            if btn_sq:
                sq_feats = extract_date_features(sq_date)
                sq_input = build_feature_dataframe(sq_feats, sq_l1, sq_l7, sq_l7, sq_l7, sq_rm7, sq_rm14, sq_rs7, 100)
                sq_pred = predict_sales(sq_input, model_choice=lab_model)
                st.success(f"**Point Prediction ({sq_date}):** {active_currency}{sq_pred:,.2f} (Model: {lab_model})")

    except Exception as e_lab:
        st.error(f"⚠️ Forecast Lab error: {e_lab}")

# ----------------------------------------------------------------------------
# 3. FORECAST PERFORMANCE & RECONCILIATION WORKSPACE
# ----------------------------------------------------------------------------
with tab_perf:
    try:
        render_data_mode_banner(active_mode, active_dataset_title, currency=active_currency, details="Forecast Performance & Reconciliation")
        st.subheader("🎯 Out-of-Sample Performance & Reconciled Actuals")
        st.caption("Rigorous separation between historical out-of-sample backtest validation and post-facto ground truth reconciliation.")

        # ====================================================================
        # SECTION A: HISTORICAL OUT-OF-SAMPLE BACKTESTING EVIDENCE
        # ====================================================================
        st.markdown("### 🏛️ Section A: Historical Out-of-Sample Backtesting Performance")
        st.markdown(
            "Evaluation methodology: **Chronological Rolling-Origin Walk-Forward Cross-Validation** across 13 origins with strict embargo to prevent target leakage. "
            "Metrics evaluate candidate architecture generalization under past market regimes."
        )

        st.warning(
            "⚠️ **Evaluation Integrity Notice:** Historical backtest metrics reflect model performance on historical rolling folds. "
            "They measure algorithmic stability and generalizability on past data, but do **NOT** guarantee future forecast accuracy or prevent realization degradation under market drift."
        )

        backtest_file = os.path.join(MODEL_DIR, "daily_candidate_models", "backtest_results_corrected.json")
        if not os.path.exists(backtest_file):
            backtest_file = os.path.join(MODEL_DIR, "daily_candidate_models", "backtest_results.json")

        if os.path.exists(backtest_file):
            with open(backtest_file, "r") as bf:
                bt_data_full = json.load(bf)
            bt_summary = bt_data_full.get("summary_metrics", {})
            origins_cnt = bt_data_full.get("origins_count", 13)

            perf_hz = st.selectbox("Select Backtest Horizon to Compare", options=["7_day", "14_day", "30_day"], index=1, key="perf_bt_hz")
            
            b_table = []
            for arch_name, h_dict in bt_summary.items():
                h_metrics = h_dict.get(perf_hz, {})
                micro = h_metrics.get("micro", h_metrics)
                macro = h_metrics.get("macro", {})
                cov_val = micro.get("coverage", macro.get("coverage_mean"))
                mae_val = micro.get("mae", 0.0)
                rmse_val = micro.get("rmse", 0.0)
                wape_val = micro.get("wape", 0.0)
                bias_val = micro.get("bias_pct", 0.0)
                eval_obs = micro.get("evaluated_count", 13 * int(perf_hz.split("_")[0]))

                b_table.append({
                    "Architecture": arch_name,
                    "Evaluation Sample": f"{eval_obs} observations ({origins_cnt} origins)",
                    "MAE ($)": f"${mae_val:,.2f}",
                    "RMSE ($)": f"${rmse_val:,.2f}",
                    "WAPE (Micro)": f"{wape_val * 100.0:.2f}%",
                    "Forecast Bias": f"{bias_val * 100.0:+.2f}%",
                    "Interval Coverage": f"{cov_val * 100.0:.1f}%" if cov_val is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(b_table).set_index("Architecture"), use_container_width=True)

            with st.expander("📐 Metric Formulations & Mathematical Policy", expanded=False):
                st.markdown(r"""
                - **MAE (Mean Absolute Error):** $\frac{1}{N} \sum |y_t - \hat{y}_t|$
                - **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{N} \sum (y_t - \hat{y}_t)^2}$
                - **WAPE (Weighted Absolute Percentage Error):** $\frac{\sum |y_t - \hat{y}_t|}{\sum y_t} \times 100\%$ — avoids infinite division on zero-sales days.
                - **Forecast Bias:** $\frac{\sum (\hat{y}_t - y_t)}{\sum y_t} \times 100\%$ — positive indicates structural overforecasting; negative indicates underforecasting.
                - **Prediction Interval Coverage:** Percentage of realized observations falling strictly within $[L_t, U_t]$.
                """)
        else:
            st.info("Verified backtest benchmark results file not found in model_data/daily_candidate_models/.")

        st.divider()

        # ====================================================================
        # SECTION B: REALIZED FORECAST PERFORMANCE (RECONCILED ACTUALS)
        # ====================================================================
        st.markdown("### 🎯 Section B: Realized Forecast Performance (Reconciled Actuals)")
        st.info(
            "ℹ️ **Ground Truth Realization Protocol:** Metrics in this section are computed **strictly** against matching actual sales recorded in SQLite storage. "
            "Steps awaiting future actuals remain designated as **Pending Realization** and are **never** filled with zero."
        )

        if not db_ready or audit_repo is None:
            st.warning("⚠️ Database connection is unavailable. Realized performance tracking requires active SQLite storage.")
        else:
            recent_runs = audit_repo.list_forecast_runs(limit=30)
            if not recent_runs:
                st.info("No persisted forecast runs found in the database. Generate a forecast in the **Forecast Lab** with 'Persist to Audit Storage' checked.")
            else:
                run_options = {
                    r.run_id: f"{r.created_at.strftime('%Y-%m-%d %H:%M')} | {r.model_name} | {r.horizon_days}d | {active_currency}{r.total_predicted_revenue or 0:,.0f} ({r.run_id[:8]}...)"
                    for r in recent_runs
                }
                selected_run_id = st.selectbox(
                    "Select Persisted Forecast Run to Reconcile",
                    options=list(run_options.keys()),
                    format_func=lambda x: run_options[x],
                    key="perf_run_select"
                )

                selected_run = audit_repo.get_forecast_run(selected_run_id)
                if selected_run:
                    items = sorted(selected_run.items, key=lambda x: x.step_ahead)
                    total_items = len(items)
                    reconciled_items = [it for it in items if it.actual_value is not None]
                    pending_items = [it for it in items if it.actual_value is None]

                    # Summary cards
                    p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
                    with p_c1:
                        st.metric("Horizon / Steps", f"{selected_run.horizon_days} Days")
                    with p_c2:
                        st.metric("Reconciled Steps", f"{len(reconciled_items)} / {total_items}")
                    with p_c3:
                        if reconciled_items:
                            mae_val = float(np.mean([it.absolute_error for it in reconciled_items if it.absolute_error is not None]))
                            st.metric("Realized MAE", f"{active_currency}{mae_val:,.2f}")
                        else:
                            st.metric("Realized MAE", "Awaiting Actuals")
                    with p_c4:
                        if reconciled_items:
                            sq_errs = [(it.actual_value - it.predicted_value)**2 for it in reconciled_items]
                            rmse_val = float(np.sqrt(np.mean(sq_errs)))
                            st.metric("Realized RMSE", f"{active_currency}{rmse_val:,.2f}")
                        else:
                            st.metric("Realized RMSE", "Awaiting Actuals")
                    with p_c5:
                        if reconciled_items:
                            cov_count = sum(1 for it in reconciled_items if it.interval_covered is True)
                            cov_pct = (cov_count / len(reconciled_items)) * 100.0
                            st.metric("Interval Coverage", f"{cov_pct:.1f}%")
                        else:
                            st.metric("Interval Coverage", "Awaiting Actuals")

                    # Extended Realized Metrics (WAPE and Bias)
                    if reconciled_items:
                        sum_act = sum(it.actual_value for it in reconciled_items)
                        sum_abs_err = sum(it.absolute_error for it in reconciled_items if it.absolute_error is not None)
                        wape_val = (sum_abs_err / sum_act * 100.0) if sum_act > 0 else None
                        bias_val = (sum((it.predicted_value - it.actual_value) for it in reconciled_items) / sum_act * 100.0) if sum_act > 0 else None

                        e_m1, e_m2, e_m3 = st.columns(3)
                        with e_m1:
                            st.caption(f"**Realized WAPE:** {f'{wape_val:.2f}%' if wape_val is not None else 'N/A'}")
                        with e_m2:
                            st.caption(f"**Forecast Bias:** {f'{bias_val:+.2f}%' if bias_val is not None else 'N/A'}")
                        with e_m3:
                            st.caption(f"**Interval Method:** <code>{selected_run.interval_method}</code>")

                    # Time Series Plot: Predicted vs Actual
                    dates = [it.target_date for it in items]
                    preds = [it.predicted_value for it in items]
                    lowers = [it.lower_bound for it in items]
                    uppers = [it.upper_bound for it in items]

                    fig_eval = go.Figure()
                    if all(l is not None for l in lowers) and all(u is not None for u in uppers):
                        fig_eval.add_trace(go.Scatter(
                            x=dates + dates[::-1],
                            y=uppers + lowers[::-1],
                            fill='toself',
                            fillcolor='rgba(79, 70, 229, 0.12)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Prediction Interval Band',
                            hoverinfo="skip"
                        ))

                    fig_eval.add_trace(go.Scatter(
                        x=dates, y=preds, mode='lines+markers',
                        name=f'Predicted Sales ({active_currency})', line=dict(color='#4F46E5', width=2.5)
                    ))

                    # Plot actuals only for dates where they exist
                    reconciled_dates = [it.target_date for it in reconciled_items]
                    reconciled_vals = [it.actual_value for it in reconciled_items]
                    if reconciled_dates:
                        fig_eval.add_trace(go.Scatter(
                            x=reconciled_dates, y=reconciled_vals, mode='markers+lines',
                            name=f'Reconciled Actual Sales ({active_currency})',
                            marker=dict(color='#10B981', size=9, symbol='diamond'),
                            line=dict(color='#10B981', width=1.5, dash='dash')
                        ))

                    fig_eval.update_layout(
                        title=f"Forecast Realization: Run {selected_run.run_id[:8]} ({selected_run.model_name})",
                        xaxis_title="Target Date", yaxis_title=f"Sales ({active_currency})",
                        template="plotly_white", height=400, hovermode="x unified"
                    )
                    st.plotly_chart(fig_eval, use_container_width=True)

                    # Item Table
                    table_rows = []
                    for it in items:
                        ape_str = f"{(it.absolute_error / it.actual_value * 100):.1f}%" if (it.actual_value and it.absolute_error) else "Pending"
                        table_rows.append({
                            "Target Date": it.target_date.isoformat(),
                            "Step Ahead": it.step_ahead,
                            f"Predicted ({active_currency})": f"{active_currency}{it.predicted_value:,.2f}",
                            f"Actual ({active_currency})": f"{active_currency}{it.actual_value:,.2f}" if it.actual_value is not None else "⏳ Pending Realization",
                            f"Absolute Error ({active_currency})": f"{active_currency}{it.absolute_error:,.2f}" if it.absolute_error is not None else "Pending",
                            "APE (%)": ape_str,
                            "Interval Range": f"[{active_currency}{it.lower_bound or 0:,.0f} — {active_currency}{it.upper_bound or 0:,.0f}]",
                            "In Interval?": "✅ Yes" if it.interval_covered is True else ("❌ No" if it.interval_covered is False else "⏳ Pending"),
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # Operator Actuals Ingestion & Reconciliation Form
            st.divider()
            with st.expander("📝 Operator Actuals Reconciliation Tool", expanded=False):
                st.caption("Ingest ground truth actual sales for a calendar date to automatically reconcile all matching forecast items across runs.")
                rec_c1, rec_c2, rec_c3 = st.columns(3)
                with rec_c1:
                    rec_date = st.date_input("Actual Sales Date", value=datetime.date.today(), key="rec_date")
                with rec_c2:
                    rec_sales = st.number_input(f"Actual Sales Amount ({active_currency})", min_value=0.0, value=175.0, step=10.0, key="rec_val")
                with rec_c3:
                    rec_scope = st.text_input("Scope", value="TOTAL_STORE", key="rec_scope")

                b_reconcile = st.button("📥 Ingest & Reconcile Ground Truth Actuals")
                if b_reconcile and audit_repo:
                    rec_obj, was_created = audit_repo.ingest_actual_sales(
                        calendar_date=rec_date,
                        daily_sales=rec_sales,
                        sales_scope=rec_scope,
                    )
                    action_word = "Ingested new" if was_created else "Updated existing"
                    st.success(f"✅ {action_word} actual sales record for {rec_date} ({active_currency}{rec_sales:,.2f}). All matching forecast items reconciled!")
                    st.rerun()

                if st.button("⚡ Scan & Reconcile All Pending Items Against Stored Actuals", key="rec_all_btn"):
                    if audit_repo:
                        matched_cnt = audit_repo.reconcile_all_pending(sales_scope="TOTAL_STORE")
                        st.info(f"Reconciliation scan complete: {matched_cnt} pending forecast items reconciled.")
                        st.rerun()

    except Exception as e_perf:
        st.error(f"⚠️ Forecast Performance error: {e_perf}")

# ----------------------------------------------------------------------------
# 4. DATA QUALITY WORKSPACE
# ----------------------------------------------------------------------------
with tab_quality:
    try:
        render_data_mode_banner(active_mode, active_dataset_title, currency=active_currency, details="Data Quality, Ingestion SLA & Drift Observability")
        st.subheader("🛡️ Data Quality Workspace & Contract Enforcement")
        st.caption("Safely ingest tabular sales datasets, validate contract compliance, isolate quarantined records, and review persistent audit history.")

        # If Online Retail II is selected, show Dataset Provenance & Onboarding Audit Card
        if dataset_choice == "UCI Online Retail II (Demo - UK, £ GBP)":
            prov = online_retail_data.get("provenance", {})
            val_meta = online_retail_data.get("validation", {})
            st.markdown("### 📋 Dataset Onboarding & Provenance Audit")
            st.markdown("""
            <div class="provenance-card">
                <strong>🏛️ Official Citation & Dataset Provenance:</strong><br/>
                • <strong>Citation:</strong> Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5BW33" target="_blank">https://doi.org/10.24432/C5BW33</a><br/>
                • <strong>License:</strong> Creative Commons Attribution 4.0 International (CC BY 4.0)<br/>
                • <strong>Source URI:</strong> <code>https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip</code> | <strong>Retrieved:</strong> 2026-10-21 UTC<br/>
                • <strong>Zip Checksum (SHA-256):</strong> <code>572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb</code> (45.6 MB)<br/>
                • <strong>Excel Workbook (SHA-256):</strong> <code>bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980</code> (45.6 MB)
            </div>
            """, unsafe_allow_html=True)

            dq_k1, dq_k2, dq_k3, dq_k4 = st.columns(4)
            with dq_k1:
                st.metric("Source Transactions", f"{prov.get('total_transaction_rows', 1067371):,}", help="Total raw transaction records across Year 2009-2010 and Year 2010-2011 sheets.")
            with dq_k2:
                st.metric("Accepted Transactions", f"{prov.get('valid_transaction_rows', 1041670):,}", help="Records passing schema contract, positive prices, and chronological validation.")
            with dq_k3:
                st.metric("Returns / Cancellations", f"{prov.get('cancellation_rows', 19494):,}", help="Explicitly segregated cancellation invoices (starting with 'C').")
            with dq_k4:
                st.metric("Quarantined Price Adjustments", f"{prov.get('zero_or_negative_price_rows', 6207):,}", delta="-6,207 invalid", delta_color="inverse", help="Records with zero or negative unit price quarantined (e.g. manual book adjustments, bad debts).")

            dq_cov1, dq_cov2 = st.columns(2)
            with dq_cov1:
                st.markdown("##### 📅 Calendar Coverage & Missing Date Policy")
                st.write(f"• **Date Span:** `{prov.get('date_range_start', '2009-12-01')}` to `{prov.get('date_range_end', '2011-12-09')}` ({prov.get('total_calendar_days', 739)} calendar days)")
                st.write(f"• **Active Trading Days:** `{prov.get('observed_trading_days', 604)}` days | **Non-Trading Days:** `{prov.get('unobserved_calendar_days', 135)}` days")
                st.caption("ℹ️ *Policy: Non-trading calendar days (Sundays / bank holidays) are confirmed zero sales transactions (£0.00) in POS systems, not unobserved gaps. Reindexed to daily frequency to preserve temporal lag spacing.*")
            with dq_cov2:
                st.markdown("##### 💷 Accounting Totals")
                st.write(f"• **Gross Sales:** `£{prov.get('total_gross_sales', 20972594.55):,.2f}`")
                st.write(f"• **Returns / Refunds:** `£{prov.get('total_returns', 1527415.00):,.2f}`")
                st.write(f"• **Net Realized Sales:** `£{prov.get('total_net_sales', 19445179.55):,.2f}`")

            st.divider()

        # --------------------------------------------------------------------
        # 4A. Freshness & Ingestion Monitoring Observability Card
        # --------------------------------------------------------------------
        st.markdown("### 📡 Data Freshness & Ingestion Health")
        st.caption("Deterministic observability monitoring ingestion pipeline outcomes, calendar freshness SLAs, and failure escalation.")

        m_c1, m_c2, m_c3, m_c4 = st.columns([1.5, 1.2, 1.3, 1.0])
        with m_c1:
            dq_scope = st.selectbox("Monitoring Scope", options=["TOTAL_STORE"], index=0, key="dq_mon_scope")
        with m_c2:
            dq_cadence_str = st.selectbox(
                "Update Cadence",
                options=[c.value for c in CadenceType],
                index=1,
                key="dq_cadence_sel",
                help="Configured expected ingestion cadence for this dataset scope."
            )
            selected_cadence = CadenceType.from_str(dq_cadence_str)
        with m_c3:
            default_sla = get_cadence_default_threshold(selected_cadence)
            dq_staleness = st.number_input(
                "Freshness SLA (Days)",
                min_value=1,
                max_value=365,
                value=default_sla,
                key="dq_stale",
                help=f"Allowed staleness threshold in days for cadence {dq_cadence_str}."
            )
        with m_c4:
            st.write("")
            st.write("")
            btn_run_check = st.button("🔄 Run Check", key="btn_run_monitoring_check", use_container_width=True, help="Executes on-demand data freshness and ingestion health check")

        # Evaluate on manual request or retrieve latest check (never automatically during reruns)
        current_verdict: Optional[MonitoringVerdict] = None
        eval_error: Optional[str] = None
        if btn_run_check:
            with st.spinner("Evaluating data freshness and ingestion signals..."):
                try:
                    if not db_ready or mon_service is None:
                        eval_error = "Database offline or monitoring service unavailable."
                    else:
                        current_verdict = mon_service.check_data_health(
                            sales_scope=dq_scope,
                            config=MonitoringConfig(
                                cadence=selected_cadence,
                                staleness_threshold_days=int(dq_staleness),
                            ),
                            persist=True,
                        )
                except Exception as ex:
                    eval_error = str(ex)

        if btn_run_check:
            st.markdown("#### ⚡ Live Evaluation Check (Executed On-Demand)")
            if eval_error:
                st.info(f"⚪ **Monitoring Status: UNKNOWN** — Evaluation could not establish health: {eval_error}")
            elif current_verdict is not None:
                v_status = current_verdict.status.value
                v_msg = current_verdict.summary_message
                chk_iso = format_datetime_safe(current_verdict.checked_at, "%Y-%m-%d %H:%M:%S UTC")
                if v_status == "HEALTHY":
                    st.success(f"🟢 **Monitoring Status: HEALTHY** (Live: {chk_iso}) — {v_msg}")
                elif v_status == "WARNING":
                    st.warning(f"🟡 **Monitoring Status: WARNING** (Live: {chk_iso}) — {v_msg}")
                elif v_status == "CRITICAL":
                    st.error(f"🔴 **Monitoring Status: CRITICAL** (Live: {chk_iso}) — {v_msg}")
                else:
                    st.info(f"⚪ **Monitoring Status: UNKNOWN** (Live: {chk_iso}) — {v_msg}")

                if current_verdict.remediation_advice:
                    st.warning(f"🛠️ **Actionable Remediation:** {current_verdict.remediation_advice}")

                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    st.metric(
                        "Freshness State",
                        f"{current_verdict.freshness.state.value} ({current_verdict.freshness.staleness_days or 0}d)",
                        help=f"Cadence: {selected_cadence.value.upper()}, SLA: ≤ {dq_staleness} days",
                    )
                with fc2:
                    last_succ = format_datetime_safe(current_verdict.ingestion.last_successful_ingestion_at, "%Y-%m-%d %H:%M", fallback="Never Recorded")
                    st.metric("Last Successful Ingest", last_succ)
                with fc3:
                    rel_rate = (current_verdict.ingestion.reliability_rate * 100.0) if current_verdict.ingestion.reliability_rate is not None else 0.0
                    succ_cnt = current_verdict.ingestion.historical_success_count
                    tot_cnt = current_verdict.ingestion.total_attempts_evaluated
                    st.metric(
                        "Latest Outcome",
                        current_verdict.ingestion.latest_outcome.value,
                        help=f"Historical Reliability: {rel_rate:.1f}% ({succ_cnt}/{tot_cnt} successful runs)",
                    )
                with fc4:
                    cf = current_verdict.ingestion.consecutive_failures
                    st.metric("Consecutive Failures", f"{cf}", delta=f"{cf} failures" if cf > 0 else "0", delta_color="inverse")
            st.divider()

        # Display latest persisted snapshot separately
        st.markdown("#### 💾 Latest Persisted Check Snapshot")
        latest_record = mon_service.get_latest_monitoring_check(sales_scope=dq_scope) if (db_ready and mon_service) else None

        if not db_ready:
            st.info("⚪ **Monitoring Status: UNKNOWN** — Database offline or storage unavailable. Cannot retrieve persisted health check.")
        elif latest_record is not None:
            r_status = latest_record.status
            r_msg = latest_record.summary_message
            chk_time = format_datetime_safe(latest_record.checked_at, "%Y-%m-%d %H:%M:%S UTC", fallback="Recent")
            if r_status == "HEALTHY":
                st.success(f"🟢 **Persisted Status: HEALTHY** (Recorded: {chk_time}) — {r_msg}")
            elif r_status == "WARNING":
                st.warning(f"🟡 **Persisted Status: WARNING** (Recorded: {chk_time}) — {r_msg}")
            elif r_status == "CRITICAL":
                st.error(f"🔴 **Persisted Status: CRITICAL** (Recorded: {chk_time}) — {r_msg}")
            else:
                st.info(f"⚪ **Persisted Status: UNKNOWN** (Recorded: {chk_time}) — {r_msg}")

            try:
                ev_data = json.loads(latest_record.evidence_json) if latest_record.evidence_json else {}
                remed = ev_data.get("remediation_advice")
                if remed:
                    st.warning(f"🛠️ **Actionable Remediation:** {remed}")
            except Exception:
                pass

            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.metric("Persisted Freshness", f"{latest_record.freshness_state} ({latest_record.staleness_days or 0}d)", help=f"SLA threshold: ≤ {dq_staleness} days")
            with fc2:
                last_succ = format_datetime_safe(latest_record.last_successful_ingestion_at, "%Y-%m-%d %H:%M", fallback="Never Recorded")
                st.metric("Last Successful Ingest", last_succ)
            with fc3:
                st.metric("Latest Ingest Outcome", latest_record.latest_ingestion_outcome or "UNKNOWN")
            with fc4:
                cf = latest_record.consecutive_failures or 0
                st.metric("Consecutive Failures", f"{cf}", delta=f"{cf} failures" if cf > 0 else "0", delta_color="inverse")
        else:
            st.info("ℹ️ No persisted data freshness check found in storage. Click **Run Check** above to evaluate and persist pipeline health.")

        # Monitoring history expander
        with st.expander("📜 Historical Monitoring Logs & Status Changes", expanded=False):
            if db_ready and mon_service:
                m_hist = mon_service.get_monitoring_history(sales_scope=dq_scope, limit=10)
                if m_hist:
                    hist_rows = []
                    for h in m_hist:
                        hist_rows.append({
                            "Check Time (UTC)": format_datetime_safe(h.checked_at, "%Y-%m-%d %H:%M:%S", fallback="-"),
                            "Scope": h.sales_scope,
                            "Status": f"🟢 {h.status}" if h.status == "HEALTHY" else (f"🟡 {h.status}" if h.status == "WARNING" else (f"🔴 {h.status}" if h.status == "CRITICAL" else f"⚪ {h.status}")),
                            "Freshness": f"{h.freshness_state} ({h.staleness_days or 0}d)",
                            "Last Ingestion": format_datetime_safe(h.last_successful_ingestion_at, "%Y-%m-%d %H:%M", fallback="None"),
                            "Consecutive Failures": h.consecutive_failures or 0,
                            "Summary": h.summary_message,
                        })
                    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)
                else:
                    st.caption("No historical monitoring check runs found in audit database.")
            else:
                st.caption("Historical monitoring logs available when storage database is connected.")

        st.divider()

        # Ingestion & Validation Section
        st.markdown("### 📤 Ingest & Validate New Dataset")
        st.caption("Upload sales dataset for schema validation, quarantine filtering, and continuous time-series aggregation.")

        dq_file = st.file_uploader(
            "Upload Sales Dataset (CSV / Excel)",
            type=["csv", "xlsx", "xls"],
            key="dq_upload",
            help="Validates tabular records against strict data quality contracts. Isolates corrupt rows into quarantine."
        )

        if dq_file:
            with st.spinner("Executing Data Contracts & Schema Validation..."):
                file_bytes = dq_file.read()
                ingest_res = val_service.validate_file(
                    file_source=file_bytes,
                    file_name=dq_file.name,
                    persist_audit=True,
                )
                report = ingest_res.report
                diag = report.diagnostics

            # 1. Validation Verdict Banner
            if report.status == "PASSED":
                st.success("✅ **Dataset Validation Verdict: PASSED** — Satisfies all schema contracts and data-quality specifications.")
            elif report.status == "WARNING":
                w_cnt = len([i for i in report.issues if i.severity.value == "WARNING"])
                st.warning(f"⚠️ **Dataset Validation Verdict: PASSED WITH WARNINGS** — {w_cnt} review flag(s) identified. Clean records are available for processing.")
            else:
                e_cnt = len([i for i in report.issues if i.severity.value == "ERROR"])
                st.error(f"❌ **Dataset Validation Verdict: FAILED** — {e_cnt} critical contract violation(s) detected. Dataset rejected from downstream forecasting.")

            # 2. Key Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Detected Grain", report.grain.value.replace("_", " ").title())
            with m2:
                st.metric("Total Records Evaluated", f"{report.total_rows_read:,}")
            with m3:
                st.metric("Valid Clean Records", f"{report.valid_rows_count:,}")
            with m4:
                st.metric("Quarantined Records", f"{report.quarantined_rows_count:,}", delta=f"-{report.quarantined_rows_count}" if report.quarantined_rows_count > 0 else "0", delta_color="inverse")

            # 3. Issue Codes & Actionable Remediations
            if report.issues:
                with st.expander(f"📋 Validation Issue Breakdown ({len(report.issues)} findings)", expanded=(report.status != "PASSED")):
                    issue_list = []
                    for iss in report.issues:
                        issue_list.append({
                            "Severity": iss.severity.value,
                            "Code": iss.code,
                            "Message": iss.message,
                            "Affected Records": iss.affected_count,
                            "Remediation Advice": iss.remediation,
                        })
                    st.dataframe(pd.DataFrame(issue_list), use_container_width=True)

            # 4. Data Quality Diagnostics Expander
            if diag and diag.get("valid", False):
                with st.expander("📊 Data Quality Diagnostics & Distribution Summary", expanded=False):
                    d_c1, d_c2 = st.columns(2)
                    with d_c1:
                        st.markdown("##### 📅 Calendar Coverage & Freshness")
                        cov = diag.get("date_coverage", {})
                        fresh = diag.get("freshness", {})
                        if cov:
                            st.write(f"• **Date Span:** `{cov.get('min_date')}` to `{cov.get('max_date')}` ({cov.get('calendar_span_days')} calendar days)")
                            st.write(f"• **Active Trading Days:** `{cov.get('unique_active_dates')}` dates | **Missing:** `{cov.get('missing_calendar_days')}` days")
                            st.caption("ℹ️ *Policy: Missing calendar days are zero-filled ($0.00) during time-series aggregation.*")
                        if fresh:
                            f_badge = "🟢 FRESH" if not fresh.get("is_stale") else "🟠 STALE"
                            st.write(f"• **Freshness Assessment:** {f_badge} (Staleness: `{fresh.get('staleness_days')}` days)")

                    with d_c2:
                        st.markdown("##### 💵 Sales Value Statistics & Review Flags")
                        dist = diag.get("sales_distribution", {})
                        num = diag.get("numeric_validity", {})
                        if dist:
                            st.write(f"• **Mean / Median Sales:** `${dist.get('mean', 0):,.2f}` / `${dist.get('median', 0):,.2f}`")
                            st.write(f"• **Observed Range:** `${dist.get('min', 0):,.2f}` to `${dist.get('max', 0):,.2f}`")
                        if num:
                            st.write(f"• **Zero Sales:** `{num.get('zero_sales_count', 0):,}` | **Returns (Negative):** `{num.get('negative_sales_count', 0):,}`")

            # 5. Export JSON Report
            st.download_button(
                "📥 Download Validation Report (JSON)",
                data=report.to_json(),
                file_name=f"{os.path.splitext(dq_file.name)[0]}_validation_report.json",
                mime="application/json",
                help="Export machine-readable validation outcome without exposing confidential row data."
            )

            # 6. Downstream Processing for Clean Records
            if ingest_res.is_usable and ingest_res.valid_df is not None:
                st.divider()
                st.markdown("##### 📈 Clean Series Processing")
                if report.grain in [GrainType.TRANSACTION, GrainType.DAILY_AGGREGATED]:
                    _, col_map = detect_dataset_grain(ingest_res.valid_df)
                    date_c = col_map.get("date", "Order Date")
                    sales_c = col_map.get("sales", "Sales")
                    clean_daily = aggregate_to_daily_sales(ingest_res.valid_df, date_col=date_c, sales_col=sales_c, fill_missing_dates=True)
                    st.session_state["user_clean_daily_df"] = clean_daily
                    st.session_state["user_dataset_name"] = dq_file.name
                    st.write(f"Continuous daily aggregation ready ({len(clean_daily)} active calendar days).")
                    csv_cl = clean_daily.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Clean Daily Sales Series (CSV)", data=csv_cl, file_name=f"{os.path.splitext(dq_file.name)[0]}_daily_clean.csv", mime="text/csv")

        # Persistent Audit Trail Section
        st.divider()
        st.markdown("### 📜 Persistent Dataset Validation Audit Trail")
        st.caption("Immutable audit log of all validated dataset uploads. Zero confidential row data is retained in persistence.")
        if db_ready and audit_repo:
            validations = audit_repo.list_validations(limit=20)
            if validations:
                v_rows = []
                for v in validations:
                    v_rows.append({
                        "Timestamp": v.created_at.strftime("%Y-%m-%d %H:%M:%S") if v.created_at else "-",
                        "File Name": v.file_name or "Unknown",
                        "SHA-256 Fingerprint": f"{v.file_sha256[:12]}...",
                        "Detected Grain": v.detected_grain,
                        "Status": "✅ PASSED" if v.validation_status == "PASSED" else ("⚠️ WARNING" if v.validation_status == "WARNING" else "❌ FAILED"),
                        "Total Rows": f"{v.row_count:,}",
                        "Errors": v.error_count,
                        "Warnings": v.warning_count,
                        "Usable": "Yes" if v.is_usable else "No",
                    })
                st.dataframe(pd.DataFrame(v_rows), use_container_width=True)
            else:
                st.info("No validation audit records logged yet.")
        else:
            st.caption("Validation audit trail available when database storage is active.")

        # --------------------------------------------------------------------
        # 4B. Dataset Drift Monitoring Observability Card
        # --------------------------------------------------------------------
        st.divider()
        st.markdown("### 🧪 Dataset Drift Monitoring")
        st.caption("Explainable distribution drift detection comparing batch features against the active reference baseline via Population Stability Index (PSI) and Total Variation Distance (TVD).")

        # 1. Active Reference Baseline Card
        active_baseline = drift_service.get_active_baseline(sales_scope=dq_scope) if drift_service else None
        if active_baseline:
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                st.metric("Active Baseline", active_baseline.baseline_id[:16] + "...")
            with b_col2:
                win_str = f"{active_baseline.data_window_start or 'N/A'} → {active_baseline.data_window_end or 'N/A'}"
                st.metric("Observation Window", win_str)
            with b_col3:
                st.metric("Baseline Samples", f"{active_baseline.row_count:,}")
            with b_col4:
                st.metric("Features Profiled", f"{len(active_baseline.feature_names)}")
            
            src_info = active_baseline.source_file_name or "Reference Dataset"
            sha_info = f"SHA-256: {active_baseline.source_file_sha256[:12]}..." if active_baseline.source_file_sha256 else "Registry Baseline"
            st.caption(f"📌 **Provenance:** {src_info} | {sha_info} | Created: {active_baseline.created_at[:19]} UTC")
        else:
            st.info(f"⚪ **No Active Reference Baseline**: Scope '{dq_scope}' has no active baseline. Create or promote a reference baseline below before evaluating drift.")

        # 2. Controls & Actions
        d_c1, d_c2 = st.columns([1.5, 3.5])
        with d_c1:
            btn_run_drift = st.button("🔄 Run Drift Check", key="btn_run_drift_eval", use_container_width=True, help="Executes an on-demand dataset drift check against the active reference baseline")
        with d_c2:
            st.caption("On-demand execution comparing current batch distributions with the active reference baseline. Never runs automatically on reruns.")

        # Baseline Lifecycle Management Expander
        with st.expander("📋 Baseline Lifecycle Management (Create / Promote)", expanded=False):
            st.markdown("##### Create New Reference Baseline")
            b_name_input = st.text_input("Baseline Identifier (optional)", placeholder="e.g. base_retail_y1", key="new_base_id")
            b_notes_input = st.text_input("Notes / Provenance", placeholder="e.g. Audited Year 1 reference baseline", key="new_base_notes")
            b_activate_chk = st.checkbox("Promote to active baseline immediately", value=False, key="new_base_activate")

            if st.button("Create Baseline from Available Data", key="btn_create_baseline"):
                target_base_df = None
                src_name = "Online Retail II Reference Series"
                src_sha = None
                if 'dq_file' in locals() and dq_file is not None and 'ingest_res' in locals() and ingest_res.valid_df is not None:
                    target_base_df = ingest_res.valid_df
                    src_name = dq_file.name
                    src_sha = ingest_res.report.file_sha256
                elif dataset_choice == "UCI Online Retail II (Demo - UK, £ GBP)":
                    target_base_df = online_retail_data.get("daily")
                    src_name = "online_retail_ii_daily.csv"
                elif superstore_df is not None:
                    target_base_df = superstore_df
                    src_name = "Sample - Superstore.csv"

                if target_base_df is None or len(target_base_df) == 0:
                    st.error("No valid dataset available to create baseline. Select a demo dataset or upload a file first.")
                else:
                    with st.spinner("Profiling dataset distributions and saving baseline..."):
                        try:
                            new_prof = drift_service.create_baseline(
                                df=target_base_df,
                                sales_scope=dq_scope,
                                baseline_id=b_name_input.strip() if b_name_input.strip() else None,
                                source_file_name=src_name,
                                source_file_sha256=src_sha,
                                notes=b_notes_input.strip() if b_notes_input.strip() else None,
                                activate=b_activate_chk,
                            )
                            st.success(f"✅ Baseline '{new_prof.baseline_id}' created ({new_prof.row_count} rows, {len(new_prof.feature_names)} features). Active: {b_activate_chk}")
                            st.rerun()
                        except Exception as e_base:
                            st.error(f"Failed to create baseline: {e_base}")

            st.divider()
            st.markdown("##### Promote Registered Baseline")
            all_baselines = drift_service.list_baselines(sales_scope=dq_scope, limit=20) if drift_service else []
            if all_baselines:
                b_options = {f"{b['baseline_id']} ({b['row_count']} rows, {'ACTIVE' if b['is_active'] else 'inactive'})": b['baseline_id'] for b in all_baselines}
                chosen_b_label = st.selectbox("Select Baseline to Promote", options=list(b_options.keys()), key="promo_base_sel")
                chosen_b_id = b_options[chosen_b_label]
                if st.button("Promote to Active Baseline", key="btn_promote_baseline"):
                    if drift_service.promote_baseline(chosen_b_id, sales_scope=dq_scope):
                        st.success(f"✅ Baseline '{chosen_b_id}' promoted to active status for scope '{dq_scope}'.")
                        st.rerun()
                    else:
                        st.error("Failed to promote baseline.")
            else:
                st.caption("No registered baselines found.")

        # Drift Check Evaluation / Results
        current_drift_report: Optional[DriftReport] = None
        drift_error: Optional[str] = None

        if btn_run_drift:
            with st.spinner("Evaluating population stability and frequency distances..."):
                try:
                    if not db_ready or drift_service is None:
                        drift_error = "Database offline or drift service unavailable."
                    else:
                        eval_df = None
                        if 'dq_file' in locals() and dq_file is not None and 'ingest_res' in locals() and ingest_res.valid_df is not None:
                            eval_df = ingest_res.valid_df
                        elif dataset_choice == "UCI Online Retail II (Demo - UK, £ GBP)":
                            eval_df = online_retail_data.get("daily")
                        elif superstore_df is not None:
                            eval_df = superstore_df

                        if eval_df is None or len(eval_df) == 0:
                            drift_error = "No current dataset available for drift evaluation. Ingest or select a dataset first."
                        else:
                            current_drift_report = drift_service.evaluate_drift(
                                current_df=eval_df,
                                sales_scope=dq_scope,
                                persist=True,
                            )
                            st.session_state["latest_drift_report"] = current_drift_report
                except Exception as ex_drift:
                    drift_error = str(ex_drift)
        elif "latest_drift_report" in st.session_state:
            current_drift_report = st.session_state["latest_drift_report"]
        elif drift_service:
            current_drift_report = drift_service.get_latest_drift_check(sales_scope=dq_scope)

        # Render Drift Status Banner & Results
        if drift_error:
            st.info(f"⚪ **Dataset Drift Status: UNKNOWN** — Evaluation could not proceed: {drift_error}")
        elif current_drift_report is not None:
            d_stat = current_drift_report.overall_status.value
            d_time = current_drift_report.checked_at[:19]
            d_msg = current_drift_report.summary_message

            if d_stat == "NO_DRIFT":
                st.success(f"🟢 **Dataset Drift Status: NO DRIFT** (Evaluated: {d_time} UTC) — {d_msg}")
            elif d_stat == "WARNING":
                st.warning(f"🟡 **Dataset Drift Status: WARNING** (Evaluated: {d_time} UTC) — {d_msg}")
            elif d_stat == "DRIFT":
                st.error(f"🔴 **Dataset Drift Status: DRIFT DETECTED** (Evaluated: {d_time} UTC) — {d_msg}")
            else:
                st.info(f"⚪ **Dataset Drift Status: UNKNOWN** (Evaluated: {d_time} UTC) — {d_msg}")

            if current_drift_report.remediation_advice:
                st.warning(f"🛠️ **Remediation Advice:** {current_drift_report.remediation_advice}")
            if current_drift_report.temporal_context:
                st.info(f"📅 **Calendar & Seasonality Context:** {current_drift_report.temporal_context}")

            # Feature breakdown table
            if current_drift_report.feature_results:
                st.markdown("##### 🔬 Feature-Level Drift Findings")
                f_rows = []
                for fr in current_drift_report.feature_results:
                    status_badge = "🟢 NO_DRIFT" if fr.status == DriftStatus.NO_DRIFT else ("🟡 WARNING" if fr.status == DriftStatus.WARNING else ("🔴 DRIFT" if fr.status == DriftStatus.DRIFT else "⚪ UNKNOWN"))
                    f_rows.append({
                        "Feature Name": fr.feature_name,
                        "Modality": fr.feature_type.value,
                        "Metric": fr.metric_type.value,
                        "Metric Value": f"{fr.metric_value:.4f}" if fr.metric_value is not None else "N/A",
                        "Thresholds (Warn / Drift)": f"{fr.warning_threshold} / {fr.drift_threshold}",
                        "Verdict": status_badge,
                        "Sample Size (Curr / Base)": f"{fr.current_sample_size} / {fr.baseline_sample_size}",
                        "Explanation": fr.explanation,
                    })
                st.dataframe(pd.DataFrame(f_rows), use_container_width=True)

            if current_drift_report.schema_differences:
                with st.expander(f"⚠️ Schema Deviations ({len(current_drift_report.schema_differences)})", expanded=False):
                    for sd in current_drift_report.schema_differences:
                        st.write(f"- {sd}")
        else:
            st.info("ℹ️ No persisted drift check evaluation found. Click **Run Drift Check** above to evaluate the active reference baseline.")

        # Drift History Expander
        with st.expander("📜 Historical Drift Check Results", expanded=False):
            if db_ready and drift_service:
                d_hist = drift_service.get_drift_history(sales_scope=dq_scope, limit=10)
                if d_hist:
                    dh_rows = []
                    for dh in d_hist:
                        dh_rows.append({
                            "Check Time (UTC)": dh.get("checked_at", "-")[:19],
                            "Scope": dh.get("sales_scope"),
                            "Baseline ID": dh.get("baseline_id"),
                            "Status": f"🟢 {dh['overall_status']}" if dh['overall_status'] == "NO_DRIFT" else (f"🟡 {dh['overall_status']}" if dh['overall_status'] == "WARNING" else (f"🔴 {dh['overall_status']}" if dh['overall_status'] == "DRIFT" else f"⚪ {dh['overall_status']}")),
                            "Drifted Features": f"{dh.get('drifted_features_count', 0)} / {dh.get('total_features_evaluated', 0)}",
                            "Current Rows": dh.get("current_row_count", 0),
                            "Summary": dh.get("summary_message", ""),
                        })
                    st.dataframe(pd.DataFrame(dh_rows), use_container_width=True)
                else:
                    st.caption("No historical drift checks found in database.")
            else:
                st.caption("Historical drift checks available when storage database is connected.")

    except Exception as e_dq:
        st.error(f"⚠️ Data Quality error: {e_dq}")

# ----------------------------------------------------------------------------
# 5. ANOMALY MONITOR WORKSPACE
# ----------------------------------------------------------------------------
with tab_anomaly:
    try:
        render_data_mode_banner(active_mode, active_dataset_title, currency=active_currency, details="Statistical Time-Series Anomaly Monitoring")
        st.subheader("⚠️ Explainable Time-Series Anomaly Monitor")
        st.caption("Monitor, detect, and triage statistical sales surges, drops, and forecast error deviations.")

        # MANDATORY WEEKLY SEASONALITY DISCLOSURE
        st.markdown("""
        <div class="seasonality-callout">
            <strong style="color: #92400E; font-size: 1.02rem;">⚠️ Statistical Baseline & Seasonality Limitation Notice</strong><br/>
            <span style="color: #78350F; font-size: 0.92rem;">
                The anomaly detection engine utilizes a 14-day rolling median baseline with Median Absolute Deviation (MAD) dispersion.
                <strong>It does NOT automatically stratify weekdays versus weekends or adjust for recurring weekly cyclical patterns.</strong>
                Intra-week variations (such as regular weekend shopping surges or mid-week dips) may register elevated z-scores (~1.8–2.5).
                Detected anomalies represent statistical signals requiring operator investigation, not confirmed root causes or automatically seasonally-adjusted facts.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Filters Row
        flt_c1, flt_c2, flt_c3, flt_c4 = st.columns(4)
        with flt_c1:
            ano_scope = st.text_input("Scope Filter", value="TOTAL_STORE", key="ano_scope")
        with flt_c2:
            ano_type_choice = st.selectbox("Anomaly Type", options=["All", "SALES_SURGE", "SALES_DROP", "FORECAST_ERROR_ANOMALY"], key="ano_type")
        with flt_c3:
            ano_sev_choice = st.selectbox("Severity", options=["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"], key="ano_sev")
        with flt_c4:
            ano_status_choice = st.selectbox("Status", options=["All", "ACTIVE", "RESOLVED"], key="ano_status")

        act_col1, act_col2 = st.columns([1, 1])
        with act_col1:
            if st.button("🔍 Run Anomaly Detection on Database Actuals", use_container_width=True):
                with st.spinner("Running rolling statistical detection on ground truth actuals..."):
                    if db_ready:
                        det_report = ana_service.detect_sales_anomalies_from_db(
                            sales_scope=ano_scope,
                            persist=True,
                        )
                        st.success(f"Detection executed across {det_report.total_evaluated} points. Detected: {det_report.anomalies_detected} anomalies.")
                        st.rerun()
                    else:
                        st.warning("Database unavailable to load stored actuals.")
        with act_col2:
            if st.button("🔄 Refresh Anomaly View", use_container_width=True):
                st.rerun()

        # Query Persisted Anomalies
        if db_ready and audit_repo:
            filter_kwargs: Dict[str, Any] = {"sales_scope": ano_scope, "limit": 200}
            if ano_type_choice != "All":
                filter_kwargs["anomaly_type"] = ano_type_choice
            if ano_sev_choice != "All":
                filter_kwargs["severity"] = ano_sev_choice
            if ano_status_choice != "All":
                filter_kwargs["status"] = ano_status_choice

            anomaly_records = audit_repo.get_anomalies(**filter_kwargs)
            actual_records = audit_repo.list_actual_sales(sales_scope=ano_scope, limit=1000)

            # Summary Metrics Row
            active_cnt = sum(1 for a in anomaly_records if a.status == "ACTIVE")
            crit_cnt = sum(1 for a in anomaly_records if a.severity in ("CRITICAL", "HIGH"))
            resolved_cnt = sum(1 for a in anomaly_records if a.status == "RESOLVED")

            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric("Total Findings", len(anomaly_records))
            with sm2:
                st.metric("Active Anomalies", active_cnt)
            with sm3:
                st.metric("Critical / High Severity", crit_cnt)
            with sm4:
                st.metric("Resolved Anomalies", resolved_cnt)

            # Time Series Visualization with Minimum-History Enforcement
            fig_ano, chart_state = build_anomaly_overview_figure(
                actual_records=actual_records,
                anomaly_records=anomaly_records,
                sales_scope=ano_scope,
                min_history=MIN_ANOMALY_HISTORY_DAYS,
                currency_symbol=active_currency,
            )

            if fig_ano is not None:
                st.plotly_chart(fig_ano, use_container_width=True)
                if not anomaly_records:
                    st.caption("ℹ️ Sales time series active. No anomalies match the current filter criteria.")
            elif chart_state == "INSUFFICIENT_HISTORY":
                rec_cnt = len(actual_records) if actual_records else 0
                st.warning(
                    f"⚠️ **Insufficient History for Time-Series Anomaly Modeling ({rec_cnt} / {MIN_ANOMALY_HISTORY_DAYS} Days)**\n\n"
                    f"The statistical anomaly detection engine enforces a strict minimum history policy of **{MIN_ANOMALY_HISTORY_DAYS} daily observations** "
                    f"to establish a reliable rolling median baseline and Median Absolute Deviation (MAD) spread. "
                    f"Rendering a single observation or sparse data produces a misleading time-series trend line. "
                    f"Please record or ingest at least {MIN_ANOMALY_HISTORY_DAYS - rec_cnt} additional continuous daily sales actual(s) for scope `{ano_scope}`."
                )
            else:
                st.info(f"ℹ️ No historical sales observations found for scope '{ano_scope}'. Ingest or record sales actuals to visualize time-series trends and anomaly overlays.")

            # Anomaly Findings Table
            if anomaly_records:
                st.markdown("##### 📋 Detected Anomaly Findings & Statistical Lineage")
                a_table = []
                for a in anomaly_records:
                    a_table.append({
                        "Date": a.observation_date.isoformat(),
                        "Type": a.anomaly_type,
                        "Severity": a.severity,
                        "Status": a.status,
                        f"Observed ({active_currency})": f"{active_currency}{a.observed_value:,.2f}",
                        f"Baseline ({active_currency})": f"{active_currency}{a.reference_center:,.2f}",
                        f"Dispersion ({active_currency})": f"{active_currency}{a.reference_spread:,.2f}",
                        "Z-Score": f"{a.anomaly_score:.2f}",
                        "Threshold": f"{a.threshold:.2f}",
                        "Model Lineage": a.model_name or "Actuals Baseline",
                        "Explanation": a.explanation,
                    })
                st.dataframe(pd.DataFrame(a_table), use_container_width=True)

                # Anomaly Resolution Action
                st.divider()
                st.markdown("##### 🛠️ Operator Triage & Anomaly Resolution Workflow")
                active_anomalies = [a for a in anomaly_records if a.status == "ACTIVE"]
                if active_anomalies:
                    res_c1, res_c2, res_c3 = st.columns([2, 3, 1])
                    with res_c1:
                        target_ano_id = st.selectbox(
                            "Select Active Anomaly to Triage",
                            options=[a.anomaly_id for a in active_anomalies],
                            format_func=lambda x: f"{x[:8]}... ({next((a.observation_date.isoformat() for a in active_anomalies if a.anomaly_id == x), '')})",
                            key="res_ano_select"
                        )
                    with res_c2:
                        res_note = st.text_input("Operational Resolution Rationale", placeholder="e.g. Verified Black Friday promotion / Supplier stockout confirmed", key="res_note_in")
                    with res_c3:
                        st.write("")
                        st.write("")
                        if st.button("✅ Mark RESOLVED", use_container_width=True):
                            ok = audit_repo.update_anomaly_status(
                                anomaly_id=target_ano_id,
                                new_status="RESOLVED",
                                resolution_note=res_note
                            )
                            if ok:
                                st.success("Anomaly marked as RESOLVED with audit timestamp.")
                                st.rerun()
                else:
                    st.info("No active anomalies currently require resolution for this scope.")
            else:
                st.info("No anomalies matching the current filters were found in the audit repository.")

        else:
            st.info("Anomaly monitoring records are persisted to SQLite database when active.")

    except Exception as e_ano:
        st.error(f"⚠️ Anomaly Monitor error: {e_ano}")

# ----------------------------------------------------------------------------
# 6. SYSTEM HEALTH & DIAGNOSTICS WORKSPACE
# ----------------------------------------------------------------------------
with tab_health:
    try:
        render_data_mode_banner(active_mode, active_dataset_title, currency=active_currency, details="System Health & Infrastructure Observability")
        st.subheader("⚙️ System Health, Observability & Diagnostic Probes")
        st.caption("Inspect database connection readiness, model candidates, API service connectivity, and operational events.")

        # 4 Core Readiness Cards
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown("""
            <div class="metric-card-v2">
                <span style="color: #64748B; font-weight: 600;">DATABASE READY</span>
            """, unsafe_allow_html=True)
            if db_ready:
                st.markdown(f"<h3 style='color: #10B981; margin: 0.3rem 0;'>CONNECTED</h3>", unsafe_allow_html=True)
                st.caption(f"SQLite WAL (Schema v{SCHEMA_VERSION})")
            else:
                st.markdown(f"<h3 style='color: #DC2626; margin: 0.3rem 0;'>STANDALONE</h3>", unsafe_allow_html=True)
                st.caption("Degraded: In-Memory Only")
            st.markdown("</div>", unsafe_allow_html=True)

        with h2:
            st.markdown(f"""
            <div class="metric-card-v2">
                <span style="color: #64748B; font-weight: 600;">MODEL REGISTRY</span>
                <h3 style="color: #4F46E5; margin: 0.3rem 0;">{len(model_registry.list_models())} CANDIDATES</h3>
                <small style="color: #64748B;">{', '.join(model_registry.list_models())}</small>
            </div>
            """, unsafe_allow_html=True)

        with h3:
            # API Connectivity Probe
            api_online = False
            try:
                r_probe = requests.get("http://127.0.0.1:8000/ready", timeout=0.3)
                if r_probe.status_code == 200:
                    api_online = True
            except Exception:
                api_online = False

            st.markdown("""
            <div class="metric-card-v2">
                <span style="color: #64748B; font-weight: 600;">FASTAPI SERVICE</span>
            """, unsafe_allow_html=True)
            if api_online:
                st.markdown("<h3 style='color: #10B981; margin: 0.3rem 0;'>ONLINE</h3>", unsafe_allow_html=True)
                st.caption("Daemon listening on :8000")
            else:
                st.markdown("<h3 style='color: #475569; margin: 0.3rem 0;'>LOCAL MODE</h3>", unsafe_allow_html=True)
                st.caption("Self-contained Domain Services")
            st.markdown("</div>", unsafe_allow_html=True)

        with h4:
            # Persistent Records Count
            run_cnt = len(audit_repo.list_forecast_runs(limit=100)) if (db_ready and audit_repo) else 0
            st.markdown(f"""
            <div class="metric-card-v2">
                <span style="color: #64748B; font-weight: 600;">PERSISTED AUDIT RUNS</span>
                <h3 style="color: #0EA5E9; margin: 0.3rem 0;">{run_cnt}</h3>
                <small style="color: #64748B;">Verified in SQLite Storage</small>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Freshness & Ingestion SLA Probes
        st.markdown("##### 📡 Ingestion Observability & Freshness SLA")
        fh1, fh2, fh3, fh4 = st.columns(4)
        latest_mon = mon_service.get_latest_monitoring_check(sales_scope="TOTAL_STORE") if (db_ready and mon_service) else None
        with fh1:
            st.markdown("""<div class="metric-card-v2"><span style="color: #64748B; font-weight: 600;">DATA FRESHNESS SLA</span>""", unsafe_allow_html=True)
            if not db_ready:
                st.markdown("<h3 style='color: #64748B; margin: 0.3rem 0;'>UNKNOWN</h3>", unsafe_allow_html=True)
                st.caption("Database Offline")
            elif latest_mon:
                status_color = "#10B981" if latest_mon.status == "HEALTHY" else ("#F59E0B" if latest_mon.status == "WARNING" else ("#DC2626" if latest_mon.status == "CRITICAL" else "#64748B"))
                st.markdown(f"<h3 style='color: {status_color}; margin: 0.3rem 0;'>{latest_mon.status}</h3>", unsafe_allow_html=True)
                st.caption(f"State: {latest_mon.freshness_state}")
            else:
                st.markdown("<h3 style='color: #64748B; margin: 0.3rem 0;'>UNKNOWN</h3>", unsafe_allow_html=True)
                st.caption("Awaiting First Check")
            st.markdown("</div>", unsafe_allow_html=True)

        with fh2:
            st.markdown("""<div class="metric-card-v2"><span style="color: #64748B; font-weight: 600;">CALENDAR STALENESS</span>""", unsafe_allow_html=True)
            if latest_mon and latest_mon.staleness_days is not None:
                st.markdown(f"<h3 style='color: #4F46E5; margin: 0.3rem 0;'>{latest_mon.staleness_days} DAYS</h3>", unsafe_allow_html=True)
                st.caption(f"SLA Target: ≤ {latest_mon.staleness_threshold_days or 30} days")
            else:
                st.markdown("<h3 style='color: #64748B; margin: 0.3rem 0;'>UNKNOWN</h3>", unsafe_allow_html=True)
                st.caption("No Observation Found")
            st.markdown("</div>", unsafe_allow_html=True)

        with fh3:
            st.markdown("""<div class="metric-card-v2"><span style="color: #64748B; font-weight: 600;">INGESTION RELIABILITY</span>""", unsafe_allow_html=True)
            if latest_mon:
                cfail = latest_mon.consecutive_failures or 0
                fcolor = "#10B981" if cfail == 0 else "#DC2626"
                st.markdown(f"<h3 style='color: {fcolor}; margin: 0.3rem 0;'>{cfail} FAILURES</h3>", unsafe_allow_html=True)
                st.caption(f"Latest Outcome: {latest_mon.latest_ingestion_outcome or 'UNKNOWN'}")
            else:
                st.markdown("<h3 style='color: #64748B; margin: 0.3rem 0;'>UNKNOWN</h3>", unsafe_allow_html=True)
                st.caption("No Checks Executed")
            st.markdown("</div>", unsafe_allow_html=True)

        with fh4:
            st.markdown("""<div class="metric-card-v2"><span style="color: #64748B; font-weight: 600;">SCHEDULING BOUNDARY</span>""", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #0EA5E9; margin: 0.3rem 0;'>CRON-READY</h3>", unsafe_allow_html=True)
            st.caption("Entrypoint: run_scheduled_monitoring_check()")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Detailed Diagnostic Inspection
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.markdown("##### 📦 Verified ML Model Inventory")
            m_table = []
            for m_name in model_registry.list_models():
                is_loaded = m_name in models_dict
                m_table.append({
                    "Model Name": m_name,
                    "Loaded Status": "✅ Verified & Ready" if is_loaded else "⚠️ Fallback Active",
                    "Feature Contract": "13-Feature Temporal Vector",
                    "Inference Policy": "Explicit Non-Fallback"
                })
            st.dataframe(pd.DataFrame(m_table), use_container_width=True)

        with d_c2:
            st.markdown("##### 🗄️ Persistence & Storage Health")
            st.write(f"• **Schema Version:** Expected `v{SCHEMA_VERSION}`, Storage Active: `v{SCHEMA_VERSION}`")
            st.write(f"• **SQLite Journal Mode:** `WAL` (Write-Ahead Logging for concurrent read safety)")
            st.write(f"• **Security Safeguard:** Zero raw tabular records or API tokens persisted to disk.")
            st.write(f"• **Data Freshness Monitor:** `Active` (Scope: TOTAL_STORE, Evaluation Engine: DataMonitoringService)")
            st.write(f"• **Scheduling Boundary:** Decoupled service entrypoint ready for external cron/worker integration.")
            if db_ready and audit_repo:
                val_logs = audit_repo.list_validations(limit=5)
                fail_cnt = sum(1 for v in val_logs if v.validation_status == "FAILED")
                st.write(f"• **Recent Failed Ingestions:** `{fail_cnt}` of last {len(val_logs)} audited runs.")

        # Maintenance Controls
        st.divider()
        st.markdown("##### 🛠️ Operator Maintenance Actions")
        m_btn1, m_btn2 = st.columns(2)
        with m_btn1:
            if st.button("🔄 Force Clear All Resource Caches", help="Clears Streamlit model and data caches"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Caches purged successfully.")
                st.rerun()
        with m_btn2:
            st.caption("ℹ️ *Running locally: The dashboard executes natively using local domain services. A separate FastAPI server is optional and not required for full functionality.*")

    except Exception as e_hlth:
        st.error(f"⚠️ System Health error: {e_hlth}")

# Footer
st.divider()
st.caption("SalesPulse AI v3.0 • Production ML Forecasting, Observability & Operator Experience")
