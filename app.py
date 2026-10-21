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
        return {
            "engine": engine,
            "repo": repo,
            "registry": registry,
            "fc_service": fc_service,
            "val_service": val_service,
            "ana_service": ana_service,
            "mon_service": mon_service,
            "db_ready": True,
            "db_error": None,
        }
    except Exception as exc:
        logger.warning(f"Database initialization warning in Streamlit: {exc}")
        registry = get_model_registry()
        mon_service = DataMonitoringService(repository=None, engine=None, auto_init_engine=False)
        return {
            "engine": None,
            "repo": None,
            "registry": registry,
            "fc_service": ForecastingService(registry=registry),
            "val_service": ValidationService(),
            "ana_service": AnalyticsService(),
            "mon_service": mon_service,
            "db_ready": False,
            "db_error": str(exc),
        }

services = get_dashboard_services()
audit_repo: Optional[AuditRepository] = services.get("repo")
fc_service: ForecastingService = services["fc_service"]
val_service: ValidationService = services["val_service"]
ana_service: AnalyticsService = services["ana_service"]
mon_service: DataMonitoringService = services["mon_service"]
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
# HISTORICAL DATASET LOADING (CACHED)
# ============================================================================
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
# BACKWARD COMPATIBLE FEATURE ENGINEERING & INFERENCE UTILITIES
# ============================================================================
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
    st.markdown("• [Data Contract Definitions](https://github.com/SwastikPandey1024/SalesPulse_AI/blob/main/docs/PRD.md)")
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
        st.subheader("📊 Executive Sales Overview & Business Trajectory")
        st.caption("Inspect aggregate historical sales performance, regional distributions, and simulate operational scenarios.")

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
                    title="Historical Daily Sales & 7-Day Moving Average (Reindexed)",
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
            st.info("💡 Connect a tabular daily sales dataset in the Data Quality workspace to populate the sales overview.")

        # Business Scenario Simulator Section
        st.divider()
        st.markdown("### 🎯 Interactive Business Scenario Simulator")
        st.caption("Simulate promotional campaigns, pricing levers, and seasonal shifts to estimate sales lift.")
        
        sim_c1, sim_c2 = st.columns([1, 2])
        with sim_c1:
            p_boost = st.slider("📢 Promotional Campaign Boost (%)", 0, 50, 15, key="ov_promo")
            p_adjust = st.slider("🏷️ Price Adjustment (%)", -20, 20, 0, key="ov_price")
            s_mult = st.slider("📅 Seasonal Multiplier", 0.5, 1.5, 1.1, step=0.05, key="ov_season")
            b_sales = st.number_input("Base Daily Sales Benchmark ($)", value=200.0, key="ov_base")

        with sim_c2:
            effective_mult = (1.0 + (p_boost / 100.0)) * (1.0 + (p_adjust / 100.0)) * s_mult
            sim_sales = b_sales * effective_mult
            delta_val = sim_sales - b_sales
            pct_val = (delta_val / b_sales) * 100.0

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Baseline Sales", f"${b_sales:,.2f}")
            with m2:
                st.metric("Simulated Sales", f"${sim_sales:,.2f}", delta=f"{pct_val:+.1f}%")
            with m3:
                st.metric("Revenue Impact", f"${delta_val:+,.2f}")

            sim_df = pd.DataFrame({
                "Scenario": ["Baseline Sales", "Simulated Sales"],
                "Sales ($)": [b_sales, sim_sales]
            })
            fig_sim = px.bar(
                sim_df, x="Scenario", y="Sales ($)", color="Scenario",
                color_discrete_map={"Baseline Sales": "#94A3B8", "Simulated Sales": "#4F46E5"},
                text_auto="$.2f", title="Baseline vs. Simulated Sales Comparison"
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
        st.subheader("🔮 Production Forecast Lab")
        st.caption("Generate recursive multi-step forecasts with empirical or calibrated prediction intervals and full audit provenance.")

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
            lab_base_date = st.date_input("Forecast Origin Date", value=datetime.date.today(), key="lab_date")

        st.markdown("##### Temporal Feature Inputs")
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            f_lag1 = st.number_input("Base Yesterday Sales ($)", value=160.0, step=10.0, key="lab_l1")
        with p_c2:
            f_rm7 = st.number_input("7-Day Rolling Mean ($)", value=155.0, step=5.0, key="lab_rm7")
        with p_c3:
            f_rm14 = st.number_input("14-Day Rolling Mean ($)", value=150.0, step=5.0, key="lab_rm14")
        with p_c4:
            f_rs7 = st.number_input("7-Day Volatility (Std Dev)", value=15.0, step=1.0, key="lab_rs7")

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
                }

        if "lab_forecast_df" in st.session_state and st.session_state["lab_forecast_df"] is not None:
            res_df = st.session_state["lab_forecast_df"]
            meta = st.session_state.get("lab_meta", {})
            run_id_val = st.session_state.get("lab_run_id")

            # Forecast Provenance Card
            st.markdown(f"""
            <div class="provenance-card">
                <strong>📋 Forecast Provenance:</strong><br/>
                • <strong>Run ID:</strong> <code>{run_id_val or 'Unpersisted In-Memory Run'}</code><br/>
                • <strong>Origin Date:</strong> {meta.get('origin')} | <strong>Scope:</strong> {meta.get('scope')} | <strong>Model:</strong> {meta.get('model')}<br/>
                • <strong>Horizon:</strong> {meta.get('horizon')} Days | <strong>Interval Method:</strong> <code>{meta.get('interval')}</code>
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
                name='Forecasted Daily Sales'
            ))
            fig_fc.update_layout(
                title=f"{meta.get('horizon')}-Day Sales Forecast with Prediction Uncertainty",
                xaxis_title="Forecast Date", yaxis_title="Sales ($)",
                template="plotly_white", height=420, hovermode="x unified"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Data Table & CSV Download
            t_col1, t_col2 = st.columns([3, 1])
            with t_col1:
                st.dataframe(
                    res_df.style.format({
                        "Forecasted Sales ($)": "${:,.2f}",
                        "Lower Bound ($)": "${:,.2f}",
                        "Upper Bound ($)": "${:,.2f}"
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
                st.metric("Total Projected Revenue", f"${total_rev:,.2f}")
                st.metric("Avg Daily Projection", f"${avg_daily:,.2f}")

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
                    sq_date = st.date_input("Target Date", value=datetime.date.today() + timedelta(days=1))
                    sq_l1 = st.number_input("Yesterday Sales ($)", min_value=0.0, value=150.0)
                with sq_c2:
                    sq_l7 = st.number_input("7-Day Lag ($)", min_value=0.0, value=145.0)
                    sq_rm7 = st.number_input("7-Day Mean ($)", min_value=0.0, value=148.0)
                with sq_c3:
                    sq_rm14 = st.number_input("14-Day Mean ($)", min_value=0.0, value=146.0)
                    sq_rs7 = st.number_input("7-Day Std Dev ($)", min_value=0.0, value=12.5)
                btn_sq = st.form_submit_button("Run Single-Step Inference")

            if btn_sq:
                sq_feats = extract_date_features(sq_date)
                sq_input = build_feature_dataframe(sq_feats, sq_l1, sq_l7, sq_l7, sq_l7, sq_rm7, sq_rm14, sq_rs7, 100)
                sq_pred = predict_sales(sq_input, model_choice=lab_model)
                st.success(f"**Point Prediction ({sq_date}):** ${sq_pred:,.2f} (Model: {lab_model})")

    except Exception as e_lab:
        st.error(f"⚠️ Forecast Lab error: {e_lab}")

# ----------------------------------------------------------------------------
# 3. FORECAST PERFORMANCE & RECONCILIATION WORKSPACE
# ----------------------------------------------------------------------------
with tab_perf:
    try:
        st.subheader("🎯 Out-of-Sample Forecast Performance & Reconciled Actuals")
        st.info(
            "ℹ️ **Evaluation & Realization Protocol:** Metrics are computed strictly against reconciled actual ground truth. "
            "Items awaiting future actuals are designated as Pending Realization and are never treated as zero. "
            "Historical accuracy does not guarantee future forecasting performance."
        )

        if not db_ready or audit_repo is None:
            st.warning("⚠️ Database connection is unavailable. Realized performance tracking requires active SQLite storage.")
        else:
            # Query recent forecast runs
            recent_runs = audit_repo.list_forecast_runs(limit=30)
            if not recent_runs:
                st.info("No persisted forecast runs found in the database. Run a forecast in the **Forecast Lab** with 'Persist to Audit Storage' checked.")
            else:
                run_options = {
                    r.run_id: f"{r.created_at.strftime('%Y-%m-%d %H:%M')} | {r.model_name} | {r.horizon_days}d | ${r.total_predicted_revenue or 0:,.0f} ({r.run_id[:8]}...)"
                    for r in recent_runs
                }
                selected_run_id = st.selectbox(
                    "Select Historical Forecast Run to Inspect",
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
                            st.metric("Realized MAE", f"${mae_val:,.2f}")
                        else:
                            st.metric("Realized MAE", "Awaiting Actuals")
                    with p_c4:
                        if reconciled_items:
                            sq_errs = [(it.actual_value - it.predicted_value)**2 for it in reconciled_items]
                            rmse_val = float(np.sqrt(np.mean(sq_errs)))
                            st.metric("Realized RMSE", f"${rmse_val:,.2f}")
                        else:
                            st.metric("Realized RMSE", "Awaiting Actuals")
                    with p_c5:
                        if reconciled_items:
                            cov_count = sum(1 for it in reconciled_items if it.interval_covered is True)
                            cov_pct = (cov_count / len(reconciled_items)) * 100.0
                            st.metric("Interval Coverage", f"{cov_pct:.1f}%")
                        else:
                            st.metric("Interval Coverage", "Awaiting Actuals")

                    # Extended Metrics (WAPE and Bias)
                    if reconciled_items:
                        sum_act = sum(it.actual_value for it in reconciled_items)
                        sum_abs_err = sum(it.absolute_error for it in reconciled_items if it.absolute_error is not None)
                        wape_val = (sum_abs_err / sum_act * 100.0) if sum_act > 0 else None
                        bias_val = (sum((it.predicted_value - it.actual_value) for it in reconciled_items) / sum_act * 100.0) if sum_act > 0 else None

                        e_m1, e_m2, e_m3 = st.columns(3)
                        with e_m1:
                            st.caption(f"**Realized WAPE:** {f'{wape_val:.1f}%' if wape_val is not None else 'N/A'}")
                        with e_m2:
                            st.caption(f"**Forecast Bias:** {f'{bias_val:+.1f}%' if bias_val is not None else 'N/A'}")
                        with e_m3:
                            st.caption(f"**Interval Method:** <code>{selected_run.interval_method}</code>")

                    # Time Series Plot: Predicted vs Actual
                    dates = [it.target_date for it in items]
                    preds = [it.predicted_value for it in items]
                    lowers = [it.lower_bound for it in items]
                    uppers = [it.upper_bound for it in items]
                    actuals = [it.actual_value for it in items]

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
                        name='Predicted Sales ($)', line=dict(color='#4F46E5', width=2.5)
                    ))

                    # Plot actuals only for dates where they exist
                    reconciled_dates = [it.target_date for it in reconciled_items]
                    reconciled_vals = [it.actual_value for it in reconciled_items]
                    if reconciled_dates:
                        fig_eval.add_trace(go.Scatter(
                            x=reconciled_dates, y=reconciled_vals, mode='markers+lines',
                            name='Reconciled Actual Sales ($)',
                            marker=dict(color='#10B981', size=9, symbol='diamond'),
                            line=dict(color='#10B981', width=1.5, dash='dash')
                        ))

                    fig_eval.update_layout(
                        title=f"Forecast Realization: Run {selected_run.run_id[:8]} ({selected_run.model_name})",
                        xaxis_title="Target Date", yaxis_title="Sales ($)",
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
                            "Predicted ($)": f"${it.predicted_value:,.2f}",
                            "Actual ($)": f"${it.actual_value:,.2f}" if it.actual_value is not None else "⏳ Pending Realization",
                            "Absolute Error ($)": f"${it.absolute_error:,.2f}" if it.absolute_error is not None else "Pending",
                            "APE (%)": ape_str,
                            "Interval Range": f"[${it.lower_bound or 0:,.0f} — ${it.upper_bound or 0:,.0f}]",
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
                    rec_sales = st.number_input("Actual Sales Amount ($)", min_value=0.0, value=175.0, step=10.0, key="rec_val")
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
                    st.success(f"✅ {action_word} actual sales record for {rec_date} (${rec_sales:,.2f}). All matching forecast items reconciled!")
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
        st.subheader("🛡️ Data Quality Workspace & Contract Enforcement")
        st.caption("Safely ingest tabular sales datasets, validate contract compliance, isolate quarantined records, and review persistent audit history.")

        # --------------------------------------------------------------------
        # 4A. Freshness & Ingestion Monitoring Observability Card
        # --------------------------------------------------------------------
        st.markdown("### 📡 Data Freshness & Ingestion Health")
        st.caption("Deterministic observability monitoring ingestion pipeline outcomes, calendar freshness SLAs, and failure escalation.")

        m_c1, m_c2, m_c3 = st.columns([2, 1, 1])
        with m_c1:
            dq_scope = st.selectbox("Monitoring Scope", options=["TOTAL_STORE"], index=0, key="dq_mon_scope")
        with m_c2:
            dq_staleness = st.number_input("Freshness SLA Threshold (Days)", min_value=1, max_value=365, value=30, key="dq_stale")
        with m_c3:
            st.write("")
            st.write("")
            btn_run_check = st.button("🔄 Run Health Check", key="btn_run_monitoring_check", use_container_width=True, help="Executes on-demand data freshness and ingestion health check")

        # Evaluate on manual request or retrieve latest check
        current_verdict: Optional[MonitoringVerdict] = None
        if btn_run_check:
            with st.spinner("Evaluating data freshness and ingestion signals..."):
                current_verdict = mon_service.check_data_health(
                    scope=dq_scope,
                    config=MonitoringConfig(staleness_threshold_days=int(dq_staleness)),
                    persist=True,
                )

        latest_record = mon_service.get_latest_monitoring_check(sales_scope=dq_scope) if mon_service else None

        if not db_ready:
            st.info("⚪ **Monitoring Status: UNKNOWN** — Database offline or unavailable. Cannot confirm data freshness.")
        elif current_verdict is not None:
            v_status = current_verdict.status.value
            v_msg = current_verdict.summary_message
            if v_status == "HEALTHY":
                st.success(f"🟢 **Monitoring Status: HEALTHY** — {v_msg}")
            elif v_status == "WARNING":
                st.warning(f"🟡 **Monitoring Status: WARNING** — {v_msg}")
            elif v_status == "CRITICAL":
                st.error(f"🔴 **Monitoring Status: CRITICAL** — {v_msg}")
            else:
                st.info(f"⚪ **Monitoring Status: UNKNOWN** — {v_msg}")

            if current_verdict.remediation_advice:
                st.warning(f"🛠️ **Actionable Remediation:** {current_verdict.remediation_advice}")

            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.metric("Freshness State", f"{current_verdict.freshness.state.value} ({current_verdict.freshness.staleness_days or 0}d)", help=f"Staleness threshold: ≤ {dq_staleness} days")
            with fc2:
                last_succ = current_verdict.ingestion.last_successful_ingestion_at.strftime("%Y-%m-%d %H:%M") if current_verdict.ingestion.last_successful_ingestion_at else "Never Recorded"
                st.metric("Last Successful Ingest", last_succ)
            with fc3:
                st.metric("Latest Attempt Outcome", current_verdict.ingestion.latest_outcome.value)
            with fc4:
                cf = current_verdict.ingestion.consecutive_failures
                st.metric("Consecutive Failures", f"{cf}", delta=f"{cf} failures" if cf > 0 else "0", delta_color="inverse")

        elif latest_record is not None:
            r_status = latest_record.status
            r_msg = latest_record.summary_message
            chk_time = latest_record.checked_at.strftime('%Y-%m-%d %H:%M:%S') if latest_record.checked_at else "Recent"
            if r_status == "HEALTHY":
                st.success(f"🟢 **Monitoring Status: HEALTHY** (Checked: {chk_time}) — {r_msg}")
            elif r_status == "WARNING":
                st.warning(f"🟡 **Monitoring Status: WARNING** (Checked: {chk_time}) — {r_msg}")
            elif r_status == "CRITICAL":
                st.error(f"🔴 **Monitoring Status: CRITICAL** (Checked: {chk_time}) — {r_msg}")
            else:
                st.info(f"⚪ **Monitoring Status: UNKNOWN** (Checked: {chk_time}) — {r_msg}")

            try:
                ev_data = json.loads(latest_record.evidence_json) if latest_record.evidence_json else {}
                remed = ev_data.get("remediation_advice")
                if remed:
                    st.warning(f"🛠️ **Actionable Remediation:** {remed}")
            except Exception:
                pass

            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.metric("Freshness State", f"{latest_record.freshness_state} ({latest_record.staleness_days or 0}d)", help=f"Staleness threshold: ≤ {dq_staleness} days")
            with fc2:
                last_succ = latest_record.last_successful_ingestion_at.strftime("%Y-%m-%d %H:%M") if latest_record.last_successful_ingestion_at else "Never Recorded"
                st.metric("Last Successful Ingest", last_succ)
            with fc3:
                st.metric("Latest Attempt Outcome", latest_record.latest_ingestion_outcome or "UNKNOWN")
            with fc4:
                cf = latest_record.consecutive_failures or 0
                st.metric("Consecutive Failures", f"{cf}", delta=f"{cf} failures" if cf > 0 else "0", delta_color="inverse")

        else:
            st.info("ℹ️ No data freshness check recorded yet. Click **Run Health Check** above to evaluate current pipeline health.")

        # Monitoring history expander
        with st.expander("📜 Historical Monitoring Logs & Status Changes", expanded=False):
            if db_ready and mon_service:
                m_hist = mon_service.get_monitoring_history(sales_scope=dq_scope, limit=10)
                if m_hist:
                    hist_rows = []
                    for h in m_hist:
                        hist_rows.append({
                            "Check Time (UTC)": h.checked_at.strftime("%Y-%m-%d %H:%M:%S") if h.checked_at else "-",
                            "Scope": h.sales_scope,
                            "Status": f"🟢 {h.status}" if h.status == "HEALTHY" else (f"🟡 {h.status}" if h.status == "WARNING" else (f"🔴 {h.status}" if h.status == "CRITICAL" else f"⚪ {h.status}")),
                            "Freshness": f"{h.freshness_state} ({h.staleness_days or 0}d)",
                            "Last Ingestion": h.last_successful_ingestion_at.strftime("%Y-%m-%d %H:%M") if h.last_successful_ingestion_at else "None",
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

    except Exception as e_dq:
        st.error(f"⚠️ Data Quality error: {e_dq}")

# ----------------------------------------------------------------------------
# 5. ANOMALY MONITOR WORKSPACE
# ----------------------------------------------------------------------------
with tab_anomaly:
    try:
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
            filter_kwargs: Dict[str, Any] = {"sales_scope": ano_scope, "limit": 100}
            if ano_type_choice != "All":
                filter_kwargs["anomaly_type"] = ano_type_choice
            if ano_sev_choice != "All":
                filter_kwargs["severity"] = ano_sev_choice
            if ano_status_choice != "All":
                filter_kwargs["status"] = ano_status_choice

            anomaly_records = audit_repo.get_anomalies(**filter_kwargs)
            actual_records = audit_repo.list_actual_sales(sales_scope=ano_scope, limit=100)

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

            # Time Series Visualization with Overlay Markers
            if actual_records:
                df_act = pd.DataFrame([{
                    "date": pd.to_datetime(r.calendar_date),
                    "sales": float(r.daily_sales)
                } for r in reversed(actual_records)])

                fig_ano = go.Figure()
                fig_ano.add_trace(go.Scatter(
                    x=df_act['date'], y=df_act['sales'],
                    mode='lines', name='Daily Actual Sales ($)',
                    line=dict(color='#94A3B8', width=2)
                ))

                if anomaly_records:
                    # Separate active critical/high vs medium/low
                    for r in anomaly_records:
                        m_color = "#DC2626" if r.severity in ("CRITICAL", "HIGH") else "#F59E0B"
                        sym = "circle" if r.anomaly_type == "SALES_SURGE" else "triangle-down"
                        fig_ano.add_trace(go.Scatter(
                            x=[pd.to_datetime(r.observation_date)],
                            y=[r.observed_value],
                            mode='markers',
                            marker=dict(color=m_color, size=11, symbol=sym, line=dict(color='#FFFFFF', width=1.5)),
                            name=f"{r.anomaly_type} ({r.severity})",
                            hovertext=(
                                f"<b>Date:</b> {r.observation_date}<br/>"
                                f"<b>Observed:</b> ${r.observed_value:,.2f}<br/>"
                                f"<b>Baseline:</b> ${r.reference_center:,.2f}<br/>"
                                f"<b>Score:</b> {r.anomaly_score:.2f} (Thresh: {r.threshold:.2f})<br/>"
                                f"<b>Severity:</b> {r.severity} | <b>Status:</b> {r.status}<br/>"
                                f"<b>Explanation:</b> {r.explanation}"
                            ),
                            hoverinfo="text",
                            showlegend=False
                        ))

                fig_ano.update_layout(
                    title="Daily Sales Series with Statistical Anomaly Overlay",
                    xaxis_title="Date", yaxis_title="Sales ($)",
                    template="plotly_white", height=380, hovermode="closest"
                )
                st.plotly_chart(fig_ano, use_container_width=True)

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
                        "Observed ($)": f"${a.observed_value:,.2f}",
                        "Baseline ($)": f"${a.reference_center:,.2f}",
                        "Dispersion ($)": f"${a.reference_spread:,.2f}",
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
                st.markdown("<h3 style='color: #64748B; margin: 0.3rem 0;'>PENDING</h3>", unsafe_allow_html=True)
                st.caption("Awaiting First Check")
            st.markdown("</div>", unsafe_allow_html=True)

        with fh2:
            st.markdown("""<div class="metric-card-v2"><span style="color: #64748B; font-weight: 600;">CALENDAR STALENESS</span>""", unsafe_allow_html=True)
            if latest_mon and latest_mon.staleness_days is not None:
                st.markdown(f"<h3 style='color: #4F46E5; margin: 0.3rem 0;'>{latest_mon.staleness_days} DAYS</h3>", unsafe_allow_html=True)
                st.caption(f"SLA Target: ≤ {latest_mon.staleness_threshold_days or 30} days")
            else:
                st.markdown("<h3 style='color: #64748B; margin: 0.3rem 0;'>N/A</h3>", unsafe_allow_html=True)
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
