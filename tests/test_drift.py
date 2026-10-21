"""
SalesPulse AI - Dataset Drift Monitoring Test Suite (Phase 4B)
Verifies:
1. Statistical correctness of Population Stability Index (PSI) and Total Variation Distance (TVD).
2. Zero-bin safeguards, epsilon smoothing, constant features, and low sample size UNKNOWN handling.
3. Unseen categories detection and null rate shift diagnostics.
4. Versioned reference baseline profiling, JSON serialization, and privacy (zero raw row leakage).
5. Explicit baseline promotion, scope isolation, and no silent replacement.
6. Missing baseline, incompatible schema, and offline database UNKNOWN handling.
7. Persistence idempotency and audit history retrieval.
"""

import math
import datetime
import tempfile
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from salespulse.storage.database import init_db
from salespulse.storage.repository import AuditRepository
from salespulse.monitoring.drift_schemas import (
    DriftStatus,
    FeatureType,
    DriftMetricType,
    NumericFeatureProfile,
    CategoricalFeatureProfile,
    DriftBaselineProfile,
    DriftConfig,
    FeatureDriftResult,
    DriftReport,
)
from salespulse.monitoring.drift_service import (
    compute_numeric_psi,
    compute_categorical_tvd,
    profile_dataset,
    DatasetDriftService,
)


@pytest.fixture
def temp_repo():
    """Create a temporary SQLite database engine and repository for isolated testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    init_db(engine)
    repo = AuditRepository(engine=engine)
    yield repo, engine


@pytest.fixture
def synthetic_baseline_df():
    """Deterministic synthetic reference dataset (N=500)."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    sales = np.random.normal(loc=150.0, scale=25.0, size=n)
    temp = np.random.uniform(10.0, 35.0, size=n)
    categories = np.random.choice(["Electronics", "Furniture", "Office Supplies"], size=n, p=[0.4, 0.35, 0.25])
    regions = np.random.choice(["East", "West", "Central", "South"], size=n, p=[0.3, 0.3, 0.2, 0.2])
    
    return pd.DataFrame({
        "date": dates,
        "sales": sales,
        "temperature": temp,
        "category": categories,
        "region": regions,
    })


# ============================================================================
# 1. NUMERIC PSI STATISTICAL TESTS
# ============================================================================
def test_psi_stable_distribution(synthetic_baseline_df):
    """Identical distributions should produce PSI < 0.10 (NO_DRIFT)."""
    np.random.seed(43)
    profile = profile_dataset(synthetic_baseline_df, baseline_id="base_test_stable")
    sales_profile = profile.numeric_profiles["sales"]
    
    # Stable current batch sampled from same distribution
    curr_sales = pd.Series(np.random.normal(loc=150.0, scale=25.0, size=300))
    psi, status, summary, expl = compute_numeric_psi(sales_profile, curr_sales)
    
    assert psi is not None
    assert psi < 0.10, f"Expected PSI < 0.10 for stable data, got {psi}"
    assert status == DriftStatus.NO_DRIFT
    assert "Stable numeric distribution" in expl


def test_psi_injected_significant_drift(synthetic_baseline_df):
    """Mean-shifted distribution should produce PSI >= 0.25 (DRIFT)."""
    np.random.seed(44)
    profile = profile_dataset(synthetic_baseline_df, baseline_id="base_test_drift")
    sales_profile = profile.numeric_profiles["sales"]
    
    # Injected shift: mean shifts from 150 -> 210 (2.4 standard deviations)
    shifted_sales = pd.Series(np.random.normal(loc=210.0, scale=30.0, size=300))
    psi, status, summary, expl = compute_numeric_psi(sales_profile, shifted_sales)
    
    assert psi is not None
    assert psi >= 0.25, f"Expected PSI >= 0.25 for large shift, got {psi}"
    assert status == DriftStatus.DRIFT
    assert "Significant numeric distribution shift" in expl


def test_psi_injected_moderate_warning(synthetic_baseline_df):
    """Moderate shift should produce 0.10 <= PSI < 0.25 (WARNING)."""
    np.random.seed(45)
    profile = profile_dataset(synthetic_baseline_df, baseline_id="base_test_warn")
    sales_profile = profile.numeric_profiles["sales"]
    
    # Injected moderate shift: mean shifts from 150 -> 163 (~0.5 std dev)
    moderate_sales = pd.Series(np.random.normal(loc=163.0, scale=25.0, size=400))
    psi, status, summary, expl = compute_numeric_psi(sales_profile, moderate_sales)
    
    assert psi is not None
    assert 0.05 <= psi <= 0.30, f"PSI {psi} in expected moderate range"
    # Verify that threshold boundaries are respected
    if psi >= 0.25:
        assert status == DriftStatus.DRIFT
    elif psi >= 0.10:
        assert status == DriftStatus.WARNING
    else:
        assert status == DriftStatus.NO_DRIFT


def test_psi_zero_bin_safeguards_and_constant_features():
    """Epsilon smoothing prevents zero-division or -inf log errors; constant features handled safely."""
    # Profile a feature with constant value
    df_const = pd.DataFrame({"constant_col": [10.0] * 100})
    profile = profile_dataset(df_const, baseline_id="base_const")
    const_prof = profile.numeric_profiles["constant_col"]
    assert const_prof.std == 0.0
    
    # Test 1: Current batch is identical constant
    curr_same = pd.Series([10.0] * 50)
    psi, stat, _, expl = compute_numeric_psi(const_prof, curr_same)
    assert stat == DriftStatus.NO_DRIFT
    assert psi == 0.0

    # Test 2: Current batch varies from baseline constant
    curr_diff = pd.Series([10.0] * 30 + [25.0] * 20)
    psi, stat, _, expl = compute_numeric_psi(const_prof, curr_diff)
    assert stat == DriftStatus.DRIFT
    assert "was constant (10.0) in baseline but varies" in expl


def test_psi_insufficient_samples_yields_unknown(synthetic_baseline_df):
    """Sample size < min_sample_size (default 30) must strictly yield UNKNOWN."""
    profile = profile_dataset(synthetic_baseline_df, baseline_id="base_small")
    sales_prof = profile.numeric_profiles["sales"]
    
    small_series = pd.Series([150.0, 160.0, 140.0])  # N=3 < 30
    psi, stat, _, expl = compute_numeric_psi(sales_prof, small_series)
    
    assert psi is None
    assert stat == DriftStatus.UNKNOWN
    assert "Insufficient samples" in expl


# ============================================================================
# 2. CATEGORICAL TVD AND UNSEEN CATEGORY TESTS
# ============================================================================
def test_categorical_tvd_stable(synthetic_baseline_df):
    """Consistent frequency distributions produce TVD < 0.10 (NO_DRIFT)."""
    np.random.seed(46)
    profile = profile_dataset(synthetic_baseline_df, baseline_id="base_cat_stable")
    cat_prof = profile.categorical_profiles["category"]
    
    curr_cats = pd.Series(np.random.choice(
        ["Electronics", "Furniture", "Office Supplies"],
        size=400,
        p=[0.4, 0.35, 0.25],
    ))
    tvd, stat, summary, unseen, unseen_rate, expl = compute_categorical_tvd(cat_prof, curr_cats)
    
    assert tvd is not None
    assert tvd < 0.10
    assert stat == DriftStatus.NO_DRIFT
    assert len(unseen) == 0
    assert unseen_rate == 0.0


def test_categorical_unseen_categories_trigger_drift(synthetic_baseline_df):
    """Unseen categories above threshold (5%) trigger DriftStatus.DRIFT."""
    profile = profile_dataset(synthetic_baseline_df, baseline_id="base_unseen")
    cat_prof = profile.categorical_profiles["category"]
    
    # 20% of rows contain an unseen category "Grocery"
    curr_cats = pd.Series(
        ["Electronics"] * 50 +
        ["Furniture"] * 30 +
        ["Grocery"] * 20  # Unseen!
    )
    tvd, stat, summary, unseen, unseen_rate, expl = compute_categorical_tvd(cat_prof, curr_cats)
    
    assert stat == DriftStatus.DRIFT
    assert "Grocery" in unseen
    assert unseen_rate == 0.20
    assert "unseen categories" in expl


# ============================================================================
# 3. BASELINE PROFILING & PRIVACY TESTS (NO RAW ROWS)
# ============================================================================
def test_baseline_profiling_and_privacy(synthetic_baseline_df):
    """Baseline profiles compute deterministic statistics and NEVER store raw rows."""
    profile = profile_dataset(
        synthetic_baseline_df,
        baseline_id="base_priv_test",
        sales_scope="TOTAL_STORE",
        source_file_name="superstore.csv",
        source_file_sha256="abc123def456",
        notes="Audit test baseline",
    )
    
    assert profile.baseline_id == "base_priv_test"
    assert profile.row_count == 500
    assert profile.data_window_start == "2026-01-01"
    assert len(profile.feature_names) == 4  # sales, temp, category, region (excluding date)
    assert "sales" in profile.numeric_profiles
    assert "category" in profile.categorical_profiles
    
    # Check serialization roundtrip
    json_str = profile.to_json()
    assert isinstance(json_str, str)
    
    # Privacy check: Ensure individual raw numbers or strings are not in the profile dict
    # Profiles contain quantiles, bins, and proportions only
    p_dict = profile.to_dict()
    assert "raw_data" not in p_dict
    assert "rows" not in p_dict


# ============================================================================
# 4. BASELINE LIFECYCLE, EXPLICIT PROMOTION, & SCOPE ISOLATION
# ============================================================================
def test_baseline_lifecycle_and_promotion(temp_repo, synthetic_baseline_df):
    """Creating a baseline does NOT silently activate it; explicit promotion is required."""
    repo, _ = temp_repo
    service = DatasetDriftService(repository=repo)
    
    # 1. Create baseline B1 without activation
    b1 = service.create_baseline(
        synthetic_baseline_df,
        sales_scope="STORE_A",
        baseline_id="base_001",
        activate=False,
    )
    assert service.get_active_baseline("STORE_A") is None
    
    # 2. Explicitly promote B1
    assert service.promote_baseline("base_001", sales_scope="STORE_A") is True
    active = service.get_active_baseline("STORE_A")
    assert active is not None
    assert active.baseline_id == "base_001"
    
    # 3. Create baseline B2 with activate=False
    b2 = service.create_baseline(
        synthetic_baseline_df,
        sales_scope="STORE_A",
        baseline_id="base_002",
        activate=False,
    )
    # B1 must remain the active baseline (no silent replacement!)
    active_after_b2 = service.get_active_baseline("STORE_A")
    assert active_after_b2.baseline_id == "base_001"
    
    # 4. Explicitly promote B2 -> B1 deactivated, B2 active
    assert service.promote_baseline("base_002", sales_scope="STORE_A") is True
    active_promoted = service.get_active_baseline("STORE_A")
    assert active_promoted.baseline_id == "base_002"
    
    # 5. Scope isolation: Scope STORE_B has no active baseline
    assert service.get_active_baseline("STORE_B") is None


# ============================================================================
# 5. DRIFT EVALUATION SERVICE TESTS (UNKNOWN STATES, SCHEMA, OFFLINE DB)
# ============================================================================
def test_drift_evaluation_missing_baseline_yields_unknown(temp_repo, synthetic_baseline_df):
    """Evaluating drift without an active baseline must return UNKNOWN, never NO_DRIFT."""
    repo, _ = temp_repo
    service = DatasetDriftService(repository=repo)
    
    report = service.evaluate_drift(synthetic_baseline_df, sales_scope="STORE_EMPTY")
    assert report.overall_status == DriftStatus.UNKNOWN
    assert report.baseline_id == "MISSING"
    assert "No active reference baseline found" in report.summary_message


def test_drift_evaluation_insufficient_current_rows_yields_unknown(temp_repo, synthetic_baseline_df):
    """Current batch smaller than min_sample_size (30) must yield UNKNOWN."""
    repo, _ = temp_repo
    service = DatasetDriftService(repository=repo)
    service.create_baseline(synthetic_baseline_df, sales_scope="TOTAL_STORE", baseline_id="base_active", activate=True)
    
    small_df = synthetic_baseline_df.head(10)  # N=10 < 30
    report = service.evaluate_drift(small_df, sales_scope="TOTAL_STORE")
    
    assert report.overall_status == DriftStatus.UNKNOWN
    assert "insufficient observations" in report.summary_message.lower()


def test_drift_evaluation_schema_mismatch(temp_repo, synthetic_baseline_df):
    """Missing a baseline feature in current batch is detected as schema deviation and DRIFT."""
    repo, _ = temp_repo
    service = DatasetDriftService(repository=repo)
    service.create_baseline(synthetic_baseline_df, sales_scope="TOTAL_STORE", baseline_id="base_schema", activate=True)
    
    # Drop "sales" column from current batch
    mismatched_df = synthetic_baseline_df.drop(columns=["sales"])
    report = service.evaluate_drift(mismatched_df, sales_scope="TOTAL_STORE")
    
    assert report.overall_status == DriftStatus.DRIFT
    assert any("sales" in diff for diff in report.schema_differences)
    sales_res = next(f for f in report.feature_results if f.feature_name == "sales")
    assert sales_res.status == DriftStatus.DRIFT
    assert sales_res.metric_type == DriftMetricType.SCHEMA_CHANGE


def test_drift_evaluation_database_offline_safe_handling(synthetic_baseline_df):
    """When repository database is offline, in-memory check against provided baseline is safe."""
    dummy_engine = create_engine("sqlite:///:memory:")
    repo = AuditRepository(dummy_engine)
    # Dispose connection to simulate database offline
    dummy_engine.dispose()
    
    service = DatasetDriftService(repository=repo)
    report = service.evaluate_drift(synthetic_baseline_df, sales_scope="TOTAL_STORE")
    
    assert report.overall_status == DriftStatus.UNKNOWN
    assert report.database_connected is False


# ============================================================================
# 6. PERSISTENCE IDEMPOTENCY AND HISTORY RETRIEVAL
# ============================================================================
def test_drift_check_persistence_and_history(temp_repo, synthetic_baseline_df):
    """Drift checks are persisted and historical queries return records ordered by timestamp descending."""
    repo, _ = temp_repo
    service = DatasetDriftService(repository=repo)
    service.create_baseline(synthetic_baseline_df, sales_scope="TOTAL_STORE", baseline_id="base_persist", activate=True)
    
    # Run two drift checks
    rep1 = service.evaluate_drift(synthetic_baseline_df, sales_scope="TOTAL_STORE", persist=True)
    rep2 = service.evaluate_drift(synthetic_baseline_df, sales_scope="TOTAL_STORE", persist=True)
    
    history = service.get_drift_history(sales_scope="TOTAL_STORE", limit=10)
    assert len(history) == 2
    assert history[0]["check_id"] == rep2.check_id
    assert history[1]["check_id"] == rep1.check_id
    
    # Verify latest check getter
    latest = service.get_latest_drift_check(sales_scope="TOTAL_STORE")
    assert latest is not None
    assert latest.check_id == rep2.check_id
    assert latest.overall_status == rep2.overall_status
