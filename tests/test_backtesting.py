"""
Unit tests for Chronological Rolling-Origin Backtesting & Regression Tests
"""

import pytest
import pandas as pd
import numpy as np

from salespulse.evaluation.backtesting import RollingOriginBacktester
from salespulse.models.baseline import SeasonalNaiveModel
from salespulse.models.daily_forecaster import DailySalesForecaster
from salespulse.features.temporal import ENGINEERED_FEATURES


class DummyModel:
    """Dummy model returning a constant value for testing pipeline wiring."""
    def __init__(self, const_val=100.0):
        self.const_val = const_val

    def predict(self, X):
        return np.full(len(X), self.const_val)


def test_rolling_origin_origins_generation():
    """Verify chronological forecast origins strictly advance forward in time."""
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    sales = np.random.uniform(50, 150, size=500)
    df = pd.DataFrame({"date": dates, "sales": sales})

    tester = RollingOriginBacktester(
        daily_df=df,
        horizons=[7, 14, 30],
        min_train_days=300,
        step_days=14,
        eval_window_days=100,
    )

    origins = tester.generate_origins()
    assert len(origins) > 1
    # Check that origins are strictly increasing
    assert origins == sorted(origins)
    # Check bounds
    assert origins[0] >= 300
    assert origins[-1] + 30 < len(df)


def test_backtesting_identical_evaluation():
    """Verify all candidate models are evaluated on identical forecast origins and actual targets."""
    dates = pd.date_range("2021-01-01", periods=400, freq="D")
    sales = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] * 57 + [10.0]  # 400 days
    df = pd.DataFrame({"date": dates, "sales": sales})

    tester = RollingOriginBacktester(
        daily_df=df,
        horizons=[7, 14, 30],
        min_train_days=250,
        step_days=14,
        eval_window_days=60,
    )

    baseline = SeasonalNaiveModel(seasonality_period=7)
    dummy_forecaster = DailySalesForecaster(
        model_type="Dummy",
        model_instance=DummyModel(const_val=40.0),
        residual_std=15.0,
    )

    results = tester.run_backtest(
        models_or_factories={
            "SeasonalNaive": baseline,
            "DummyForecaster": dummy_forecaster,
        }
    )

    assert results["folds_evaluated"] > 0
    assert "SeasonalNaive" in results["summary_metrics"]
    assert "DummyForecaster" in results["summary_metrics"]

    # Both models evaluated on 7, 14, and 30 day horizons
    for h in [7, 14, 30]:
        h_key = f"{h}_day"
        assert h_key in results["summary_metrics"]["SeasonalNaive"]
        assert h_key in results["summary_metrics"]["DummyForecaster"]
        sn_m = results["summary_metrics"]["SeasonalNaive"][h_key]
        dum_m = results["summary_metrics"]["DummyForecaster"][h_key]
        # Count of evaluation points must be identical!
        assert sn_m["total_evaluations"] == dum_m["total_evaluations"]
        assert sn_m["unique_target_dates"] == dum_m["unique_target_dates"]


def test_backtesting_insufficient_history():
    """Verify ValueError when total data is shorter than min_train_days + max_horizon."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "sales": [10.0] * 100})

    with pytest.raises(ValueError, match="Insufficient data length"):
        RollingOriginBacktester(
            daily_df=df,
            horizons=[7, 14, 30],
            min_train_days=200,  # exceeds 100 days
        )


def test_static_model_rejects_overlapping_origins():
    """REGRESSION TEST: Verify static models with training cutoff at/after earliest origin are rejected."""
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    df = pd.DataFrame({"date": dates, "sales": [50.0] * 300})

    tester = RollingOriginBacktester(
        daily_df=df,
        horizons=[7, 14, 30],
        min_train_days=100,
        step_days=14,
        eval_window_days=60,
    )
    origins = tester.generate_origins()
    earliest_origin = origins[0]
    earliest_date = df.loc[earliest_origin, "date"]

    # Model was trained up to a date AFTER the earliest origin (leakage!)
    bad_model = DailySalesForecaster(
        model_type="LeakyModel",
        model_instance=DummyModel(),
        training_end_date=earliest_date + pd.Timedelta(days=5),
        training_cutoff_idx=earliest_origin + 5,
    )

    with pytest.raises(ValueError, match="LEAKAGE ERROR for static model"):
        tester.run_backtest({"LeakyModel": bad_model})


def test_dynamic_fold_refitting_isolation():
    """REGRESSION TEST: Verify dynamic walk-forward factory receives strictly data up to origin."""
    dates = pd.date_range("2022-01-01", periods=300, freq="D")
    base_sales = np.arange(1.0, 301.0)
    df = pd.DataFrame({"date": dates, "sales": base_sales})

    passed_history_lengths = []

    def mock_factory(history_df: pd.DataFrame):
        passed_history_lengths.append(len(history_df))
        last_val = history_df["sales"].iloc[-1]
        # Return dummy forecaster predicting last observed value
        return DailySalesForecaster("Mock", DummyModel(const_val=last_val), residual_std=5.0)

    tester = RollingOriginBacktester(
        daily_df=df,
        horizons=[7, 14, 30],
        min_train_days=100,
        step_days=14,
        eval_window_days=60,
    )
    origins = tester.generate_origins()

    results = tester.run_backtest({"DynamicModel": mock_factory})

    # Factory must be called once per origin
    assert len(passed_history_lengths) == len(origins)
    # Each fold's history length must equal origin_idx + 1 exactly!
    for expected_origin, actual_len in zip(origins, passed_history_lengths):
        assert actual_len == expected_origin + 1


def test_macro_vs_micro_aggregation():
    """REGRESSION TEST: Verify dual micro (pooled) and macro (mean across folds) metrics."""
    dates = pd.date_range("2021-01-01", periods=350, freq="D")
    # Alternating low/high demand
    sales = [10.0, 500.0] * 175
    df = pd.DataFrame({"date": dates, "sales": sales})

    tester = RollingOriginBacktester(
        daily_df=df,
        horizons=[7, 14, 30],
        min_train_days=150,
        step_days=14,
        eval_window_days=80,
    )

    baseline = SeasonalNaiveModel(seasonality_period=7)
    results = tester.run_backtest({"Baseline": baseline})

    m_7 = results["summary_metrics"]["Baseline"]["7_day"]
    assert "micro" in m_7
    assert "macro" in m_7

    # Micro WAPE and Macro WAPE are both calculated
    assert isinstance(m_7["micro"]["wape"], float)
    assert isinstance(m_7["macro"]["wape_mean"], float)
    assert isinstance(m_7["macro"]["wape_std"], float)

    # 30-day horizon with 14-day step must have overlap ratio > 1.0
    m_30 = results["summary_metrics"]["Baseline"]["30_day"]
    assert m_30["overlap_ratio"] > 1.0
    assert m_30["total_evaluations"] > m_30["unique_target_dates"]


def test_interval_coverage_reporting():
    """REGRESSION TEST: Verify interval coverage and average interval width are reported."""
    dates = pd.date_range("2021-01-01", periods=300, freq="D")
    df = pd.DataFrame({"date": dates, "sales": [100.0] * 300})

    tester = RollingOriginBacktester(
        daily_df=df,
        horizons=[7, 14, 30],
        min_train_days=120,
        step_days=14,
        eval_window_days=60,
    )

    forecaster = DailySalesForecaster(
        model_type="Const",
        model_instance=DummyModel(const_val=100.0),
        residual_std=10.0,
    )
    results = tester.run_backtest({"Const": forecaster})

    for h in [7, 14, 30]:
        h_m = results["summary_metrics"]["Const"][f"{h}_day"]
        assert "mean_interval_width" in h_m
        assert h_m["mean_interval_width"] > 0.0
        assert "coverage" in h_m["micro"]
        assert h_m["micro"]["coverage"] == 1.0  # Perfect prediction of constant 100.0
