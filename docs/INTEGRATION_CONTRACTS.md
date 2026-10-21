# SalesPulse AI — Shared Data & Integration Contracts Specification

**Version:** 3.0.0 (Phase 4C Production Stabilization)
**Status:** Active Canonical Specification
**Last Updated:** October 2026
**Audience:** ML Engineers, Data Engineers, Platform Maintainers, Streamlit Operators

---

## 1. Executive Summary

This document establishes the canonical data contracts, schemas, status enumerations, accounting definitions, and backward-compatibility rules across SalesPulse AI v3.0. It defines the formal integration interface connecting data ingestion, storage models, monitoring services, business logic, and the Streamlit operator dashboard.

Adherence to these contracts guarantees that:
1. All dashboard workspaces consume predictable, typed schemas without runtime `AttributeError` or missing-column crashes.
2. Historical demonstration datasets (e.g., UCI Online Retail II) are strictly distinguished from live production telemetry and user-supplied files.
3. Accounting metrics (gross sales, returns, net sales) follow deterministic, auditable rules.
4. Error, empty, and unavailable states are rendered transparently rather than disguised with misleading default metrics.

---

## 2. Canonical Data Models & Schemas

### 2.1 Daily Time-Series Sales Contract (`daily_df`)
Applies to continuous daily sales tables (`online_retail_ii_daily.csv`, reindexed Superstore, and user uploads).

| Field Name | Type | Nullable | Description / Business Rule |
| :--- | :--- | :--- | :--- |
| `date` | `datetime64[ns]` / `Date` | No | Calendar date (`YYYY-MM-DD`). Monotonically increasing continuous calendar sequence. |
| `sales` | `float64` | No | **Net commercial sales revenue** (`gross_sales - returns`). Primary target for ML models. |
| `gross_sales` | `float64` | Yes | Total gross revenue from valid dispatched transactions (`Quantity > 0`, `Price > 0`). |
| `returns` | `float64` | Yes | Realized returns and order cancellations (`Quantity < 0` or `Invoice` starting with `'C'`). |
| `units_sold` | `int64` | Yes | Gross physical unit count dispatched in valid sales. |
| `units_returned` | `int64` | Yes | Gross physical units returned or cancelled. |
| `order_count` | `int64` | Yes | Distinct valid sales invoice count for the calendar day. |
| `return_count` | `int64` | Yes | Distinct cancellation invoice count for the calendar day. |
| `avg_order_value` | `float64` | Yes | `gross_sales / order_count` (or `0.0` when `order_count == 0`). |
| `is_observed_trading_day` | `bool` | Yes | `True` if trading transactions occurred; `False` if zero-filled non-trading calendar day. |

---

### 2.2 Canonical Product-Sales Contract (`top_products_df`)
Module: `salespulse.data.product_aggregation`
Artifact: `data/processed/online_retail_ii_top_products.csv`

| Field Name | Type | Nullable | Canonical Specification |
| :--- | :--- | :--- | :--- |
| `product_id` | `str` | No | Unique SKU / stock identifier (e.g., `StockCode`). |
| `description` | `str` | No | Product title. If missing in raw data, defaults to `'Unknown SKU (<product_id>)'`. |
| `units_sold` | `int` | No | Gross physical units dispatched across valid non-cancelled orders. |
| `gross_sales` | `float` | No | $\sum (\text{Quantity} \times \text{UnitPrice})$ for valid positive orders. |
| `returns` | `float` | No | $\sum (\lvert\text{Quantity}\rvert \times \text{UnitPrice})$ for cancelled/returned lines. |
| `net_sales` | `float` | No | $\text{Gross Sales} - \text{Returns}$. Primary ranking metric for revenue leadership. |
| `order_count` | `int` | No | Number of distinct non-cancellation invoices containing the SKU. |

**Chart Consumer Contract:**
UI bar charts rank products by `net_sales` descending. If `net_sales` is absent in legacy feeds, fallback order is `NetSales` $\to$ `gross_sales` $\to$ first numeric column. Product name falls back from `description` $\to$ `Description` $\to$ `product_id`.

---

### 2.3 Geographic Market Contract (`countries_df`)
Artifact: `data/processed/online_retail_ii_countries.csv`

| Field Name | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| `country` | `str` | No | Country name or sovereign trading territory. |
| `order_count` | `int` | No | Count of distinct non-cancellation orders originating from territory. |
| `gross_sales` | `float` | No | Total gross sales value dispatched to territory. |
| `net_sales` | `float` | No | Gross sales minus returns for territory. |

---

## 3. Freshness & Monitoring Service Contracts

### 3.1 `DataFreshnessReport`
Dataclass: `salespulse.monitoring.schemas.DataFreshnessReport`

| Field Name | Type | Default | Contract Role |
| :--- | :--- | :--- | :--- |
| `freshness_state` | `FreshnessState` | (required) | **Canonical state enum** (`FRESH`, `STALE`, `EMPTY_DATASET`, `MISSING_DATA`, `UNKNOWN`). |
| `state` | `FreshnessState` | (property) | **Compatibility alias** returning `self.freshness_state`. Prevents consumer crashes. |
| `latest_observation_date` | `Optional[str]` | `None` | ISO date string (`YYYY-MM-DD`) of latest observation in evaluated scope. |
| `staleness_days` | `Optional[int]` | `None` | Calendar days between `latest_observation_date` and evaluation reference date. |
| `staleness_threshold_days`| `int` | `30` | Configured SLA threshold days before dataset is flagged `STALE`. |
| `cadence` | `str` | `"DAILY"` | Evaluation cadence string (`"DAILY"`, `"HOURLY"`, `"WEEKLY"`). |
| `is_stale` | `bool` | `False` | Boolean flag indicating whether `staleness_days > staleness_threshold_days`. |
| `zero_sales_dates_count` | `int` | `0` | Verified zero-sales dates in series. |
| `missing_calendar_dates_count` | `int` | `0` | Absent calendar dates in span. |
| `reference_date` | `str` | `""` | Evaluation reference date (`YYYY-MM-DD`). |
| `assessment` | `str` | `""` | Natural language evaluation rationale. |

---

### 3.2 `DataMonitoringRecord` (Storage Model)
SQLAlchemy Model: `salespulse.storage.models.DataMonitoringRecord`
Database Table: `data_monitoring_records`

| Column / Property | Storage Type | Source / Derivation |
| :--- | :--- | :--- |
| `check_id` | `String(64)` | Unique evaluation identifier (UUID or deterministic hash). |
| `checked_at` | `DateTime` | UTC timestamp when evaluation ran. |
| `sales_scope` | `String(50)` | Evaluated sales scope (default: `"TOTAL_STORE"`). |
| `status` | `String(20)` | Overall verdict status (`"HEALTHY"`, `"WARNING"`, `"CRITICAL"`, `"UNKNOWN"`). |
| `freshness_state` | `@property` $\to$ `str` | **Canonical adapter property**: Inspects `evidence_json` for freshness state, falls back to staleness vs threshold. Preserves `UNKNOWN`, `STALE`, `CRITICAL`, and `FRESH`. Never defaults to `HEALTHY` blindly. |
| `latest_ingestion_outcome`| `@property` $\to$ `str` | **Canonical adapter property**: Returns `last_ingestion_status` or parses outcome from `evidence_json`. Defaults to `"UNKNOWN"`. |
| `latest_observation_date` | `Date` | Latest date observed in storage actuals. |
| `staleness_days` | `Integer` | Computed staleness days at evaluation time. |
| `staleness_threshold_days`| `Integer` | Configured SLA limit at check time. |
| `last_successful_ingestion_at`| `DateTime` | Timestamp of last verified ingestion attempt. |
| `last_ingestion_attempt_at` | `DateTime` | Timestamp of most recent ingestion attempt. |
| `last_ingestion_status` | `String(20)` | Status of latest ingestion (`"PASSED"`, `"WARNING"`, `"FAILED"`, `"NO_ATTEMPTS"`). |
| `consecutive_failures` | `Integer` | Consecutive failed validation runs. |
| `error_count` | `Integer` | Fatal contract errors in latest attempt. |
| `warning_count` | `Integer` | Non-fatal warnings in latest attempt. |
| `summary_message` | `Text` | Executive evaluation summary string. |
| `evidence_json` | `Text` | Strict RFC 8259 JSON payload containing full verdict report. |

---

## 4. Status Enums & State Transition Rules

### 4.1 Enumeration Definitions

```python
class FreshnessState(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    EMPTY_DATASET = "EMPTY_DATASET"
    MISSING_DATA = "MISSING_DATA"
    UNKNOWN = "UNKNOWN"

class MonitoringStatus(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

class IngestionOutcome(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    NO_ATTEMPTS = "NO_ATTEMPTS"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"

class CadenceType(str, Enum):
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
```

### 4.2 State Precedence & Transition Hierarchy

1. **Storage / Telemetry Unreachable:** $\to$ `UNKNOWN`
   If SQLite connection fails or telemetry is missing, status is strictly `UNKNOWN`. Do not synthesize passing results.
2. **Missing Ingestion History:** $\to$ `UNKNOWN`
   If no observation baseline exists, status is `UNKNOWN` with actionable remediation to ingest data.
3. **Critical Failures:** $\to$ `CRITICAL`
   Triggered if:
   - `consecutive_failures >= 3`
   - Latest ingestion validation `FAILED`
   - `staleness_days > (threshold * 1.5)`
4. **Warnings:** $\to$ `WARNING`
   Triggered if:
   - `freshness.is_stale == True` (`threshold < staleness_days <= threshold * 1.5`)
   - Latest ingestion passed with non-fatal review flags (`outcome == WARNING`)
   - `recent_warnings_count > 0`
5. **Healthy:** $\to$ `HEALTHY`
   Strictly when all conditions hold:
   - `freshness_state == FRESH`
   - `staleness_days <= threshold`
   - Ingestion `outcome` in (`PASSED`, `NO_ATTEMPTS`)
   - `consecutive_failures == 0`
   - `recent_warnings_count == 0`

---

## 5. Dataset Identity, Semantics & Currency

| Dataset Identity | Data Mode | Currency | Grain | Coverage Span | Trading Days | Status Semantics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UCI Online Retail II** | `DEMO DATA` | `£` (GBP) | Daily (`D`) | 2009-12-01 to 2011-12-09 | 604 observed (739 calendar) | **Historical Demo.** Finished in Dec 2011. Never describe as 2026 live sales. |
| **US Superstore** | `DEMO DATA` | `$` (USD) | Daily (`D`) | 2014-01-03 to 2017-12-30 | 1,458 calendar days | **Historical Demo.** US multi-category retail sample. |
| **User Upload** | `USER DATA` | User/`$` | Daily (`D`) | Arbitrary user span | Dynamic | **Active Session Data.** Validated against `salespulse.data.ingestion`. |

---

## 6. Error, Empty, and Unavailable State Policy

In accordance with Section 1 Safety Rules and Section 5 Operator UX:

1. **No Data State (`NO_DATA`):**
   Render clear info card instructing operator to select a demo dataset or upload a file. Never render zeroes or blank charts.
2. **Unavailable Data / Storage Offline (`DATA_UNAVAILABLE`):**
   Render alert with actionable guidance (e.g., "Run database initialization or check SQLite permissions").
3. **Empty Catalog Breakdown:**
   Render `"Product-level catalog breakdown is not available for the active dataset scope."`
4. **Insufficient Anomaly History (< 14 days):**
   Render explicit explanatory state explaining minimum sample requirement. Do not imply series is normal.
5. **Future Forecast Steps:**
   Future forecast steps with pending actuals are rendered as **Pending Evaluation** (`-`), not 0.0 error.

---

## 7. Backward-Compatibility Strategy

1. **`DataMonitoringRecord.freshness_state`:**
   Implemented as a model property returning canonical string. Safe against legacy records without `evidence_json`.
2. **`DataFreshnessReport.state`:**
   Implemented as a property returning `self.freshness_state`. Fully supports older code expecting `.state`.
3. **Chart Column Normalization:**
   Chart rendering functions dynamically inspect column candidate lists (`['net_sales', 'NetSales', 'gross_sales']`), ensuring resilience against schema evolution.
