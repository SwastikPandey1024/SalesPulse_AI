# 📑 Software Requirements Specification (SRS) - SalesPulse AI v2.0

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) document details the functional, non-functional, architectural, and interface requirements for **SalesPulse AI Version 2.0 (v2.0)**. It serves as the primary technical specification for data scientists, machine learning engineers, and application developers maintaining or deploying the platform.

### 1.2 Scope
SalesPulse AI v2.0 is an interactive ML analytics platform designed for retail time-series sales forecasting, business scenario simulation, exploratory data analytics, and multi-model benchmarking.

### 1.3 Definitions, Acronyms, and Abbreviations
- **MAE:** Mean Absolute Error
- **XGBoost:** eXtreme Gradient Boosting Regressor
- **RF:** RandomForest Regressor
- **EDA:** Exploratory Data Analytics
- **Lag Feature:** Historical observation from prior time steps ($t-1, t-7, t-14, t-30$)
- **Rolling Metric:** Moving statistics calculated over a rolling window (e.g., 7-day mean, 7-day standard deviation)
- **Prediction Interval:** Range of values within which future observations are expected to fall with a 95% probability

---

## 2. Overall Description

### 2.1 Product Perspective
SalesPulse AI operates as a self-contained, web-based analytics dashboard deployed on **Streamlit Community Cloud**, connecting serialized Machine Learning model pipelines (`model.pkl`, `rf_model.pkl`) to historical transaction data (`Sample - Superstore.csv`).

### 2.2 Product Functions
1. **Real-Time Single-Day Sales Inference:** User inputs temporal and rolling metrics to generate single-day sales predictions accompanied by actionable business recommendations.
2. **Multi-Horizon Forecasting Engine:** Generates 7-day, 14-day, and 30-day iterative predictions complete with 95% confidence prediction bands.
3. **Interactive Business Scenario Simulator:** Simulates sales impact under custom promotional lift (%), price adjustments (%), and seasonal multipliers.
4. **Superstore Exploratory Data Analytics (EDA):** Interactive filters and charts analyzing category revenue share, regional performance, and transaction KPIs.
5. **Model Benchmarking & Feature Explainability:** Comparative matrix evaluating XGBoost vs. RandomForest vs. Ensemble, featuring feature importance visualizations.
6. **Smart Batch Data Processing Engine:** Parses uploaded CSV files, automatically validates features, computes predictions in parallel, and exports downloadable CSVs.

### 2.3 User Classes and Characteristics
- **Small & Medium Business (SMB) Executives:** Require clear, high-level forecasts, revenue impact simulations, and strategic recommendations without technical complexity.
- **Sales & Operations Managers:** Require 7/14/30-day inventory and staffing guidance.
- **Data Analysts & ML Engineers:** Require model explainability, feature importance rankings, and multi-architecture benchmarking.

---

## 3. Specific System Requirements

### 3.1 Functional Requirements

#### FR-1: Single-Day Prediction Module
- **FR-1.1:** The system shall accept user inputs for Date, Yesterday's Sales, 7-Day Prior Sales, 14-Day Prior Sales, 30-Day Prior Sales, 7-Day Rolling Mean, 14-Day Rolling Mean, 7-Day Rolling Std Dev, and Trend Index.
- **FR-1.2:** The system shall validate all numeric inputs to ensure non-negative values and absence of NaN or infinite floats.
- **FR-1.3:** The system shall compute sales predictions in under 2 seconds and display formatted currency output (`$XX.XX`).
- **FR-1.4:** The system shall dynamically generate textual business recommendations based on predicted sales growth or contraction relative to historical benchmarks.

#### FR-2: Multi-Horizon Forecasting Engine
- **FR-2.1:** The system shall support selectable forecast horizons of 7, 14, and 30 days.
- **FR-2.2:** The system shall perform auto-regressive iterative predictions, updating lag values and rolling metrics dynamically at each step.
- **FR-2.3:** The system shall calculate upper and lower 95% confidence interval bounds using rolling volatility scaling:
  $$y \pm 1.96 \cdot \max(10.0, \sigma_7) \cdot \sqrt{\frac{i}{2}}$$
- **FR-2.4:** The system shall render interactive Plotly line charts displaying the forecast line and shaded confidence bands.

#### FR-3: Business Scenario Simulator ("What-If" Engine)
- **FR-3.1:** The system shall provide sliders for Promotional Campaign Lift (0% to +50%), Price Adjustment (-20% to +20%), and Seasonal Multipliers (0.5x to 1.5x).
- **FR-3.2:** The system shall compute Base Sales, Simulated Sales, Revenue Impact ($), and Percentage Delta (%).
- **FR-3.3:** The system shall visualize scenario comparisons using comparative Plotly bar charts.

#### FR-4: Superstore Exploratory Data Analytics (EDA)
- **FR-4.1:** The system shall ingest `Sample - Superstore.csv` and parse `Order Date`, `Sales`, `Category`, `Region`, and `Segment`.
- **FR-4.2:** The system shall provide dynamic multi-select filters for Region, Category, and Segment.
- **FR-4.3:** The system shall update metrics for Total Revenue, Total Transactions, Average Order Value, and Top Category in real time upon filter change.

#### FR-5: Model Benchmarking & Interpretability
- **FR-5.1:** The system shall display a comparison matrix contrasting XGBoost Regressor, RandomForest Regressor, and Ensemble models.
- **FR-5.2:** The system shall extract and display feature importances for all 13 engineered features in a horizontal Plotly bar chart.

#### FR-6: Smart Batch Processing Engine
- **FR-6.1:** The system shall accept CSV file uploads containing multi-record batch data.
- **FR-6.2:** The system shall validate each row individually, logging success (`✅ Success`) or error (`❌ Error`) status without halting the batch run.
- **FR-6.3:** The system shall generate downloadable result CSV files.

---

### 3.2 Non-Functional Requirements

| ID | Category | Requirement Description |
| :--- | :--- | :--- |
| **NFR-1** | **Performance** | Single prediction latency shall be $< 500 \text{ ms}$; multi-day forecast latency shall be $< 2.0 \text{ s}$. |
| **NFR-2** | **Availability** | The system shall maintain $\ge 99.5\%$ uptime when hosted on Streamlit Cloud. |
| **NFR-3** | **Reliability** | The system shall execute startup pre-flight model schema validation to auto-purge incompatible models and prevent runtime crashes. |
| **NFR-4** | **Usability** | The web UI shall be responsive, mobile-friendly, and adhere to accessible color contrast guidelines. |
| **NFR-5** | **Maintainability** | All code shall follow PEP 8 standards with modular function organization and inline docstrings. |

---

## 4. Verification & Validation Matrix

| Requirement | Test Method | Acceptance Criteria |
| :--- | :--- | :--- |
| **FR-1 (Single Prediction)** | Automated Unit Test | Returns positive float with insights array for valid inputs. |
| **FR-2 (Forecast Horizon)** | Integration Test | Generates DataFrame with exact $N$ rows matching chosen horizon ($N \in \{7, 14, 30\}$). |
| **FR-3 (Scenario Simulator)** | Calculation Check | Output matches mathematical formula: $\text{Base} \cdot \text{Multiplier}$. |
| **NFR-3 (Reliability)** | Model Load Test | Gracefully handles missing or corrupted pickle files via XGBoost fallback. |
