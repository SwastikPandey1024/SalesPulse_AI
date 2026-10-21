# 📊 PRODUCT REQUIREMENT DOCUMENT (PRD)

## SalesPulse AI – Intelligent Sales Analytics Dashboard

- **Current Live Deployment:** [https://salespulseai.streamlit.app/](https://salespulseai.streamlit.app/)
- **Source Code Repository:** [https://github.com/SwastikPandey1024/SalesPulse_AI](https://github.com/SwastikPandey1024/SalesPulse_AI)
- **Deployment Platform:** Streamlit Community Cloud (Entrypoint: `app.py`)

> [!NOTE]
> **Deployment Architecture Note:** The hosted deployment at [`https://salespulseai.streamlit.app/`](https://salespulseai.streamlit.app/) serves as a live interactive demonstration of the current Streamlit dashboard application. It demonstrates the product capabilities outlined in this document using public reference datasets. Public availability of the hosted frontend does not constitute evidence of enterprise security hardening, multi-tenant isolation, or a publicly accessible FastAPI backend service.

---

## 🎯 Product Objective

To build an AI-powered web application that predicts future sales using historical trends and provides actionable business insights through an interactive dashboard.

---

## 👥 Target Users

* **Small & Medium Business Owners** – Need accurate sales forecasts for planning
* **Sales Managers** – Require trend analysis and performance metrics
* **Data Analysts** – Want interpretable ML predictions with explainability
* **E-commerce Operators** – Need real-time inventory and revenue planning

---

## 🚀 Core Features

### 1️⃣ Sales Prediction (Single Day)

* Predict next-day sales using XGBoost machine learning model
* Based on engineered lag features and rolling average metrics
* User provides:
  - Date of interest
  - Historical sales metrics (lag values, rolling averages)
  - Trend counter
* Output: Predicted sales value with confidence metrics

**Why It Matters:** Enables one-off "what-if" scenario testing without manual calculations

---

### 2️⃣ Trend Analysis (7-Day Forecast)

* Generate iterative 7-day sales forecast
* Visualize trends with line charts and comparative analysis
* Compare predicted values against historical averages
* Detect growth, decline, or stability patterns

**Why It Matters:** Helps businesses plan inventory and staffing in advance

---

### 3️⃣ Business Insights Engine

Automatically generate contextual business recommendations:

* **📈 Growth Detection** → "Strong growth expected - Scale inventory and staffing"
* **📉 Decline Detection** → "Sales decline expected - Run promotional activities"
* **➡️ Stability** → "Sales stable - Maintain current strategy"
* **⬆️ Above Average** → "+$X above 7-day average"
* **⬇️ Below Average** → "-$X below 7-day average"
* **⚡ Volatility Alerts** → "High volatility detected - Review marketing strategy"

**Why It Matters:** Non-technical users get actionable recommendations without data science knowledge

---

### 4️⃣ KPI Dashboard

Display critical metrics at a glance:

* Yesterday's Sales
* 7-day rolling average
* 14-day rolling average  
* 30-day rolling average
* Predicted sales value
* Percentage change indicators
* Trend indicators (↑ ↓ →)

**Why It Matters:** Executives can assess business health in seconds

---

### 5️⃣ Feature Importance Visualization

* Extract and visualize top influencing factors from ML model
* Show which features drive sales predictions
* Ranked importance chart
* Downloadable feature rankings

**Why It Matters:** Builds trust in model predictions and enables data-driven strategy

---

### 6️⃣ Bulk Prediction (Batch CSV Upload)

* Upload CSV file with multiple records
* Automatic feature validation
* Parallel prediction processing
* Download results as CSV
* Error handling per record (some failures won't block entire batch)

**Why It Matters:** Enables forecasting for entire product lines or multiple locations at once

---

### 7️⃣ Time-Based Feature Engineering

Automatic extraction of temporal features:

* **Date Components**: day, month, weekday, is_weekend, weekofyear
* **Lag Features**: Previous day (lag-1), 7-day, 14-day, 30-day sales
* **Rolling Metrics**: 7-day mean, 14-day mean, 7-day std deviation
* **Trend Counter**: Continuous trend indicator

**Why It Matters:** Captures seasonal patterns, day-of-week effects, and sales momentum

---

### 8️⃣ Model Explainability

* Display model architecture details
* Show training performance metrics (MAE, R² score)
* Feature importance rankings
* Model validation results

**Why It Matters:** Maintains user confidence and regulatory compliance

---

### 9️⃣ Performance Optimization

* **Cached Model Loading** – Model loaded once at app startup (not per prediction)
* **Efficient Data Processing** – Vectorized operations with NumPy/Pandas
* **Lazy Loading** – Feature importance computed only when needed

**Why It Matters:** Sub-2-second prediction times even under load

---

### 🔟 Error Resilience & Boot Diagnostics

* Boot diagnostics verify all dependencies before UI renders
* Safe model loading with detailed error messages
* File existence checks (model.pkl verification)
* Graceful degradation: fallback UI if model fails
* All errors visible to user (no silent crashes)

**Why It Matters:** Production-ready reliability for business-critical forecasting

---

## 🛠️ Technical Architecture

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit (UI framework) |
| **Backend** | Python 3.8+ |
| **ML Model** | XGBoost (Regression) |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Joblib |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git / GitHub |
| **Preprocessing** | Scikit-learn Pipeline |

### Model Architecture

```
Input Features (13) 
    ↓
Scikit-learn Preprocessing Pipeline
    ↓
XGBoost Regressor
    ↓
Predicted Sales Value
```

---

## 🧪 Functional Requirements

### User Input Validation

* All numeric inputs must be positive or zero
* Date inputs must be valid dates (not future dates for historical lags)
* No NaN or infinite values accepted
* Feature set must match training schema exactly (13 features)

### Model Operations

* Model must load successfully from `model.pkl` before app renders
* Model must validate feature count matches training features
* Predictions must return within 2 seconds
* All predictions must be positive values (business logic constraint)

### Output Requirements

* Predictions displayed with 2 decimal places
* Charts must render without errors
* CSV downloads must include headers and proper formatting
* Error messages must be clear and actionable

---

## ⚠️ Non-Functional Requirements

| Requirement | Target |
|------------|--------|
| **Performance** | Prediction response < 2 seconds |
| **Availability** | 99.5% uptime on Streamlit Cloud |
| **Reliability** | Zero silent crashes; all errors visible |
| **Usability** | No technical knowledge required; guidance provided |
| **Scalability** | Handle batch uploads up to 1,000 records |
| **Security** | No API keys exposed; safe file handling |

---

## 🔄 User Workflows

### Workflow 1: Single Day Prediction
```
User Input (Date + Metrics) 
  → Feature Engineering 
  → Model Prediction 
  → Display Result + Insights 
  → Visualization
```

### Workflow 2: 7-Day Forecast
```
User Input (Current Metrics)
  → Iterative Prediction Loop (7 times)
  → Update Lags Each Iteration
  → Display Forecast Table + Chart
  → Download CSV
```

### Workflow 3: Batch Processing
```
Upload CSV
  → Validate Columns
  → Process Each Row
  → Skip Invalid Rows (continue batch)
  → Display Results Table
  → Download Results CSV
```

---

## 📦 Deliverables

✅ **Deployed Streamlit Application**
- Live dashboard at Streamlit Cloud
- 4-tab interface with all features

✅ **Trained ML Model**
- `model.pkl` with XGBoost + preprocessing pipeline
- Feature metadata included

✅ **Source Code (GitHub)**
- `app.py` – Main application (~700 lines)
- `retrain_model.py` – Model training script
- Complete documentation

✅ **Documentation**
- README.md – Setup & usage guide
- PRD.md – This document
- BRD.md – Business requirements
- Inline code comments explaining validation logic

---

## 🎨 UI/UX Design

### Layout

**Header**
- App title: "📊 SalesPulse AI"
- Status indicator: ✅ Model Ready or ❌ Error

**Sidebar**
- System health check
- Model status
- Feature count
- Links to documentation

**Main Content (4 Tabs)**
1. **Single Prediction Tab**
   - Input form (9 fields)
   - Predict button
   - Result card with insights
   
2. **Trend Analysis Tab**
   - Input form (9 fields)
   - Generate Forecast button
   - Results table + line chart
   - Download button
   
3. **KPI Dashboard Tab**
   - 4 metric cards
   - Feature importance bar chart
   - Feature rankings table
   
4. **Batch Predictions Tab**
   - File upload widget
   - File preview table
   - Process button
   - Results table
   - Download button

---

## 🚀 Success Metrics

| Metric | Target | Measurement |
|--------|--------|------------|
| **Prediction Accuracy** | MAE < $50 | Compare against validation set |
| **Load Time** | < 2 seconds | Measure prediction latency |
| **Uptime** | 99.5% | Monitor Streamlit Cloud |
| **User Engagement** | 100+ app runs/week | Track Streamlit Cloud analytics |
| **Error Rate** | < 1% | Monitor exception logs |
| **Feature Importance Stability** | Top 3 features consistent | Track across retraining cycles |

---

## 🔮 Future Scope

### Phase 2 (Q2 2026)
* Real-time data integration (connect to sales databases)
* Multi-product forecasting (separate models per product)
* Advanced seasonality detection
* Confidence intervals on predictions

### Phase 3 (Q3 2026)
* Cloud database integration (PostgreSQL/MongoDB)
* API endpoints for programmatic access
* Mobile-responsive UI
* Advanced analytics (anomaly detection, clustering)

### Phase 4 (Q4 2026)
* SaaS subscription model
* Multi-tenant support
* Custom branding for enterprises
* Webhooks for external integrations

---

## 📋 Acceptance Criteria

The product is considered **DONE** when:

✅ All 10 core features are implemented and tested  
✅ App deploys to Streamlit Cloud without errors  
✅ Boot diagnostics verify dependencies before UI renders  
✅ All 4 tabs render with correct data  
✅ Single prediction works with sample inputs  
✅ 7-day forecast generates without errors  
✅ Batch CSV upload processes correctly  
✅ Feature importance displays correctly  
✅ Error messages are clear and user-friendly  
✅ Documentation is complete (README + PRD + BRD)  
✅ Code is committed to GitHub with clean history  

---

## 📞 Contact & Support

For questions about this PRD:
- Create an issue on GitHub
- Review inline code comments in `app.py`
- Check troubleshooting section in README.md
