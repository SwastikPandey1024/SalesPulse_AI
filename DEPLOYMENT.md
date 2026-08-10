# 🌐 SalesPulse AI v2.0 - Live Streamlit Deployment Guide

This guide provides step-by-step instructions to deploy **SalesPulse AI Version 2.0** live on **Streamlit Community Cloud**.

---

## 📋 Prerequisites & Checklist

Before starting deployment, verify that the following files are present in the repository root:

- [x] **`app.py`** – Main application entry point located in the root directory.
- [x] **`requirements.txt`** – Dependencies (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `plotly`, `joblib`).
- [x] **`model_data/model.pkl`** – Primary trained XGBoost model bundle.
- [x] **`model_data/rf_model.pkl`** – Alternative trained RandomForest model bundle.
- [x] **`model_data/Sample - Superstore.csv`** – Historical dataset for EDA Explorer.
- [x] **`.streamlit/config.toml`** – Custom UI theme configuration.

---

## 🚀 Deployment Steps (GitHub -> Streamlit Cloud)

### Step 1: Push Changes to GitHub

Commit and push all upgraded Version 2.0 files to your GitHub repository (`main` branch):

```bash
git add .
git commit -m "Upgrade to SalesPulse AI v2.0 with multi-model engine, Plotly forecasting & EDA"
git push origin main
```

---

### Step 2: Sign In to Streamlit Cloud

1. Open your browser and navigate to [https://share.streamlit.io](https://share.streamlit.io).
2. Sign in with your **GitHub account**.

---

### Step 3: Create & Configure New App

1. Click the **"New App"** button (top right of your dashboard).
2. Fill in the deployment details:
   - **Repository:** `SwastikPandey1024/MLmodelStreamlit` (or your repository name)
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** Custom domain slug (e.g. `salespulse-ai`)

---

### Step 4: Launch & Deploy

1. Click **"Deploy!"**.
2. Streamlit Cloud will automatically build the environment, install packages from `requirements.txt`, load `model.pkl`, and launch your app.
3. Once built, your live URL will be active (e.g. `https://salespulse-ai.streamlit.app`).

---

## 🔍 Post-Deployment Verification

After the app launches:

1. **Verify Boot Status:** Ensure the sidebar shows `✅ Model: Loaded (XGBoost)` and `✅ Status: Ready`.
2. **Test Predictions:** Run a single-day prediction on Tab 1.
3. **Test Forecast:** Generate a 14-day forecast with confidence intervals on Tab 2.
4. **Test Scenario Simulator:** Adjust promotional and price levers on Tab 3.
5. **Test EDA Dashboard:** Filter regional and category metrics on Tab 4.

---

## 🛠️ Troubleshooting & FAQs

| Symptom | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError` | Missing package in `requirements.txt` | Add package name to `requirements.txt` and push to GitHub |
| `FileNotFoundError: model.pkl` | Model file missing from `model_data/` | Verify `model_data/model.pkl` is tracked by Git (`git add model_data/model.pkl`) |
| `MemoryLimitExceeded` | Streamlit free tier memory limit | Models are optimized under 100MB combined |
| Stale Cache / Old Version | Streamlit app cached previous state | Open app menu (top right) -> Click "Clear Cache" -> "Rerun" |

---

## 📱 Live App Capabilities (v2.0)

- ⚡ **XGBoost & RandomForest Inference**
- 📈 **7, 14 & 30-Day Multi-Horizon Forecasting**
- 🎯 **"What-If" Business Scenario Simulator**
- 📊 **Superstore EDA & Category Heatmaps**
- 📁 **Smart Batch CSV Predictions**
