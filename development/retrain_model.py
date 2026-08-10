"""
PRODUCTION MODEL RETRAINING SCRIPT (v2.0)

Retrains both XGBoost and RandomForest models using ONLY engineered features.
Features match EXACTLY what app.py sends (13 features).
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

# Set path - Get project root from current directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_data_path = os.path.join(project_root, 'model_data')
csv_file = os.path.join(model_data_path, 'Sample - Superstore.csv')
xgb_output = os.path.join(model_data_path, 'model.pkl')
rf_output = os.path.join(model_data_path, 'rf_model.pkl')

# Load data
print('Loading data...')
df = pd.read_csv(csv_file, encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date')
df = df.ffill()

# Create all engineered features
print('Creating engineered features...')
df['day'] = df['Order Date'].dt.day
df['month'] = df['Order Date'].dt.month
df['weekday'] = df['Order Date'].dt.weekday
df['lag_1'] = df['Sales'].shift(1)
df['lag_7'] = df['Sales'].shift(7)
df['lag_14'] = df['Sales'].shift(14)
df['lag_30'] = df['Sales'].shift(30)
df['rolling_mean_7'] = df['Sales'].rolling(7).mean()
df['rolling_mean_14'] = df['Sales'].rolling(14).mean()
df['rolling_std_7'] = df['Sales'].rolling(7).std()
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['weekofyear'] = df['Order Date'].dt.isocalendar().week.astype(int)
df['trend'] = range(len(df))
df = df.dropna()

# Define engineered features (MUST MATCH app.py)
ENGINEERED_FEATURES = [
    'day', 'month', 'weekday', 'is_weekend', 'weekofyear',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7', 'trend'
]

print('=' * 80)
print('PRODUCTION DUAL MODEL RETRAINING (v2.0)')
print('=' * 80)
print(f'\nUsing {len(ENGINEERED_FEATURES)} engineered features:')
for i, feat in enumerate(ENGINEERED_FEATURES, 1):
    print(f'  {i:2d}. {feat}')

# Select only engineered features
print('\nBuilding training dataset...')
X = df[ENGINEERED_FEATURES].copy()
y = df['Sales'].copy()

# Train-test split
train_size = int(len(X) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print(f'\nTrain-test split: Training {len(X_train)} rows | Testing {len(X_test)} rows')

# 1. Train XGBoost
print(f'\n1. Training XGBoost Regressor...')
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
print(f'   XGBoost MAE: ${xgb_mae:.2f}')

# Save XGBoost model
joblib.dump({
    'model_pipeline': xgb_model,
    'expected_feature_columns': ENGINEERED_FEATURES,
    'model_type': 'XGBoost',
    'mae': xgb_mae,
    'features': ENGINEERED_FEATURES
}, xgb_output)
print(f'   XGBoost saved as {xgb_output}')

# 2. Train RandomForest
print(f'\n2. Training RandomForest Regressor...')
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
print(f'   RandomForest MAE: ${rf_mae:.2f}')

# Save RandomForest model
joblib.dump({
    'model_pipeline': rf_model,
    'expected_feature_columns': ENGINEERED_FEATURES,
    'model_type': 'RandomForest',
    'mae': rf_mae,
    'features': ENGINEERED_FEATURES
}, rf_output)
print(f'   RandomForest saved as {rf_output}')

print('\n' + '=' * 80)
print('DUAL PRODUCTION MODELS RETRAINED & COMPATIBLE FOR STREAMLIT APP v2.0')
print('=' * 80)
