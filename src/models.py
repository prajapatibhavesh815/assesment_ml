"""
Model architecture, training routines, and ensemble predictors for freight rate prediction.
"""

from typing import Dict, List, Optional, Tuple
import os
import joblib
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error


class FreightRateEnsemble:
    """
    Ensemble model combining CatBoost Rate-Per-Mile, CatBoost Residual-over-Quote,
    and XGBoost Direct Regressors for robust, low-variance freight pricing.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cat_cols: List[str] = ['pickup', 'delivery', 'equipment']
        self.feature_cols: List[str] = []
        
        # Sub-models
        self.model_cb_rpm = cb.CatBoostRegressor(
            iterations=400,
            learning_rate=0.08,
            depth=6,
            loss_function='RMSE',
            eval_metric='MAE',
            cat_features=self.cat_cols,
            verbose=0,
            random_seed=random_state
        )
        
        self.model_cb_res = cb.CatBoostRegressor(
            iterations=400,
            learning_rate=0.08,
            depth=6,
            loss_function='RMSE',
            eval_metric='MAE',
            cat_features=self.cat_cols,
            verbose=0,
            random_seed=random_state
        )
        
        self.model_cb_dir = cb.CatBoostRegressor(
            iterations=400,
            learning_rate=0.08,
            depth=6,
            loss_function='RMSE',
            eval_metric='MAE',
            cat_features=self.cat_cols,
            verbose=0,
            random_seed=random_state
        )
        
        # Ensembling weights (tuned via temporal out-of-time validation)
        self.w_rpm = 0.50
        self.w_res = 0.35
        self.w_dir = 0.15

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FreightRateEnsemble":
        self.feature_cols = [c for c in X.columns if c not in ['load_id', 'posted_rate', 'date']]
        
        # 1. Train CatBoost on Rate per Mile
        rpm_target = y / X['distance']
        print("Training CatBoost (Rate per Mile objective)...")
        self.model_cb_rpm.fit(X[self.feature_cols], rpm_target)
        
        # 2. Train CatBoost on Residual over Quote Total
        quote_total = X['quote_total']
        residual_target = y - quote_total
        print("Training CatBoost (Residual over Quote objective)...")
        self.model_cb_res.fit(X[self.feature_cols], residual_target)
        
        # 3. Train CatBoost on Direct posted_rate
        print("Training CatBoost (Direct posted_rate objective)...")
        self.model_cb_dir.fit(X[self.feature_cols], y)
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Prediction 1: RPM * distance
        p_rpm = self.model_cb_rpm.predict(X[self.feature_cols]) * X['distance'].values
        
        # Prediction 2: Quote Total + residual
        p_res = X['quote_total'].values + self.model_cb_res.predict(X[self.feature_cols])
        
        # Prediction 3: Direct posted rate
        p_dir = self.model_cb_dir.predict(X[self.feature_cols])
        
        # Blended weighted prediction
        ensemble_pred = self.w_rpm * p_rpm + self.w_res * p_res + self.w_dir * p_dir
        
        # Post-processing: freight rates must be positive
        ensemble_pred = np.maximum(ensemble_pred, 50.0)
        return np.round(ensemble_pred, 2)


class DecemberChartModel:
    """
    Dedicated model for prediction scenarios without quote signals (e.g. December Chart Inputs).
    Predicts posted rates using route, temporal, market index, and equipment dynamics.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.cat_cols = ['pickup', 'delivery', 'equipment']
        self.feature_cols: List[str] = []
        
        self.model = cb.CatBoostRegressor(
            iterations=500,
            learning_rate=0.06,
            depth=6,
            loss_function='RMSE',
            eval_metric='MAE',
            cat_features=self.cat_cols,
            verbose=0,
            random_seed=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> "DecemberChartModel":
        self.feature_cols = feature_cols
        rpm_target = y / X['distance']
        print("Training December Chart Model (CatBoost Rate per Mile)...")
        self.model.fit(X[self.feature_cols], rpm_target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred_rpm = self.model.predict(X[self.feature_cols])
        pred_rate = pred_rpm * X['distance'].values
        pred_rate = np.maximum(pred_rate, 50.0)
        return np.round(pred_rate, 2)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n[{model_name}] Performance Evaluation:")
    print(f"  - MAE:   ${mae:.2f}")
    print(f"  - RMSE:  ${rmse:.2f}")
    print(f"  - MedAE: ${medae:.2f}")
    print(f"  - MAPE:   {mape:.2f}%")
    print(f"  - R2:     {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'medae': medae, 'mape': mape, 'r2': r2}
