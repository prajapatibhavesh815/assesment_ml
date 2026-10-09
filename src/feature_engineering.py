"""
Feature engineering module for freight spot rate prediction.
Constructs spatial, temporal, cargo weight, and market interaction signals.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def compute_haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Computes great-circle distance in statute miles between two coordinate sets."""
    rad_lon1, rad_lat1, rad_lon2, rad_lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = rad_lon2 - rad_lon1
    dlat = rad_lat2 - rad_lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rad_lat1) * np.cos(rad_lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    miles = 3958.8 * c
    return miles


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive domain feature engineering for freight rate modeling.
    """
    df = df.copy()
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
        
    # --- Temporal Features ---
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Cyclical representations
    df['sin_dow'] = np.sin(2.0 * np.pi * df['dayofweek'] / 7.0)
    df['cos_dow'] = np.cos(2.0 * np.pi * df['dayofweek'] / 7.0)
    df['sin_doy'] = np.sin(2.0 * np.pi * df['dayofyear'] / 365.25)
    df['cos_doy'] = np.cos(2.0 * np.pi * df['dayofyear'] / 365.25)
    
    # --- Spatial & Route Features ---
    if all(c in df.columns for c in ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']):
        df['haversine_dist'] = compute_haversine_distance(
            df['pickup_lat'].values, df['pickup_lon'].values,
            df['delivery_lat'].values, df['delivery_lon'].values
        )
        df['circuity'] = df['distance'] / (df['haversine_dist'] + 1e-5)
        df['dlat'] = df['delivery_lat'] - df['pickup_lat']
        df['dlon'] = df['delivery_lon'] - df['pickup_lon']
        df['manhattan_deg'] = np.abs(df['dlat']) + np.abs(df['dlon'])
        df['mid_lat'] = (df['pickup_lat'] + df['delivery_lat']) / 2.0
        df['mid_lon'] = (df['pickup_lon'] + df['delivery_lon']) / 2.0
        df['bearing'] = np.arctan2(df['dlat'], df['dlon'])
        
    # --- Weight & Equipment Dynamics ---
    df['weight_per_mile'] = df['weight'] / (df['distance'] + 1e-5)
    df['weight_dist'] = (df['weight'] * df['distance']) / 1e6
    df['heavy_load'] = (df['weight'] > 40000.0).astype(int)
    df['light_load'] = (df['weight'] < 20000.0).astype(int)

    # --- Market & Macro Economics ---
    if 'market_index' in df.columns:
        df['market_dist'] = df['distance'] * df['market_index']

    # --- Broker Quote Interactions (if available) ---
    if 'quote_signal' in df.columns:
        df['quote_total'] = df['distance'] * df['quote_signal']
        if 'market_index' in df.columns:
            df['quote_market'] = df['quote_signal'] * df['market_index']
        df['quote_weight_dist'] = df['quote_signal'] * df['weight_dist']

    return df


def get_feature_lists() -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns lists of numerical and categorical features for models with and without quote_signal.
    """
    cat_cols = ['pickup', 'delivery', 'equipment']
    
    base_num_cols = [
        'distance', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'haversine_dist', 'circuity', 'dlat', 'dlon', 'manhattan_deg', 'mid_lat', 'mid_lon', 'bearing',
        'weight_per_mile', 'weight_dist', 'heavy_load', 'light_load',
        'dayofweek', 'is_weekend', 'dayofyear', 'month', 'day',
        'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
        'market_index', 'market_dist'
    ]
    
    quote_num_cols = base_num_cols + ['quote_signal', 'quote_total', 'quote_market', 'quote_weight_dist']
    
    return base_num_cols, quote_num_cols, cat_cols
