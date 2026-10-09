"""
Data processing and cleaning pipeline for freight rate prediction.
Handles data loading, negative weight correction, imputation, and coordinate lookup.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def get_city_coordinate_maps(df: pd.DataFrame) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """
    Extract deterministic latitude and longitude lookup dictionaries for pickup and delivery cities.
    """
    pickup_map = {}
    delivery_map = {}
    
    for _, row in df[['pickup', 'pickup_lat', 'pickup_lon']].dropna().drop_duplicates('pickup').iterrows():
        pickup_map[row['pickup']] = (float(row['pickup_lat']), float(row['pickup_lon']))
        
    for _, row in df[['delivery', 'delivery_lat', 'delivery_lon']].dropna().drop_duplicates('delivery').iterrows():
        delivery_map[row['delivery']] = (float(row['delivery_lat']), float(row['delivery_lon']))
        
    return pickup_map, delivery_map


def clean_dataset(
    df: pd.DataFrame,
    median_weight: Optional[float] = None,
    daily_market_map: Optional[Dict[pd.Timestamp, float]] = None,
    overall_median_market: Optional[float] = None,
    pickup_coords_map: Optional[Dict[str, Tuple[float, float]]] = None,
    delivery_coords_map: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Cleans freight dataset:
    1. Converts date to datetime.
    2. Corrects inverted negative weights via absolute value.
    3. Imputes missing weights using median.
    4. Imputes missing coordinates using city lookup maps.
    5. Imputes missing market_index values using daily median or global median.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Weight cleaning
    if 'weight' in df.columns:
        neg_count = (df['weight'] < 0).sum()
        df['weight'] = df['weight'].abs()
        if median_weight is None:
            median_weight = float(df['weight'].median())
        null_weight_count = df['weight'].isna().sum()
        df['weight'] = df['weight'].fillna(median_weight)
    else:
        neg_count = 0
        null_weight_count = 0
        
    # 2. Coordinate filling
    if pickup_coords_map is not None and 'pickup_lat' not in df.columns:
        df['pickup_lat'] = df['pickup'].map(lambda p: pickup_coords_map.get(p, (np.nan, np.nan))[0])
        df['pickup_lon'] = df['pickup'].map(lambda p: pickup_coords_map.get(p, (np.nan, np.nan))[1])
        
    if delivery_coords_map is not None and 'delivery_lat' not in df.columns:
        df['delivery_lat'] = df['delivery'].map(lambda d: delivery_coords_map.get(d, (np.nan, np.nan))[0])
        df['delivery_lon'] = df['delivery'].map(lambda d: delivery_coords_map.get(d, (np.nan, np.nan))[1])

    # 3. Market index imputation
    null_market_count = 0
    if 'market_index' in df.columns or daily_market_map is not None:
        if 'market_index' not in df.columns:
            df['market_index'] = np.nan
            
        null_market_count = df['market_index'].isna().sum()
        if daily_market_map is not None:
            df['market_index'] = df['market_index'].fillna(df['date'].map(daily_market_map))
            
        if overall_median_market is None and 'market_index' in df.columns:
            overall_median_market = float(df['market_index'].median())
            
        df['market_index'] = df['market_index'].fillna(overall_median_market)

    stats = {
        'negative_weights_corrected': int(neg_count),
        'missing_weights_imputed': int(null_weight_count),
        'missing_market_index_imputed': int(null_market_count),
        'median_weight': median_weight,
        'overall_median_market': overall_median_market
    }
    
    return df, stats
