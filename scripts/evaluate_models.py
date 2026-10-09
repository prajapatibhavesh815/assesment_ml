import pandas as pd
import numpy as np
import catboost as cb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

def add_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['dayofyear'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Haversine distance
    lon1, lat1, lon2, lat2 = map(np.radians, [df['pickup_lon'], df['pickup_lat'], df['delivery_lon'], df['delivery_lat']])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['haversine_dist'] = 3958.8 * c
    df['circuity'] = df['distance'] / (df['haversine_dist'] + 1e-5)
    df['dlat'] = df['delivery_lat'] - df['pickup_lat']
    df['dlon'] = df['delivery_lon'] - df['pickup_lon']
    df['bearing'] = np.arctan2(df['dlat'], df['dlon'])
    
    # Weight features
    df['weight_per_mile'] = df['weight'] / (df['distance'] + 1e-5)
    df['weight_dist'] = (df['weight'] * df['distance']) / 1e6
    df['heavy_load'] = (df['weight'] > 40000).astype(int)
    
    # Market interaction
    if 'market_index' in df.columns:
        df['market_dist'] = df['distance'] * df['market_index']
    
    # Quote signal interaction
    if 'quote_signal' in df.columns:
        df['quote_total'] = df['distance'] * df['quote_signal']
        if 'market_index' in df.columns:
            df['quote_market'] = df['quote_signal'] * df['market_index']
            
    return df

def clean_data(df, median_weight=None, daily_market_map=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    # Invert negative weights
    df['weight'] = df['weight'].abs()
    if median_weight is None:
        median_weight = df['weight'].median()
    df['weight'] = df['weight'].fillna(median_weight)
    
    if daily_market_map is None:
        daily_market_map = df.groupby('date')['market_index'].median().to_dict()
    df['market_index'] = df['market_index'].fillna(df['date'].map(daily_market_map))
    # If still any null market_index (e.g. date not in map), fill with overall median
    df['market_index'] = df['market_index'].fillna(df['market_index'].median())
    
    return df, median_weight, daily_market_map

def compute_metrics(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"[{name:35s}] MAE: {mae:7.2f} | RMSE: {rmse:7.2f} | MedAE: {medae:7.2f} | MAPE: {mape:5.2f}% | R2: {r2:7.4f}")
    return {"model": name, "mae": mae, "rmse": rmse, "medae": medae, "mape": mape, "r2": r2}

def main():
    print("Loading data...")
    raw_train = pd.read_csv('data/train_test.csv')
    cleaned_train, med_weight, market_map = clean_data(raw_train)
    feat_train = add_features(cleaned_train)
    
    # Temporal Train / Validation split
    # Training: Jan 1 to Aug 31 (8 months, 38,477 rows)
    # Validation: Sep 1 to Oct 31 (2 months, 9,523 rows)
    # This precisely mimics the deployment setting where train is 10 months and validation is 2 months (Nov-Dec)!
    tr = feat_train[feat_train['date'] < '2025-09-01'].copy()
    val = feat_train[feat_train['date'] >= '2025-09-01'].copy()
    print(f"Train samples (Jan-Aug): {len(tr):,}, Val samples (Sep-Oct): {len(val):,}\n")
    
    # Baseline 1: Quote Signal * Distance
    print("=== BASELINES ===")
    p_base = val['quote_total']
    compute_metrics(val['posted_rate'], p_base, "Baseline: Quote Total (dist*quote)")
    
    # Baseline 2: Historical Lane / City Rate per Mile
    lane_rpm = tr.groupby(['pickup', 'delivery'])['posted_rate'].sum() / tr.groupby(['pickup', 'delivery'])['distance'].sum()
    overall_rpm = tr['posted_rate'].sum() / tr['distance'].sum()
    val_lane_rpm = val.apply(lambda r: lane_rpm.get((r['pickup'], r['delivery']), overall_rpm), axis=1)
    compute_metrics(val['posted_rate'], val_lane_rpm * val['distance'], "Baseline: Historical Lane Rate/Mile")
    
    # Feature subsets
    num_features = [
        'distance', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'haversine_dist', 'circuity', 'dlat', 'dlon', 'bearing', 'weight_per_mile', 'weight_dist',
        'heavy_load', 'dayofweek', 'is_weekend', 'dayofyear', 'month', 'day',
        'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy', 'market_index', 'market_dist'
    ]
    all_num_features = num_features + ['quote_signal', 'quote_total', 'quote_market']
    cat_features = ['pickup', 'delivery', 'equipment']
    
    print("\n=== MODELS WITH QUOTE_SIGNAL ===")
    # 1. CatBoost on Direct posted_rate
    cb_direct = cb.CatBoostRegressor(
        iterations=800,
        learning_rate=0.06,
        depth=7,
        loss_function='RMSE',
        eval_metric='MAE',
        cat_features=cat_features,
        verbose=0,
        random_seed=42
    )
    cb_direct.fit(tr[all_num_features + cat_features], tr['posted_rate'])
    p_cb_direct = cb_direct.predict(val[all_num_features + cat_features])
    compute_metrics(val['posted_rate'], p_cb_direct, "CatBoost (Direct posted_rate)")
    
    # 2. CatBoost on Rate per Mile
    cb_rpm = cb.CatBoostRegressor(
        iterations=800,
        learning_rate=0.06,
        depth=7,
        loss_function='RMSE',
        eval_metric='MAE',
        cat_features=cat_features,
        verbose=0,
        random_seed=42
    )
    tr_rpm = tr['posted_rate'] / tr['distance']
    cb_rpm.fit(tr[all_num_features + cat_features], tr_rpm)
    p_cb_rpm = cb_rpm.predict(val[all_num_features + cat_features]) * val['distance']
    compute_metrics(val['posted_rate'], p_cb_rpm, "CatBoost (Target: Rate per Mile)")

    # 3. CatBoost on Residual over Quote Total
    tr_res = tr['posted_rate'] - tr['quote_total']
    cb_res = cb.CatBoostRegressor(
        iterations=800,
        learning_rate=0.06,
        depth=7,
        loss_function='RMSE',
        eval_metric='MAE',
        cat_features=cat_features,
        verbose=0,
        random_seed=42
    )
    cb_res.fit(tr[all_num_features + cat_features], tr_res)
    p_cb_res = val['quote_total'] + cb_res.predict(val[all_num_features + cat_features])
    compute_metrics(val['posted_rate'], p_cb_res, "CatBoost (Target: Residual over Quote)")

    # 4. XGBoost on Direct posted_rate
    # Convert categories to category dtype for XGBoost
    tr_xgb = tr[all_num_features + cat_features].copy()
    val_xgb = val[all_num_features + cat_features].copy()
    for c in cat_features:
        tr_xgb[c] = tr_xgb[c].astype('category')
        val_xgb[c] = val_xgb[c].astype('category')
        
    xgb_direct = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=7,
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb_direct.fit(tr_xgb, tr['posted_rate'])
    p_xgb_direct = xgb_direct.predict(val_xgb)
    compute_metrics(val['posted_rate'], p_xgb_direct, "XGBoost (Direct posted_rate)")
    
    # 5. Ensemble (CatBoost Direct + CatBoost RPM + XGBoost)
    p_ens = 0.45 * p_cb_direct + 0.25 * p_cb_rpm + 0.30 * p_xgb_direct
    compute_metrics(val['posted_rate'], p_ens, "Ensemble (CatBoost + XGBoost)")
    
    print("\n=== MODELS WITHOUT QUOTE_SIGNAL (FOR DECEMBER INPUTS) ===")
    cb_no_qs = cb.CatBoostRegressor(
        iterations=800,
        learning_rate=0.06,
        depth=7,
        cat_features=cat_features,
        verbose=0,
        random_seed=42
    )
    cb_no_qs.fit(tr[num_features + cat_features], tr['posted_rate'])
    p_no_qs = cb_no_qs.predict(val[num_features + cat_features])
    compute_metrics(val['posted_rate'], p_no_qs, "CatBoost (No Quote Signal)")

if __name__ == '__main__':
    main()
