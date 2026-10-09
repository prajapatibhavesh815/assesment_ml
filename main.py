"""
Main pipeline execution script for Spotter Freight Rate Prediction Challenge.
Trains production models, generates validation_predictions.csv, fills december_chart_inputs.csv,
and runs score.py validation.
"""

import sys
import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

import subprocess
import numpy as np
import pandas as pd

from src.data_processing import clean_dataset, get_city_coordinate_maps
from src.feature_engineering import engineer_features, get_feature_lists
from src.models import FreightRateEnsemble, DecemberChartModel, evaluate_predictions


def main():
    print("=" * 70)
    print("  SPOTTER FREIGHT RATE PREDICTION - END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Paths resolution
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    
    train_path = data_dir / "train_test.csv" if (data_dir / "train_test.csv").exists() else base_dir / "train-test.csv"
    val_path = data_dir / "validation.csv" if (data_dir / "validation.csv").exists() else base_dir / "validation.csv"
    val_tmpl_path = data_dir / "validation_predictions_template.csv" if (data_dir / "validation_predictions_template.csv").exists() else base_dir / "validation-predictions-template.csv"
    dec_path = data_dir / "december_chart_inputs.csv" if (data_dir / "december_chart_inputs.csv").exists() else base_dir / "december-chart-inputs.csv"
    
    out_val_pred_path = base_dir / "validation_predictions.csv"
    out_dec_path = dec_path

    print(f"Loading development data from: {train_path.name}")
    raw_train = pd.read_csv(train_path)
    print(f"Raw train shape: {raw_train.shape}")

    print(f"Loading validation data from: {val_path.name}")
    raw_val = pd.read_csv(val_path)
    print(f"Raw validation shape: {raw_val.shape}")

    # 2. Extract city coordinates & daily market values
    pickup_coords, delivery_coords = get_city_coordinate_maps(raw_train)
    combined_dates = pd.concat([raw_train[['date', 'market_index']], raw_val[['date', 'market_index']]])
    combined_dates['date'] = pd.to_datetime(combined_dates['date'])
    daily_market_map = combined_dates.groupby('date')['market_index'].median().to_dict()
    global_median_market = float(raw_train['market_index'].median())

    # 3. Clean training & validation data
    print("\n--- Data Cleaning & Preprocessing ---")
    clean_tr, tr_stats = clean_dataset(
        raw_train,
        daily_market_map=daily_market_map,
        overall_median_market=global_median_market
    )
    print(f"Train cleaning stats: {tr_stats}")

    clean_val, val_stats = clean_dataset(
        raw_val,
        median_weight=tr_stats['median_weight'],
        daily_market_map=daily_market_map,
        overall_median_market=global_median_market,
        pickup_coords_map=pickup_coords,
        delivery_coords_map=delivery_coords
    )
    print(f"Validation cleaning stats: {val_stats}")

    # 4. Feature engineering
    print("\n--- Feature Engineering ---")
    feat_tr = engineer_features(clean_tr)
    feat_val = engineer_features(clean_val)
    base_num_cols, quote_num_cols, cat_cols = get_feature_lists()
    print(f"Engineered features. Model feature count: {len(quote_num_cols) + len(cat_cols)}")

    # 5. Out-of-Time Temporal Evaluation (Jan-Aug train vs Sep-Oct val)
    print("\n--- Out-of-Time Temporal Validation ---")
    tr_oot = feat_tr[feat_tr['date'] < '2025-09-01'].copy()
    val_oot = feat_tr[feat_tr['date'] >= '2025-09-01'].copy()
    print(f"OOT Train samples: {len(tr_oot):,}, OOT Val samples: {len(val_oot):,}")
    
    oot_ensemble = FreightRateEnsemble(random_state=42)
    oot_ensemble.fit(tr_oot, tr_oot['posted_rate'])
    pred_oot = oot_ensemble.predict(val_oot)
    evaluate_predictions(val_oot['posted_rate'].values, pred_oot, "OOT Freight Rate Ensemble")

    # 6. Train Production Ensemble on Full Training Data (48,000 samples)
    print("\n--- Training Production Ensemble (Full 48,000 Development Set) ---")
    prod_ensemble = FreightRateEnsemble(random_state=42)
    prod_ensemble.fit(feat_tr, feat_tr['posted_rate'])

    # 7. Generate Predictions for validation.csv
    print("\n--- Generating Predictions for validation.csv (12,000 rows) ---")
    pred_val_rates = prod_ensemble.predict(feat_val)
    
    # Read template to ensure exact ordering
    tmpl = pd.read_csv(val_tmpl_path)
    val_id_map = dict(zip(feat_val['load_id'], pred_val_rates))
    tmpl['predicted_rate'] = tmpl['load_id'].map(val_id_map)

    # Validate output constraints
    assert len(tmpl) == 12000, f"Expected 12,000 rows, got {len(tmpl)}"
    assert tmpl['predicted_rate'].isna().sum() == 0, "Found NaN predictions!"
    assert (tmpl['predicted_rate'] > 0).all(), "Found non-positive predictions!"

    tmpl.to_csv(out_val_pred_path, index=False)
    # Also save to data/ if present
    if (data_dir / "validation_predictions.csv").parent.exists():
        tmpl.to_csv(data_dir / "validation_predictions.csv", index=False)
    print(f"Saved validation predictions to: {out_val_pred_path}")
    print(tmpl.head(5))

    # 8. Train December Chart Model & Predict december_chart_inputs.csv
    print("\n--- Training December Chart Model & Generating Predictions ---")
    dec_df = pd.read_csv(dec_path)
    print(f"December inputs shape: {dec_df.shape}")

    # Enrich December inputs with city coordinates and daily market index
    dec_df_clean, _ = clean_dataset(
        dec_df,
        median_weight=tr_stats['median_weight'],
        daily_market_map=daily_market_map,
        overall_median_market=global_median_market,
        pickup_coords_map=pickup_coords,
        delivery_coords_map=delivery_coords
    )
    dec_feat = engineer_features(dec_df_clean)

    dec_model = DecemberChartModel(random_state=42)
    dec_feature_cols = base_num_cols + cat_cols
    dec_model.fit(feat_tr, feat_tr['posted_rate'], dec_feature_cols)

    pred_dec_rates = dec_model.predict(dec_feat)
    dec_df['predicted_rate'] = pred_dec_rates

    # Save to both file locations to be robust
    dec_df.to_csv(dec_path, index=False)
    if (base_dir / "december-chart-inputs.csv").exists():
        dec_df.to_csv(base_dir / "december-chart-inputs.csv", index=False)
    if (data_dir / "december_chart_inputs.csv").exists():
        dec_df.to_csv(data_dir / "december_chart_inputs.csv", index=False)
    if (data_dir / "december-chart-inputs.csv").exists():
        dec_df.to_csv(data_dir / "december-chart-inputs.csv", index=False)

    print(f"Filled December chart inputs saved to: {dec_path}")
    print(dec_df.head(5))

    # 9. Execute Official score.py Validator
    print("\n--- Running Official Scorer: score.py ---")
    score_script = base_dir / "score.py"
    cmd = [
        sys.executable,
        str(score_script),
        "--predictions", str(out_val_pred_path),
        "--december-predictions", str(dec_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", result.stdout)
    if result.stderr:
        print("STDERR:\n", result.stderr)
        
    if result.returncode == 0:
        print("\nSUCCESS: All files validated successfully by score.py!")
    else:
        print(f"\nERROR: score.py failed with return code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
