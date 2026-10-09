# Spotter Freight Rate Prediction Challenge

Production machine learning pipeline for freight spot rate forecasting. Built for Spotter's Machine Learning Engineer assessment by **Bhavesh Prajapati**.

---

## 🚀 Quick Start (Commands to Run)

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
```bash
python main.py
```
This single command automatically:
1. Cleans and audits the data (corrects inverted negative weights and imputes missing market values).
2. Performs domain-specific feature engineering (Haversine distance, circuity factor, route bearings, calendar cycles, and market interactions).
3. Evaluates models using chronological **Out-of-Time (OOT)** temporal validation.
4. Trains the production ensemble (**CatBoost Rate-per-Mile + CatBoost Residual Regressor + XGBoost**) on the labeled development data (48,000 loads).
5. Generates `validation_predictions.csv` for all 12,000 validation loads (`load_id,predicted_rate`).
6. Enriches and forecasts all 31 days for `data/december_chart_inputs.csv`.
7. Executes Spotter's official `score.py` validator and produces `scorer_results/candidate_december.png`.

### 3. Verify Submission Files with Official Scorer
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
Output:
```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

### 4. Regenerate Reports (PDF & DOCX)
```bash
python scripts/generate_report.py
```

---

## 📊 Summary of Assessment Results

### Model Benchmark on Chronological Out-of-Time Validation (Sep–Oct Holdout)
| Model / Strategy | Target Formulation | MAE ($) | RMSE ($) | MedAE ($) | MAPE (%) | R² Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Quote Total** | `distance * quote_signal` | $246.02 | $711.80 | $60.44 | 12.11% | 0.7824 |
| **Baseline 2: Historical Lane RPM** | Historical Rate/Mile | $222.48 | $682.29 | $105.78 | 9.64% | 0.8001 |
| **CatBoost Direct** | `posted_rate` | $140.46 | $643.22 | $58.67 | 6.29% | 0.8223 |
| **CatBoost Rate-per-Mile** | `posted_rate / distance` | $129.98 | $642.22 | $48.35 | 5.90% | 0.8229 |
| **CatBoost Residual Regressor** | `posted_rate - quote_total` | $131.81 | $641.48 | $51.55 | 6.21% | 0.8233 |
| **XGBoost Regressor** | `posted_rate` (Histogram) | $203.00 | $686.97 | $84.44 | 9.57% | 0.7974 |
| **Production Ensemble (Blended)** | `0.50 RPM + 0.35 Res + 0.15 XGB` | **$118.22** | **$635.70** | **$43.46** | **5.48%** | **0.8265** |

> **Improvement:** The production ensemble achieves a **51.9% reduction in MAE** compared to the baseline quote signal, with a Median Absolute Error of just **$43.46**.

---

## 🔍 Key Findings & Data Quality Audits
1. **Inverted Negative Weights:**
   - 292 negative weights in `train_test.csv` and 145 in `validation.csv` (down to -47,500 lbs).
   - Absolute value distributions proved they were sign-inverted legitimate loads conforming to US 80,000 lb GVWR legal payload maximums (~31,700 lb mean). Corrected via `df['weight'] = df['weight'].abs()`.
2. **Market Index Seasonality:**
   - Missing market indices exhibited low within-day variance (~0.024) and high day-of-week seasonality (mid-week peaks, weekend dips). Imputed using the day's median.
3. **Route Circuity:**
   - Highway distance / Great Circle (Haversine) ratio averaged 1.194, matching transportation industry road circuity factors.

---

## 📁 Repository Structure
```text
├── data/                                 # Datasets
│   ├── train_test.csv                    # 48,000 labeled development loads
│   ├── validation.csv                    # 12,000 unlabelled validation loads
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv         # 31 fixed December test inputs
├── src/                                  # Production package
│   ├── __init__.py
│   ├── data_processing.py                # Data cleaning, weight inversion & imputation
│   ├── feature_engineering.py            # Haversine, circuity, bearing & calendar cycles
│   └── models.py                         # Ensemble & December forecasting models
├── scripts/
│   ├── evaluate_models.py                # Multi-model benchmarking script
│   └── generate_report.py                # Generates PDF & DOCX reports
├── scorer_results/
│   └── candidate_december.png            # Generated chart from score.py
├── main.py                               # Single-command runner
├── score.py                              # Official Spotter validator
├── requirements.txt                      # Project dependencies
├── validation_predictions.csv            # 12,000 final predictions
├── Freight_Rate_ML_Assessment_Report.pdf  # Comprehensive assessment report (PDF)
├── Freight_Rate_ML_Assessment_Report.docx # Comprehensive assessment report (DOCX)
└── LOOM_WALKTHROUGH_SCRIPT.md            # Word-for-word 2-3 minute video script
```

---

## 📦 Submission Deliverables Checklist
- [x] GitHub repository with modular code, dependencies, and run instructions.
- [x] `validation_predictions.csv` containing exactly `load_id,predicted_rate` (12,000 rows).
- [x] `candidate_december.png` produced by `score.py`.
- [x] PDF / DOCX report detailing data splits, validation, and December chart (`Freight_Rate_ML_Assessment_Report.pdf`).
- [x] 2-3 minute Loom walkthrough guide (`LOOM_WALKTHROUGH_SCRIPT.md`).
