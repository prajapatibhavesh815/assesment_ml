# Spotter Freight Rate ML Assessment - 2-3 Minute Loom Recording Script

**Candidate:** Bhavesh Prajapati  
**Target Duration:** ~2 minutes 30 seconds  
**Format:** Screen recording with webcam bubble (or screen share of IDE / Report / Chart)

---

## 🎬 Section-by-Section Speaking Guide

### 1. Introduction & Overview (~20 seconds)
> *"Hi Ena and the Spotter Engineering team! My name is Bhavesh Prajapati. Today, I'm excited to walk you through my machine learning solution for the Spotter Freight Rate Prediction assessment.*  
> *The objective was to forecast posted freight spot rates accurately across various lanes, equipment types, and market conditions, validated via `score.py`."*

---

### 2. Key Findings & Data Quality Issues (~45 seconds)
> *(Show your IDE or the Report Section 2 on screen)*  
> *"When conducting exploratory data analysis across the 48,000 development loads and 12,000 validation loads, I identified three major data-quality insights:*  
> 
> 1. **Inverted Negative Weights:** I discovered 292 negative weights in the training set and 145 in the validation set, reaching down to -47,500 lbs. When analyzing the distribution, their absolute values matched legitimate freight payload weights perfectly (mean ~31,700 lbs, conforming to the legal 80,000 lb GVWR threshold). I resolved this by taking the absolute value `abs(weight)` and imputing remaining missing values with the robust median.
> 2. **Market Index Seasonality:** Missing `market_index` values exhibited strong day-of-week seasonality (peaking mid-week and dipping on weekends) with very low intra-day variance (~0.024). I imputed missing values using the day's median.
> 3. **Routing Circuity:** The highway distance relative to Great Circle (Haversine) distance averaged 1.194, which aligns with commercial trucking routing circuity."*

---

### 3. Training & Validation Strategy (~35 seconds)
> *(Show the Benchmark Table on screen)*  
> *"For our validation approach, naive random K-Fold cross-validation would cause **temporal data leakage**, as future economic shocks and holiday capacity spikes would leak into earlier training iterations.  
> To simulate the real-world deployment scenario (10 months historical training -> 2 months forward validation), I implemented an **Out-of-Time (OOT) temporal split**:
> - Training fold: Months 1 to 8 (Jan–Aug 2025; 38,477 loads)
> - Holdout validation fold: Months 9 to 10 (Sep–Oct 2025; 9,523 loads)
> 
> This gave us an honest, forward-looking evaluation of generalizability before predicting on the November–December test set."*

---

### 4. Model Architecture & Reasoning (~40 seconds)
> *"Instead of predicting raw posted rates directly, I developed a domain-aware ensemble:
> 1. **CatBoost on Rate-per-Mile (`posted_rate / distance`)**: This normalizes variance between short hauls (100 miles) and cross-country routes (2,500 miles).
> 2. **CatBoost on Residual over Quote (`posted_rate - quote_total`)**: This leverages the automated broker quote signal and concentrates tree learning on pricing spreads.
> 3. **XGBoost Regressor**: For model diversity and histogram-based split optimization.
> 
> **Results:** Our blended ensemble reduced the Mean Absolute Error from the **$246.02 baseline down to $118.22**—a **51.9% error reduction**, with an $R^2$ of **0.8265** and Median Absolute Error of just **$43.46**."*

---

### 5. Code Walkthrough & December Prediction Chart (~30 seconds)
> *(Show `main.py` and `scorer_results/candidate_december.png`)*  
> *"In the code structure:
> - `src/data_processing.py`: Handles deterministic coordinate lookup, negative weight restoration, and market imputation.
> - `src/feature_engineering.py`: Builds Haversine distances, circuity factors, bearing angles, cyclical calendar encodings, and market interactions.
> - `src/models.py`: Encapsulates the multi-target ensemble and fallback cold-start pricing.
> - `main.py`: One-command execution that trains the models, generates `validation_predictions.csv`, fills `data/december_chart_inputs.csv`, and runs Spotter's `score.py`.
> 
> Here is the output `candidate_december.png`:
> Notice how the predicted rate for Lexington to Fort Wayne (Dry Van, 360 miles) naturally displays the weekly Wednesday dispatch peaks, weekend dips, and the holiday capacity crunch leading up to New Year's Eve, sitting realistically at $2.29 to $2.37 per mile.
> 
> All files passed `score.py` with 100% compliance. Thank you for your time, and I look forward to the next steps!"*

---

## 💡 Quick Tips for Recording
- Keep your camera at eye level.
- Have `candidate_december.png` and `Freight_Rate_ML_Assessment_Report.pdf` open in tabs so you can switch smoothly.
- Speak with confidence—all numbers and findings are tested and verified!
