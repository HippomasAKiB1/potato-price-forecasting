# Potato Price Forecasting

This repository contains data, models, notebooks and results for forecasting wholesale potato prices using time-series, LSTM, XGBoost and hybrid approaches. The work includes data preparation, exploratory analysis, model training, evaluation, and interpretability (SHAP) analyses.

**Key features**
- Multi-model comparison: LSTM, XGBoost, and a hybrid LSTM-residual model
- Reproducible notebook-driven analysis in `potato_forecasting.ipynb`
- Trained model artefacts in `models/` and summarized results in `tables/` and `figures/`

**License & citation**: Please see the repository owner for license and citation preferences before re-using data or code.

---

**Repository structure**

- `potato_forecasting.ipynb` — Main notebook combining preprocessing, modelling, evaluation and plots.
- `potato_master_final.csv` — Primary aggregated dataset (root-level copy).
- `datasets/` — Raw and derived datasets used in experiments:
	- `potato_features.csv`
	- `potato_master_final.csv`
- `models/` — Saved model files and model-related metadata (e.g., `lstm_model.keras`, `hybrid_lstm_residual.keras`, `xgboost_best_params.json`).
- `figures/` — Generated figures and plots used in the report.
- `tables/` — CSVs with numerical results and tables referenced in the analysis.
- `logs/` — Training logs and session logs.

**Data**

The datasets are included in the `datasets/` directory. `potato_master_final.csv` contains the final cleaned and merged features used for modelling. Inspect the notebook `potato_forecasting.ipynb` for the exact preprocessing, feature engineering and train/validation split logic.

**Getting started**

1. Recommended Python: 3.8+.
2. Create a virtual environment and install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab tensorflow xgboost shap statsmodels
```

3. Open the main notebook:

```bash
jupyter lab potato_forecasting.ipynb
```

4. Run notebook cells in order. Key sections in the notebook:
- Data loading & cleaning
- Feature engineering and stationarity checks
- Model training (LSTM, XGBoost, hybrid)
- Evaluation (RMSE, MAE, Diebold-Mariano tests)
- Interpretability (SHAP values, seasonal analyses)

**Using saved models**

Saved models are in the `models/` folder. The notebook demonstrates how to load and evaluate these models; examples:

- Load Keras model:
```python
from tensorflow import keras
model = keras.models.load_model('models/lstm_model.keras')
```
- Load XGBoost parameters / models as used in experiments (see `models/xgboost_best_params.json`).

**Results**

Summary tables and figures produced during experiments are stored in `tables/` and `figures/`. Notable outputs include:

- `tables/table05_model_comparison.csv` — Model comparison metrics
- `tables/table05b_diebold_mariano.csv` — Forecast comparison tests
- `tables/table06_shap_importance.csv` — Feature importance from SHAP

Check the main notebook sections for the code that generated each table/figure.

**Reproducibility & notes**

- Random seeds and environment details (package versions) are recorded in training logs when available in `logs/`.
- If you plan to re-train models, ensure you have sufficient GPU resources for LSTM training or reduce epochs/batch size for CPU runs.

**Extending this work**

- Add cross-validation and more robust hyperparameter tuning for XGBoost and LSTM.
- Experiment with exogenous features and external economic indicators.
- Build a simple web API for serving short-term forecasts.

**Contact / Acknowledgements**

If you have questions or want to collaborate, please open an issue or contact the repository owner. Acknowledgements and detailed references should be added here as appropriate.

---

Thank you for checking out this project — run `potato_forecasting.ipynb` to reproduce analyses and inspect modelling choices.
