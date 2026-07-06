# AI4I 2020 Predictive Maintenance Dataset Preparation

This workspace contains a complete local preparation pipeline for the AI4I 2020 predictive maintenance dataset.

## Structure

- `data/raw/ai4i2020.csv`: original dataset extracted from the zip file.
- `data/processed/ai4i_clean_standardized.csv`: cleaned dataset with engineered features, z-score columns, and type one-hot columns.
- `data/factories/iid/`: IID factory/client CSV files.
- `data/factories/moderate_non_iid/`: moderate Non-IID factory/client CSV files.
- `data/factories/highly_non_iid/`: highly Non-IID factory/client CSV files.
- `figures/`: presentation-ready PNGs for class distribution, type distribution, correlations, and client distributions.
- `reports/`: EDA tables and one-page dataset summary.
- `src/prepare_ai4i_dataset.py`: reproducible pipeline script.
- `notebooks/ai4i_dataset_preparation.ipynb`: notebook-style walkthrough.

## Run

Use the bundled Codex Python runtime if your default Python does not have pandas/Pillow:

```powershell
& 'C:\Users\8U7HYBBY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\prepare_ai4i_dataset.py --clients 5 --seed 42
```

The script is intentionally lightweight and does not require scikit-learn, matplotlib, or seaborn.

## Centralized Baseline

Run the first-layer non-federated baseline before FedAvg:

```powershell
& 'C:\Users\8U7HYBBY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\train_centralized_baseline.py --seed 42 --epochs 500
```

Generated outputs:

- `reports/centralized_baseline_results.csv`: train/validation/test metrics for majority-class and weighted logistic regression baselines.
- `reports/centralized_threshold_tuning.csv`: validation threshold sweep used to choose the F1-oriented decision threshold.
- `reports/centralized_training_history.csv`: weighted logistic regression training loss.
- `data/processed/centralized_test_predictions.csv`: test-set prediction probabilities and labels.
- `data/processed/centralized_logistic_model.json`: learned coefficients and threshold.
- `figures/centralized_baseline_metrics.png`: metric comparison.
- `figures/centralized_confusion_matrix.png`: test confusion matrix.
- `figures/centralized_training_curve.png`: training loss curve.
- `figures/centralized_threshold_curve.png`: precision/recall/F1 threshold tuning curve.

The failure-mode indicator columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are intentionally excluded from baseline features to avoid target leakage.

## FedAvg Baseline

Run the standard federated baseline after the centralized baseline:

```powershell
& 'C:\Users\8U7HYBBY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\train_fedavg_baseline.py --rounds 50 --local-epochs 5 --seed 42
```

Baseline definition:

- Uses the same leakage-safe feature set as the centralized baseline.
- Runs on `iid`, `moderate_non_iid`, and `highly_non_iid` factory splits.
- Treats each `factory_XX.csv` as one federated client.
- Uses all clients in every communication round.
- Aggregates client model parameters with sample-size weighted FedAvg.
- Uses a shared global train/validation/test split; validation and test samples are removed from client training data.

Generated outputs:

- `reports/fedavg_baseline_results.csv`: final test metrics for all three data distributions.
- `reports/fedavg_training_history.csv`: per-round validation/test metrics and mean client loss.
- `reports/fedavg_threshold_tuning.csv`: validation threshold sweep for each distribution.
- `data/processed/fedavg_models.json`: final global model coefficients and thresholds.
- `figures/fedavg_convergence.png`: validation F1 by communication round.
- `figures/fedavg_client_loss.png`: mean client loss by communication round.
- `figures/fedavg_final_metrics.png`: final test metric comparison.
