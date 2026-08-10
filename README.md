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

## Proposed Method V1: Failure-Aware FedAvg

Run the first proposed aggregation method:

```powershell
& 'C:\Users\8U7HYBBY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\train_failure_aware_fedavg.py --rounds 50 --local-epochs 5 --seed 42 --alpha 1.0
```

Aggregation rule:

```text
aggregation_weight = client_samples * (1 + alpha * client_failure_rate / global_failure_rate)
```

Interpretation:

- `alpha = 0.0` is equivalent to standard sample-size FedAvg.
- Larger `alpha` gives more aggregation weight to clients with above-average failure rates.
- This is designed for imbalanced predictive maintenance data where failure cases are rare but important.

Generated outputs:

- `reports/failure_aware_fedavg_results.csv`: final test metrics for the V1 method.
- `reports/failure_aware_fedavg_history.csv`: per-round validation/test metrics and client loss.
- `reports/failure_aware_fedavg_threshold_tuning.csv`: validation threshold sweep.
- `reports/failure_aware_vs_fedavg_results.csv`: comparison against the standard FedAvg baseline.
- `data/processed/failure_aware_fedavg_models.json`: final global model coefficients, thresholds, and aggregation metadata.
- `figures/failure_aware_fedavg_convergence.png`: validation F1 by communication round.
- `figures/failure_aware_fedavg_client_loss.png`: mean client loss by communication round.
- `figures/failure_aware_fedavg_final_metrics.png`: final V1 metric comparison across IID/Non-IID settings.
- `figures/failure_aware_vs_fedavg_metrics.png`: standard FedAvg vs failure-aware FedAvg comparison.

## FastAPI Backend, Database, And Login

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a local `.env` file from `.env.example` and replace `JWT_SECRET` before
using the API outside local development. The default SQLite database is
`iot_fl_app.db`; it is generated locally and ignored by Git.

Run the API:

```bash
PYTHONPATH=src uvicorn iot_fl.backend.main:app --reload
```

The backend creates the `users`, `factories`, and `clients` tables on startup.
It scans the existing IID, moderate Non-IID, and highly Non-IID factory CSVs
under `data/factories/` and upserts one client record per factory/distribution
pair. The seed is repeatable and refreshes row/failure metadata without
duplicating records. Available authentication and client-management endpoints:

- `GET /api/health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`
- `GET /api/clients`
- `GET /api/clients/{client_id}`
- `GET /api/clients/{client_id}/statistics`
- `GET /api/clients/{client_id}/experiments`
- `GET /api/admin/users`

Client users must be registered with a valid `factory_id`; admin users must not
be bound to a factory. Admin users can view all clients and their
`dataset_path`; client users can view only their own factory's clients and do
not receive `dataset_path`. Client statistics are computed from the stored CSV
metadata and the existing CSV files; experiment summaries reuse the generated
FedAvg and Failure-Aware result CSVs.

To initialize or refresh factory/client records without starting the API:

```bash
PYTHONPATH=src python -m iot_fl.backend.seed
```

The static website includes a Login section that calls these endpoints when the
FastAPI service is running.

Serve the static website from another terminal:

```bash
python -m http.server 8080 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8080/site/`. The Login section calls the API at
`http://127.0.0.1:8000` by default. The site is served from the project root so
that it can load `reports/`, `figures/`, and `data/processed/` correctly.

Run the backend tests:

```bash
PYTHONPATH=src pytest -q tests/test_backend_auth.py
```
