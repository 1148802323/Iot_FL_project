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

## Proposed Method V4: Dynamic Failure-Aware FedAvg

This contribution adds a standalone fourth aggregation method without modifying
the existing FedAvg, V1, or V2 implementation files.

Part 4 assigns client `i` the round-dependent aggregation score:

```text
score_i(t) = n_i * (1 + lambda_t * r_i)
```

where:

- `n_i` is the number of local training samples;
- `r_i` is the local machine-failure ratio; and
- `lambda_t` controls the strength of failure-aware weighting at round `t`.

The default linear schedule is:

```text
lambda_t = lambda_max * t / T
```

The implementation also provides two ablation alternatives:

- `fixed`: uses `lambda_max` in every round;
- `linear`: gradually increases the failure-aware contribution; and
- `recall_adaptive`: updates lambda using the validation-recall gap, bounded by
  zero and `lambda_max`.

Run the default Part 4 experiment over five paired seeds and all three client
distributions:

```powershell
python src/train_dynamic_failure_aware_variant4.py --schedule linear
```

Run the schedule ablation:

```powershell
python src/train_dynamic_failure_aware_variant4.py --schedule fixed --output-dir part4_outputs/fixed
python src/train_dynamic_failure_aware_variant4.py --schedule linear --output-dir part4_outputs/linear
python src/train_dynamic_failure_aware_variant4.py --schedule recall_adaptive --output-dir part4_outputs/adaptive
```

Quick smoke test:

```powershell
python src/train_dynamic_failure_aware_variant4.py --seeds 42 --strategies iid --rounds 2 --local-epochs 1 --output-dir smoke_part4
```

Part 4 generates:

- `part4_predictions.csv`: validation, test, and optional round-level
  probabilities in the standalone evaluator contract;
- `part4_training_history.csv`: lambda, validation Recall, and mean client loss
  by round;
- `part4_models.json`: learned model coefficients and schedule metadata; and
- `part4_manifest.json`: seeds, strategies, hyperparameters, and experiment
  settings.

## Standalone Multi-Algorithm Evaluation Framework

The standalone evaluator compares algorithms through their observable
prediction probabilities. It does not import or modify algorithm owners' source
files, so implementations based on NumPy, PyTorch, TensorFlow, or Flower can be
evaluated under one protocol.

Each algorithm supplies a CSV with these required columns:

| Column | Description |
|---|---|
| `UDI` | Unique AI4I observation identifier. |
| `seed` | Paired experimental seed. |
| `strategy` | `iid`, `moderate_non_iid`, or `highly_non_iid`. |
| `split` | `validation` or `test`. |
| `probability` | Positive-class probability in `[0, 1]`. |

An optional `round` column enables convergence analysis. Final validation and
test predictions use a blank or zero round value.

### Evaluation protocol

- The default experiment uses seeds `42, 52, 62, 72, 82`.
- Each seed reconstructs the same stratified 60%/20%/20%
  train/validation/test split for every algorithm.
- The decision threshold is selected only on validation data by maximizing F1,
  then locked before held-out test evaluation.
- Algorithms are compared seed by seed rather than between unrelated runs.
- Prediction inputs are checked for missing columns, invalid probability
  ranges, unexpected or duplicate UDIs, unknown splits, and incomplete runs.

### Reported dimensions

Global predictive metrics:

- Accuracy, Precision, Recall, Specificity;
- F1 and Recall-oriented F2;
- PR-AUC / Average Precision; and
- Balanced Accuracy.

Client-level and Non-IID metrics:

- client-macro Recall, F1, PR-AUC, and Balanced Accuracy;
- worst-client performance;
- standard deviation and best-minus-worst client gap; and
- client quantity variation, failure-rate variation, and label-distribution
  Jensen-Shannon divergence.

Statistical and convergence outputs:

- mean, standard deviation, and completed-run count;
- paired candidate-minus-FedAvg differences;
- paired bootstrap 95% confidence intervals;
- convergence round when round-level predictions are supplied; and
- a conservative Non-IID evidence scorecard.

### Cost-sensitive predictive-maintenance evaluation

The framework supplements F1-oriented evaluation with a scenario-based cost
model:

```text
C = C_FN * FN + C_FP * FP
```

The cost-optimal threshold is selected on validation data and locked before test
evaluation. Configure the assumptions with:

```powershell
--false-negative-cost 10 --false-positive-cost 1
```

Ratios such as 5:1, 10:1, and 20:1 should be described as sensitivity scenarios
unless verified factory financial data are available. The evaluator reports
total cost and cost per 1,000 observations at both the F1-oriented and
cost-oriented thresholds.

### Running the unified comparison

After all algorithm owners provide prediction CSV files, run:

```powershell
python standalone_non_iid_evaluator.py `
  --prediction fedavg=predictions/fedavg.csv `
  --prediction v1=predictions/v1.csv `
  --prediction v2=predictions/v2.csv `
  --prediction part4=part4_outputs/part4_predictions.csv `
  --baseline fedavg `
  --false-negative-cost 10 `
  --false-positive-cost 1
```

Generated evaluation outputs:

- `standalone_raw.csv`;
- `standalone_client_metrics.csv`;
- `standalone_summary.csv`;
- `standalone_history.csv`;
- `standalone_paired_bootstrap.csv`;
- `standalone_non_iid_scorecard.csv`;
- `standalone_errors.csv`; and
- `standalone_manifest.json`.

Run the automated evaluator tests:

```powershell
python -m unittest discover -s tests -v
python standalone_non_iid_evaluator.py --self-test
```

### Reproducibility and validity notes

- All algorithms must use the requested UDI splits and a comparable training
  and tuning budget.
- Test labels must not be used for training, schedule adaptation, threshold
  selection, or hyperparameter tuning.
- Five seeds provide preliminary uncertainty estimates; ten paired seeds are
  preferable when computationally feasible.
- Prediction provenance remains a team research-integrity responsibility.
- Synthetic client partitions may not represent every real factory deployment.
- Communication rounds are only a proxy for network cost unless serialized
  upload and download bytes are measured.

### Contribution boundary

The Part 4 and evaluation contribution consists of:

- new: `src/train_dynamic_failure_aware_variant4.py`;
- extended: `standalone_non_iid_evaluator.py`;
- new: `tests/test_evaluation_framework.py`; and
- extended: this `README.md`.

The existing centralized, FedAvg, V1, and V2 training scripts remain unchanged.
