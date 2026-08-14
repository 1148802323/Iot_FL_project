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

## Flower Framework

The original FedAvg workflow has been migrated to a modular Flower-based framework while preserving the numerical behaviour of the existing baseline implementation.

The framework separates client execution, server orchestration, configuration, and aggregation strategy selection. Failure-aware methods are implemented as pluggable strategy classes on top of a shared failure-aware aggregation base, allowing multiple algorithm variants to be integrated without redesigning the core Flower execution workflow.

### Architecture

```text
run_flower_simulation.py
        |
        v
FrameworkConfig
        |
        v
ServerApp
        |
        v
Strategy Factory
        |
        +-- FedAvg
        |
        +-- FailureAwareV1Strategy
        |
        +-- FailureAwareV4Strategy
                |
                +-- fixed
                +-- linear
                +-- recall_adaptive
```

The main Flower framework components are located under:

```text
src/iot_fl/flower_framework/
├── flower_client.py
├── client_app.py
├── server_app.py
├── config.py
└── strategies/
    ├── __init__.py
    ├── base_failure_aware.py
    ├── failure_aware_v1.py
    ├── failure_aware_v4.py
    └── strategy_factory.py
```

`BaseFailureAwareStrategy` contains the shared failure-aware aggregation workflow, including client metadata validation and weighted parameter aggregation. Individual failure-aware variants inherit from this base and implement their own weighting or round-dependent logic.

### Run a Flower Simulation

Activate the Flower virtual environment:

```powershell
.\.venv_flower\Scripts\Activate.ps1
```

The simulation entry point adds the project `src` directory to the Python import path. For direct test execution, `PYTHONPATH` can also be set explicitly:

```powershell
$env:PYTHONPATH="src"
```

Run the standard FedAvg strategy:

```powershell
python run_flower_simulation.py --aggregation fedavg
```

A shorter simulation can be used for development and smoke testing:

```powershell
python run_flower_simulation.py --aggregation fedavg --rounds 1 --local-epochs 1
```

Run Failure-Aware V1:

```powershell
python run_flower_simulation.py --aggregation failure_aware_v1 --alpha 1.0 --rounds 2
```

Run Failure-Aware V4 with a fixed schedule:

```powershell
python run_flower_simulation.py --aggregation failure_aware_v4 --v4-schedule fixed --v4-lambda-max 2.0 --rounds 2
```

Run Failure-Aware V4 with a linear schedule:

```powershell
python run_flower_simulation.py --aggregation failure_aware_v4 --v4-schedule linear --v4-lambda-max 2.0 --rounds 2
```

Run Failure-Aware V4 with recall-adaptive scheduling:

```powershell
python run_flower_simulation.py --aggregation failure_aware_v4 --v4-schedule recall_adaptive --v4-lambda-max 2.0 --v4-target-recall 0.85 --v4-eta 0.25 --rounds 3
```

### Aggregation Strategies

The framework currently exposes three aggregation strategy options:

- `fedavg`: standard Flower FedAvg aggregation.
- `failure_aware_v1`: Failure-Aware V1 aggregation.
- `failure_aware_v4`: Failure-Aware V4 aggregation with configurable lambda scheduling.

#### Failure-Aware V1

V1 uses the aggregation rule:

```text
aggregation_weight =
    client_samples
    * (1 + alpha * client_failure_rate / global_failure_rate)
```

Setting `alpha = 0` reduces V1 to standard sample-size weighted FedAvg.

#### Failure-Aware V4

V4 uses the aggregation rule:

```text
aggregation_weight =
    client_samples
    * (1 + current_lambda * client_failure_rate)
```

The current implementation supports three lambda schedules:

- `fixed`: uses `lambda_max` throughout training.
- `linear`: increases lambda according to the current communication round and total number of rounds.
- `recall_adaptive`: updates lambda using centralized validation recall, `target_recall`, and `eta`, while constraining lambda to the configured range.

Setting `lambda_max = 0` reduces V4 to standard sample-size weighted FedAvg.

### Failure-Aware Strategy Integration

The failure-aware strategy hierarchy is:

```text
BaseFailureAwareStrategy
        |
        +-- FailureAwareV1Strategy
        |
        +-- FailureAwareV4Strategy
```

The shared base handles the common aggregation workflow. Strategy-specific subclasses provide the algorithm-dependent behaviour through extension hooks such as round preparation and client-weight calculation.

The server selects the requested strategy through `strategy_factory.py`. Strategy-specific parameters are supplied through `FrameworkConfig` and forwarded by `server_app.py`, keeping the core Flower client/server workflow independent from the individual aggregation algorithms.

Additional failure-aware variants can therefore be added as new strategy subclasses without replacing the existing V1 or V4 implementations.

### Framework Tests

Run the complete framework test suite with:

```powershell
$env:PYTHONPATH="src"
python -m pytest -v
```

The current automated tests cover:

- framework configuration validation;
- aggregation strategy factory behaviour;
- shared failure-aware metadata validation;
- Failure-Aware V1 weighting and aggregation behaviour;
- Failure-Aware V4 fixed and linear scheduling;
- Failure-Aware V4 recall-adaptive scheduling and lambda bounds;
- FedAvg fallback behaviour for disabled failure-aware weighting.

The complete test suite passes after the V1 and V4 integrations.

### Baseline and Strategy Consistency Validation

The Flower implementation can be compared against the original FedAvg baseline using:

```powershell
python validate_baseline_consistency.py --strategy iid
```

The current regression validation passes all checks:

- single-client training consistency: PASS;
- FedAvg aggregation consistency: PASS;
- validation metrics consistency: PASS;
- Failure-Aware V1 with `alpha = 0` versus FedAvg: PASS;
- Failure-Aware V4 with `lambda = 0` versus FedAvg: PASS;
- overall consistency validation: PASS.

The maximum observed global parameter difference in the validated FedAvg-equivalent paths is approximately:

```text
1.39e-17
```

Validation metric differences for the original baseline versus the Flower FedAvg implementation are zero for loss, accuracy, precision, recall, and F1.

These results confirm that the Flower migration preserves the numerical behaviour of the original FedAvg baseline within floating-point precision, and that both integrated failure-aware strategies correctly recover standard FedAvg behaviour when their additional weighting is disabled.

### Current Integration Status

```text
FedAvg
└── integrated and validated

FailureAwareV1Strategy
└── integrated, simulated, tested, and consistency-validated

FailureAwareV4Strategy
├── fixed              integrated and tested
├── linear             integrated and tested
└── recall_adaptive    integrated and tested

FailureAwareV2Strategy
└── pending integration

FailureAwareV3Strategy
└── pending integration
```

V1 and V4 are treated as frozen integration paths. Future failure-aware variants should be added through the existing strategy architecture rather than by modifying the completed V1/V4 implementations or redesigning the Flower framework.

