# Unified Federated Learning Algorithm Interface

The unified interface lives in `src/iot_fl/algorithms/` and wraps the existing
research scripts without changing their command-line behavior.

## Algorithm Names

- `fedavg`: `src/train_fedavg_baseline.py`
- `failure_aware_v1`: `src/train_failure_aware_fedavg.py`
- `failure_aware_v2`: `src/train_failure_count_fedavg_variant2_clean.py`
- `dynamic_failure_aware`: `src/train_dynamic_failure_aware_variant4.py`

## Distributions

- `iid`
- `moderate_non_iid`
- `highly_non_iid`

## Running an Experiment

```python
from iot_fl.algorithms import run_experiment

result = run_experiment(
    algorithm="fedavg",
    distribution="highly_non_iid",
    config={
        "rounds": 5,
        "local_epochs": 1,
        "learning_rate": 0.01,
    },
)
```

The caller can swap `algorithm` to any registered name without adding
algorithm-specific branching.

Common config keys are `rounds`, `local_epochs`, and `learning_rate`. The
interface also accepts `seed`, `l2`, `data_path`, and `factory_root`.
Algorithm-specific optional keys are passed through the shared `config`
dictionary. V1/V2 accept `alpha`; the dynamic adapter accepts `schedule`,
`lambda_max`, `target_recall`, and `eta`.

## Result Format

Every adapter returns at least:

```python
{
    "algorithm": "fedavg",
    "distribution": "highly_non_iid",
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "communication_cost": 0,
    "training_time": 0.0,
    "rounds": 5,
    "convergence_history": [],
}
```

When the underlying script exposes additional safe metadata, the adapter keeps
it under keys such as `method`, `threshold`, `communication_client_updates`,
`communication_sample_updates`, `raw_final`, and `model`.

## Adding Another Algorithm

1. Add a new adapter in `src/iot_fl/algorithms/adapters/`.
2. Subclass `FederatedAlgorithm` and implement `_run(distribution, config)`.
3. Call the existing research implementation from the adapter.
4. Normalize the output with `normalize_strategy_payload()` or
   `ensure_standard_result()`.
5. Register the adapter in `ALGORITHM_REGISTRY` in
   `src/iot_fl/algorithms/registry.py`.

