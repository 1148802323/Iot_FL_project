# Part 4 and Evaluation Framework Extension

This contribution is intentionally isolated from the existing FedAvg, V1 and
V2 source files. It adds one new algorithm file and extends only the standalone
evaluation framework.

## Part 4: Dynamic Failure-Aware FedAvg

Part 4 uses the aggregation score

```text
score_i(t) = n_i * (1 + lambda_t * r_i)
```

where `n_i` is the client sample count and `r_i` is its local failure rate.
The default linear schedule is `lambda_t = lambda_max * t / T`.

Run the default five-seed experiment:

```powershell
python src/train_dynamic_failure_aware_variant4.py --schedule linear
```

Ablation schedules:

```powershell
python src/train_dynamic_failure_aware_variant4.py --schedule fixed --output-dir part4_outputs/fixed
python src/train_dynamic_failure_aware_variant4.py --schedule linear --output-dir part4_outputs/linear
python src/train_dynamic_failure_aware_variant4.py --schedule recall_adaptive --output-dir part4_outputs/adaptive
```

For a quick smoke test:

```powershell
python src/train_dynamic_failure_aware_variant4.py --seeds 42 --strategies iid --rounds 2 --local-epochs 1
```

## Unified evaluation

Part 4 directly creates `part4_predictions.csv` in the evaluator contract.
After the other algorithm owners provide their prediction CSV files, run:

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

The cost values are scenario assumptions, not claimed factory financial data.
Report sensitivity for multiple ratios such as 5:1, 10:1 and 20:1.

## Ownership boundary

- New: `src/train_dynamic_failure_aware_variant4.py`
- Modified: `standalone_non_iid_evaluator.py`
- New: `tests/test_evaluation_framework.py`
- New: this guide
- Unchanged: existing centralized, FedAvg, V1 and V2 training files
