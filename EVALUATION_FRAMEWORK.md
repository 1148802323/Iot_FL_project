# Standalone Evaluation Framework for Federated Predictive Maintenance

## 1. Overview

This repository provides a standalone evaluator for comparing federated learning (FL) algorithms in an imbalanced predictive-maintenance setting, with particular emphasis on statistical heterogeneity (Non-IID data) across factories.

The evaluator is implemented as one independent file:

```text
standalone_non_iid_evaluator.py
```

It does **not** import, modify, or depend on a teammate's source code. Each algorithm is executed separately by its owner and supplies prediction probabilities in a standard CSV format. The standalone evaluator then applies the same validation, testing, metric, client-level, and statistical procedures to every submission.

This design follows the general benchmarking principles of standardized protocols, realistic client partitions, modularity, reproducibility, and multi-dimensional FL evaluation emphasized by NIID-Bench [1], FLamby [2], FedScale [3], and OARF [4].

## 2. Scope and Important Limitation

The evaluator can compare algorithms implemented with NumPy, PyTorch, TensorFlow, Flower, or another framework because it consumes predictions rather than internal model objects.

However, an external evaluator cannot assess an arbitrary program that exposes neither predictions nor a trained model. Therefore, the universal hand-off artifact is a CSV containing positive-class probabilities. This requirement does not constrain or modify the algorithm itself; it only standardizes its observable output.

Prediction files are treated as trusted research artifacts. The evaluator verifies row identities, splits, ranges, duplicates, and completeness, but it cannot prove that a teammate did not train on held-out test observations. Experimental integrity therefore still requires a shared team protocol.

## 3. Prediction CSV Contract

Each algorithm must supply a CSV with the following required columns:

| Column | Type | Description |
|---|---:|---|
| `UDI` | integer | Unique AI4I observation identifier. |
| `seed` | integer | Random seed used for the experiment. |
| `strategy` | string | `iid`, `moderate_non_iid`, or `highly_non_iid`. |
| `split` | string | `validation` or `test`. |
| `probability` | float | Predicted probability of machine failure, in `[0, 1]`. |

Example:

```csv
UDI,seed,strategy,split,probability
123,42,iid,validation,0.1734
456,42,iid,validation,0.8127
789,42,iid,test,0.2911
101,42,iid,test,0.9342
```

An optional `round` column may contain positive communication-round numbers for validation predictions. This enables convergence analysis. Rows containing final validation and test predictions must have a blank or zero `round` value.

The evaluator rejects:

- missing required columns;
- missing or unexpected UDIs;
- duplicate UDIs within one run and split;
- probabilities outside `[0, 1]`;
- NaN or infinite probabilities;
- unknown split names; and
- incomplete seed/strategy combinations.

## 4. Preparing a Prediction Request

Generate the exact rows that algorithm owners must predict:

```powershell
python standalone_non_iid_evaluator.py --make-request-template prediction_request.csv
```

The generated file contains the required `UDI`, `seed`, `strategy`, and `split` values. Algorithm owners fill the `probability` column using their trained models.

The default experiment uses five paired seeds:

```text
42, 52, 62, 72, 82
```

For a final research report, at least ten paired seeds are preferable when computationally feasible.

## 5. Running the Evaluation

Compare a candidate against FedAvg:

```powershell
python standalone_non_iid_evaluator.py --prediction fedavg=predictions/fedavg.csv --prediction proposed=predictions/proposed.csv --baseline fedavg
```

Additional algorithms can be included by repeating `--prediction`:

```powershell
--prediction fedprox=predictions/fedprox.csv
```

Run the internal contract test:

```powershell
python standalone_non_iid_evaluator.py --self-test
```

## 6. Evaluation Protocol

For each seed, the evaluator reconstructs a stratified split:

- 60% training;
- 20% validation; and
- 20% held-out testing.

The classification threshold is selected **only** on the validation split by maximizing F1, with Recall and Precision used as tie-breakers. That threshold is then locked before the held-out test metrics are calculated.

Algorithms compared under one seed must use the same data partition. Paired comparisons are consequently made seed by seed rather than between unrelated runs.

## 7. Why Accuracy Is Not a Primary Metric

The AI4I data used in this project contain 339 machine-failure observations among 10,000 rows, corresponding to a failure prevalence of approximately 3.39%. A trivial classifier that always predicts “no failure” would therefore obtain approximately 96.61% Accuracy while achieving zero failure Recall.

Recent work on class-imbalanced FL argues that ordinary Accuracy can produce misleading interpretations and motivates Balanced Accuracy when class distributions are skewed [5]. Recent predictive-maintenance work similarly emphasizes rare-failure Recall, F-measures, and precision-recall analysis [8, 9].

Accuracy is reported for completeness, but it is not used as the principal measure of algorithm quality.

## 8. Global Predictive Metrics

### 8.1 Recall

Recall measures the fraction of actual failures detected by the model:

```math
\mathrm{Recall}=\frac{TP}{TP+FN}.
```

Recall is operationally important because a false negative represents a missed failure. In predictive maintenance, missed failures may lead to unplanned downtime, secondary damage, or safety risk.

### 8.2 Precision

Precision measures the fraction of predicted failures that are genuine:

```math
\mathrm{Precision}=\frac{TP}{TP+FP}.
```

Precision reflects false-alarm burden and unnecessary inspection cost.

### 8.3 F1-score

F1 is the harmonic mean of Precision and Recall:

```math
F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}.
```

It evaluates the selected deployment threshold while balancing missed failures and false alarms.

### 8.4 F2-score

F2 gives Recall greater weight than Precision:

```math
F_2=5\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{4\cdot\mathrm{Precision}+\mathrm{Recall}}.
```

The use of F2 is appropriate when failure detection is more important than avoiding an additional inspection. A recent industrial predictive-maintenance evaluation explicitly discusses F2 for settings in which Recall carries greater operational value [8].

### 8.5 PR-AUC / Average Precision

PR-AUC evaluates the ranking of positive examples across decision thresholds. It is included because this project is dominated by negative observations and the target of interest is the rare positive class. It complements the threshold-specific F1 and F2 measures.

### 8.6 Balanced Accuracy

Balanced Accuracy gives equal importance to positive-class Recall and negative-class Specificity:

```math
\mathrm{Balanced\ Accuracy}=\frac{\mathrm{Recall}+\mathrm{Specificity}}{2}.
```

FairTrade uses Balanced Accuracy when assessing FL under class imbalance because ordinary Accuracy may be dominated by the majority class [5].

## 9. Non-IID Evaluation

FL evaluation based only on a pooled global test score can hide severe performance degradation at individual clients. This is especially important in cross-silo FL, in which a small number of institutions or factories may have different label prevalence, equipment populations, and failure mechanisms.

The project therefore evaluates three client distributions:

1. `iid`;
2. `moderate_non_iid`; and
3. `highly_non_iid`.

NIID-Bench demonstrates the importance of systematically evaluating FL algorithms across different forms and strengths of Non-IID partitioning [1]. FLamby further supports natural client partitions and standardized cross-silo evaluation [2].

### 9.1 Client-macro metrics

The evaluator calculates each factory's Recall, F1, PR-AUC, and Balanced Accuracy, then gives every factory equal weight:

```math
\mathrm{MacroClientMetric}=\frac{1}{K}\sum_{k=1}^{K}m_k.
```

This prevents a large factory from dominating the evaluation.

### 9.2 Worst-client metrics

For each metric, the evaluator reports:

```math
\mathrm{WorstClientMetric}=\min_k m_k.
```

Worst-client evaluation checks whether a higher global score was achieved by sacrificing one factory. Recent fair-FL research explicitly evaluates client disparity and worst-performing client performance [6], while heterogeneity-aware fair FL studies performance dispersion across clients [7].

### 9.3 Client dispersion

The evaluator reports:

- standard deviation across clients; and
- the best-client minus worst-client gap.

Lower dispersion indicates more consistent service across heterogeneous factories. It does not replace predictive utility: a uniformly poor model may have low dispersion, so fairness and predictive performance must be interpreted together.

### 9.4 Data-heterogeneity descriptors

For each factory partition, the evaluator reports:

- client quantity coefficient of variation;
- client failure-rate standard deviation;
- client failure-rate range; and
- mean pairwise Jensen-Shannon divergence between binary label distributions.

These quantities describe the evaluation setting. They are not algorithm-performance metrics and must not be interpreted as evidence that one model is superior.

## 10. Stability, Convergence, and Paired Inference

When optional round-level validation probabilities are supplied, convergence is defined as the first communication round reaching 95% of that run's best validation F1 at threshold 0.5.

The evaluator also reports:

- mean and standard deviation across seeds;
- the number of completed runs;
- paired candidate-minus-FedAvg differences; and
- paired bootstrap 95% confidence intervals.

FedScale emphasizes realistic heterogeneity, reproducible protocols, and system behavior [3]. OARF argues that FL benchmarks should assess multiple dimensions, including predictive quality, communication cost, throughput, and convergence time [4]. This evaluator records communication rounds. Actual transferred bytes should be added when algorithm implementations expose reliable upload/download measurements.

## 11. Non-IID Evidence Scorecard

The file `standalone_non_iid_scorecard.csv` separately evaluates Moderate and Highly Non-IID settings using four core comparisons against FedAvg:

1. global F1 difference;
2. global PR-AUC difference;
3. client-macro F1 difference; and
4. worst-client F1 difference.

The labels are intentionally conservative:

- **`supported_improvement`**: all four mean differences are non-negative and at least two bootstrap confidence intervals have lower bounds above zero;
- **`promising_but_inconclusive`**: all four means are non-negative, but the confidence-interval evidence is insufficient; and
- **`tradeoff_or_not_supported`**: at least one core dimension decreases.

This scorecard is an evidence summary, not a substitute for research judgment. Recall, F2, convergence, uncertainty, and operational costs must also be considered.

## 12. Generated Outputs

| File | Purpose |
|---|---|
| `standalone_raw.csv` | One global result per algorithm, seed, and strategy. |
| `standalone_client_metrics.csv` | Per-factory test metrics. |
| `standalone_summary.csv` | Mean, standard deviation, and count. |
| `standalone_history.csv` | Optional round-level validation metrics. |
| `standalone_paired_bootstrap.csv` | Paired differences and 95% intervals against the baseline. |
| `standalone_non_iid_scorecard.csv` | Conservative Non-IID evidence labels. |
| `standalone_errors.csv` | Missing rows, invalid probabilities, duplicates, and other failures. |
| `standalone_manifest.json` | Data paths, algorithms, seeds, strategies, and evaluation policy. |

## 13. Reproducibility Checklist

Before comparing algorithms, confirm that:

- every algorithm used the same requested UDI split;
- validation and test probabilities are complete;
- test labels were not used for training, model selection, or threshold selection;
- the same local epochs, communication-round budget, and client-participation policy were used when the research question requires controlled comparison;
- hyperparameter search was performed on training/validation data only;
- FedAvg and any other baselines received a comparable tuning budget;
- results are reported over paired seeds; and
- failures and incomplete runs are disclosed rather than silently removed.

## 14. Threats to Validity

1. **Prediction provenance:** The evaluator cannot cryptographically prove how a prediction file was generated.
2. **Limited seeds:** Five seeds provide preliminary uncertainty estimates; more repetitions improve statistical reliability.
3. **Synthetic partitions:** IID and controlled Non-IID factory splits may not represent every real deployment. Natural factory partitions are preferable when available [2].
4. **Threshold objective:** Maximizing validation F1 may not match the actual cost of missed failures and inspections. A cost-sensitive threshold should be used when reliable cost estimates exist.
5. **Client support:** Client metrics become unstable when a test client has very few positive failures. Positive support is therefore reported with each client result.
6. **Communication measurement:** Communication rounds are only a proxy for network cost unless serialized update sizes are also measured.

## 15. References

[1] Q. Li, Y. Diao, Q. Chen, and B. He, “Federated Learning on Non-IID Data Silos: An Experimental Study,” *2022 IEEE 38th International Conference on Data Engineering (ICDE)*, pp. 965–978, 2022. [https://doi.org/10.1109/ICDE53745.2022.00077](https://doi.org/10.1109/ICDE53745.2022.00077)

[2] J. Ogier du Terrail et al., “FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings,” *Advances in Neural Information Processing Systems 35*, Datasets and Benchmarks Track, 2022. [Official proceedings page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html)

[3] F. Lai, Y. Dai, S. Singapuram, J. Liu, X. Zhu, H. Madhyastha, and M. Chowdhury, “FedScale: Benchmarking Model and System Performance of Federated Learning at Scale,” *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162:11814–11827, 2022. [https://proceedings.mlr.press/v162/lai22a.html](https://proceedings.mlr.press/v162/lai22a.html)

[4] S. Hu, Y. Li, X. Liu, Q. Li, Z. Wu, and B. He, “The OARF Benchmark Suite: Characterization and Implications for Federated Learning Systems,” *ACM Transactions on Intelligent Systems and Technology*, vol. 13, no. 4, Article 63, 2022. [https://doi.org/10.1145/3510540](https://doi.org/10.1145/3510540)

[5] M. Badar, S. Sikdar, W. Nejdl, and M. Fisichella, “FairTrade: Achieving Pareto-Optimal Trade-Offs between Balanced Accuracy and Fairness in Federated Learning,” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 38, no. 10, pp. 10962–10970, 2024. [https://doi.org/10.1609/aaai.v38i10.28971](https://doi.org/10.1609/aaai.v38i10.28971)

[6] F. Zhang, Z. Shuai, K. Kuang, F. Wu, Y. Zhuang, and J. Xiao, “Unified Fair Federated Learning for Digital Healthcare,” *Patterns*, vol. 5, no. 1, 100907, 2024. [https://doi.org/10.1016/j.patter.2023.100907](https://doi.org/10.1016/j.patter.2023.100907)

[7] X. Li, S. Zhao, C. Chen, and Z. Zheng, “Heterogeneity-Aware Fair Federated Learning,” *Information Sciences*, vol. 619, pp. 968–986, 2023. [https://doi.org/10.1016/j.ins.2022.11.031](https://doi.org/10.1016/j.ins.2022.11.031)

[8] A. Giannoulidis, A. Gounaris, A. Naskos, N. Nikolaidis, et al., “Engineering and Evaluating an Unsupervised Predictive Maintenance Solution: A Cold-Forming Press Case-Study,” *Journal of Intelligent Manufacturing*, vol. 36, pp. 2121–2139, published online 2024, volume year 2025. [https://doi.org/10.1007/s10845-024-02352-z](https://doi.org/10.1007/s10845-024-02352-z)

[9] M. Alnahhal, M. I. Tabash, S. K. Safi, M. S. M. Al-Absy, and Z. Mamadiyarov, “A Comparative Study of Imbalance-Handling Methods in Multiclass Predictive Maintenance,” *Computation*, vol. 14, no. 4, article 88, 2026. [https://doi.org/10.3390/computation14040088](https://doi.org/10.3390/computation14040088)
