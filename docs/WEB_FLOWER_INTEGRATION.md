# Web–Flower Integration

## Status

Web–Flower integration is complete and validated.

Checkpoint commit:

`f9570da feat: integrate web app with Flower framework`

Validated algorithms:

- FedAvg
- Failure-Aware FedAvg V1
- Failure-Aware FedAvg V2
- Dynamic Failure-Aware FedAvg (V4)

Failure-Aware V3 is intentionally excluded from the current integration.

---

## Architecture

The Web application does not invoke Flower directly.

The execution path is:

Browser
→ FastAPI Backend
→ Experiment Service
→ Algorithm Adapter
→ Flower Integration Adapter
→ Flower Framework
→ Aggregation Strategy
→ Evaluation
→ Integration Adapter
→ Experiment Database
→ FastAPI
→ Browser

The main integration boundary is:

`src/iot_fl/integration/flower_adapter.py`

This keeps the Web application independent from Flower-specific execution details.

---

## Algorithm Mapping

Web algorithm names are mapped to Flower aggregation strategies as follows:

| Web Algorithm | Flower Aggregation |
|---|---|
| `fedavg` | `fedavg` |
| `failure_aware_v1` | `failure_aware_v1` |
| `failure_aware_v2` | `failure_aware_v2` |
| `dynamic_failure_aware` | `failure_aware_v4` |

V3 is not currently registered.

---

## Algorithm Parameters

### FedAvg

No additional algorithm-specific parameters.

### Failure-Aware V1

- `alpha`

### Failure-Aware V2

- `alpha`

### Dynamic Failure-Aware V4

- `schedule`
- `lambda_max`
- `target_recall`
- `eta`

The Web interface sends these values through the experiment `parameters` field.

The algorithm adapters translate the Web configuration into the corresponding Flower framework configuration.

---

## Result Flow

Centralized evaluation results are captured through an optional result callback in:

`src/iot_fl/flower_framework/server_app.py`

The callback does not modify aggregation or training behaviour.

The integration layer converts Flower execution results into the common experiment result format used by the Web backend.

Stored experiment outputs include:

- Accuracy
- Precision
- Recall
- F1 score
- Communication cost
- Training time
- Convergence history
- Experiment status
- Error information

---

## Validation

The integration was validated at multiple levels.

### Framework

Existing Flower framework and strategy tests pass.

### Service Integration

Backend-to-Flower execution was validated for:

- FedAvg
- Failure-Aware V1
- Failure-Aware V2
- Dynamic Failure-Aware V4

### Parameter Propagation

Validated:

- V1 `alpha`
- V2 `alpha`
- V4 `schedule`
- V4 `lambda_max`
- V4 `target_recall`
- V4 `eta`

### HTTP Integration

A complete FedAvg experiment was executed through the FastAPI HTTP interface:

HTTP
→ Backend
→ Flower
→ Database
→ HTTP result

### Browser Integration

A complete Failure-Aware V1 experiment was executed from the Web interface and returned successfully to the browser.

### Regression Tests

Final test result:

`35 passed`

---

## Design Constraints

The integration follows the following constraints:

1. Existing standalone baseline implementations are not modified.
2. Failure-Aware V3 is not integrated.
3. Aggregation strategy logic remains isolated from the Web application.
4. The Web application communicates only with the backend API.
5. Flower-specific execution is isolated behind the integration adapter.
6. Changes to the frozen Flower framework are limited to the optional result callback required for integration.

---

## Current State

The Web–Flower integration is considered complete.

Future work should avoid redesigning this integration unless a concrete requirement requires it.

Potential separate repository maintenance work includes defining a project-level dependency manifest.