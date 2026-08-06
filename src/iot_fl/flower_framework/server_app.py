from __future__ import annotations

from collections.abc import Callable

import numpy as np
from flwr.common import NDArrays, Scalar

from flwr.common import (
    Context,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from iot_fl.metrics import classification_metrics
from iot_fl.model import predict_proba, sample_weights, weighted_log_loss

from iot_fl.flower_framework.strategies.strategy_factory import (
    create_strategy,
)

def create_fit_config_fn(
    *,
    learning_rate: float,
    local_epochs: int,
    l2: float,
) -> Callable[[int], dict[str, Scalar]]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    if local_epochs <= 0:
        raise ValueError("local_epochs must be greater than zero")

    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    def fit_config(server_round: int) -> dict[str, Scalar]:
        return {
            "learning_rate": learning_rate,
            "local_epochs": local_epochs,
            "l2": l2,
        }

    return fit_config

def create_evaluate_fn(
    *,
    x_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = 0.5,
) -> Callable[
    [int, NDArrays, dict[str, Scalar]],
    tuple[float, dict[str, Scalar]],
]:
    x_val_array = np.asarray(x_val, dtype=float)
    y_val_array = np.asarray(y_val, dtype=int)

    if x_val_array.ndim != 2:
        raise ValueError("x_val must be a two-dimensional array")

    if y_val_array.ndim != 1:
        raise ValueError("y_val must be a one-dimensional array")

    if len(x_val_array) != len(y_val_array):
        raise ValueError("x_val and y_val must contain the same number of rows")

    if len(y_val_array) == 0:
        raise ValueError("validation data must not be empty")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")

    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]]:
        if len(parameters) != 1:
            raise ValueError(
                "Expected exactly one parameter array for logistic regression"
            )

        global_parameters = np.asarray(
            parameters[0],
            dtype=float,
        )
        expected_size = x_val_array.shape[1] + 1

        if global_parameters.ndim != 1:
            raise ValueError("Global parameters must be one-dimensional")

        if global_parameters.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} global parameters, "
                f"received {global_parameters.size}"
            )
        probabilities = predict_proba(
            x_val_array,
            global_parameters,
        )
        validation_weights = sample_weights(y_val_array)

        validation_loss = weighted_log_loss(
            y_val_array,
            probabilities,
            validation_weights,
        )
        validation_metrics = classification_metrics(
            y_val_array,
            probabilities,
            threshold,
        )
        metrics_for_flower: dict[str, Scalar] = {
            "accuracy": float(validation_metrics["accuracy"]),
            "precision": float(validation_metrics["precision"]),
            "recall": float(validation_metrics["recall"]),
            "f1": float(validation_metrics["f1"]),
            "threshold": float(validation_metrics["threshold"]),
        }
        return float(validation_loss), metrics_for_flower

    return evaluate

def create_initial_parameters(
    num_features: int,
) -> Parameters:

    if num_features <= 0:
        raise ValueError("num_features must be greater than zero")

    initial_weights = np.zeros(
        num_features + 1,
        dtype=float,
    )

    return ndarrays_to_parameters([initial_weights])

def create_server_app(
    *,
    num_features: int,
    x_val: np.ndarray,
    y_val: np.ndarray,
    rounds: int,
    learning_rate: float,
    local_epochs: int,
    l2: float,
    num_clients: int,
    aggregation: str,
) -> ServerApp:
    if rounds <= 0:
        raise ValueError("rounds must be greater than zero")

    if num_clients <= 0:
        raise ValueError("num_clients must be greater than zero")

    initial_parameters = create_initial_parameters(num_features)

    fit_config_fn = create_fit_config_fn(
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        l2=l2,
    )

    evaluate_fn = create_evaluate_fn(
        x_val=x_val,
        y_val=y_val,
        threshold=0.5,
    )
    def server_fn(context: Context) -> ServerAppComponents:
        strategy = create_strategy(
            aggregation=aggregation,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=fit_config_fn,
            evaluate_fn=evaluate_fn,
        )
        server_config = ServerConfig(
            num_rounds=rounds,
        )
        return ServerAppComponents(
            strategy=strategy,
            config=server_config,
        )
    return ServerApp(server_fn=server_fn)