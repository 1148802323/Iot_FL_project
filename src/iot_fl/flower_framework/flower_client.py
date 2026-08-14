from __future__ import annotations

from typing import Final

import numpy as np
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from iot_fl.metrics import classification_metrics
from iot_fl.model import local_train, predict_proba, sample_weights, weighted_log_loss




DEFAULT_LEARNING_RATE: Final[float] = 0.05
DEFAULT_LOCAL_EPOCHS: Final[int] = 5
DEFAULT_L2: Final[float] = 0.001
DEFAULT_THRESHOLD: Final[float] = 0.5


class AI4IFlowerClient(NumPyClient):
    """Flower client wrapping one AI4I factory's local dataset.

    The class does not read files and does not modify the existing baseline
    modules. Data are supplied when the client is created, while training and
    prediction reuse the functions already extracted into ``iot_fl.model``.

    Flower represents model parameters as a list of NumPy arrays. The current
    logistic-regression model has one parameter vector, so the list always
    contains exactly one array: ``[weights]``.
    """

    def __init__(
        self,
        *,
        client_name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        initial_parameters: np.ndarray,
        x_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
    ) -> None:
        self.client_name = client_name
        self.x_train = np.asarray(x_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=int)
        self.x_eval = None if x_eval is None else np.asarray(x_eval, dtype=float)
        self.y_eval = None if y_eval is None else np.asarray(y_eval, dtype=int)
        self.parameters = np.asarray(initial_parameters, dtype=float).copy()

        self._validate_data()

    def _validate_data(self) -> None:
        if self.x_train.ndim != 2:
            raise ValueError("x_train must be a two-dimensional array")
        if self.y_train.ndim != 1:
            raise ValueError("y_train must be a one-dimensional array")
        if len(self.x_train) != len(self.y_train):
            raise ValueError("x_train and y_train must contain the same number of rows")
        if len(self.x_train) == 0:
            raise ValueError("A Flower client cannot be created with an empty training set")
        if self.parameters.ndim != 1:
            raise ValueError("initial_parameters must be a one-dimensional array")
        if len(self.parameters) != self.x_train.shape[1] + 1:
            raise ValueError(
                "initial_parameters must contain one intercept and one coefficient "
                "for every input feature"
            )

        if (self.x_eval is None) != (self.y_eval is None):
            raise ValueError("x_eval and y_eval must either both be provided or both be omitted")
        if self.x_eval is not None and self.y_eval is not None:
            if self.x_eval.ndim != 2 or self.y_eval.ndim != 1:
                raise ValueError("Evaluation arrays have invalid dimensions")
            if len(self.x_eval) != len(self.y_eval):
                raise ValueError("x_eval and y_eval must contain the same number of rows")
            if self.x_eval.shape[1] != self.x_train.shape[1]:
                raise ValueError("Training and evaluation data must use the same features")

    @staticmethod
    def _unpack_parameters(parameters: NDArrays) -> np.ndarray:
        if len(parameters) != 1:
            raise ValueError(
                f"Expected one logistic-regression parameter array, received {len(parameters)}"
            )
        unpacked = np.asarray(parameters[0], dtype=float)
        if unpacked.ndim != 1:
            raise ValueError("The logistic-regression parameter array must be one-dimensional")
        return unpacked.copy()

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Return metadata that can later support client selection/monitoring."""
        del config
        return {
            "client_name": self.client_name,
            "train_rows": len(self.y_train),
            "failure_rate": float(np.mean(self.y_train)),
        }

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters to the server."""
        del config
        return [self.parameters.copy()]

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train the global parameters on this client's local data."""
        incoming_parameters = self._unpack_parameters(parameters)

        learning_rate = float(config.get("learning_rate", DEFAULT_LEARNING_RATE))
        local_epochs = int(config.get("local_epochs", DEFAULT_LOCAL_EPOCHS))
        l2 = float(config.get("l2", DEFAULT_L2))

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if local_epochs <= 0:
            raise ValueError("local_epochs must be greater than zero")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")

        updated_parameters, local_loss = local_train(
            initial_parameters=incoming_parameters,
            x=self.x_train,
            y=self.y_train,
            learning_rate=learning_rate,
            epochs=local_epochs,
            l2=l2,
        )
        self.parameters = updated_parameters.copy()

        return (
            [self.parameters.copy()],
            len(self.y_train),
            {
                "client_name": self.client_name,
                "local_loss": float(local_loss),
                "failure_rate": float(np.mean(self.y_train)),
            },
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate parameters using local evaluation data, or training data as fallback."""
        self.parameters = self._unpack_parameters(parameters)

        x_eval = self.x_train if self.x_eval is None else self.x_eval
        y_eval = self.y_train if self.y_eval is None else self.y_eval
        threshold = float(config.get("threshold", DEFAULT_THRESHOLD))

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")

        probabilities = predict_proba(x_eval, self.parameters)
        loss = weighted_log_loss(y_eval, probabilities, sample_weights(y_eval))
        metrics = classification_metrics(y_eval, probabilities, threshold)

        return (
            float(loss),
            len(y_eval),
            {
                "client_name": self.client_name,
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            },
        )
