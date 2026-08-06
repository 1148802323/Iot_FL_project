from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Convert model logits into probabilities."""
    clipped_z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-clipped_z))


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Add a column of ones for the logistic-regression intercept."""
    return np.column_stack([np.ones(len(x)), x])


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Calculate balanced sample weights for binary classification."""
    positive_count = float(y.sum())
    negative_count = float((y == 0).sum())

    if positive_count == 0 or negative_count == 0:
        return np.ones(len(y), dtype=float)

    positive_weight = len(y) / (2 * positive_count)
    negative_weight = len(y) / (2 * negative_count)

    return np.where(
        y == 1,
        positive_weight,
        negative_weight,
    )


def weighted_log_loss(
    y: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Calculate weighted binary cross-entropy loss."""
    epsilon = 1e-8

    losses = -(
        y * np.log(probabilities + epsilon)
        + (1 - y) * np.log(1 - probabilities + epsilon)
    )

    return float(np.average(losses, weights=weights))


def local_train(
    initial_parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> tuple[np.ndarray, float]:
    """Train one logistic-regression model on one client's local data."""
    x_with_intercept = add_intercept(x)
    parameters = initial_parameters.copy()
    weights = sample_weights(y)

    for _ in range(epochs):
        probabilities = sigmoid(x_with_intercept @ parameters)

        gradient = (
            x_with_intercept.T
            @ ((probabilities - y) * weights)
        ) / len(y)

        # Do not regularise the intercept.
        gradient[1:] += l2 * parameters[1:]

        parameters -= learning_rate * gradient

    final_probabilities = sigmoid(x_with_intercept @ parameters)
    final_loss = weighted_log_loss(
        y,
        final_probabilities,
        weights,
    )

    return parameters, final_loss


def predict_proba(
    x: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """Predict the probability of machine failure."""
    x_with_intercept = add_intercept(x)
    return sigmoid(x_with_intercept @ parameters)