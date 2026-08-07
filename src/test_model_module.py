from __future__ import annotations

import numpy as np

from iot_fl.model import local_train, predict_proba


def main() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.8, 0.9],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    y = np.array([0, 0, 1, 1], dtype=int)

    initial_parameters = np.zeros(x.shape[1] + 1, dtype=float)

    trained_parameters, loss = local_train(
        initial_parameters=initial_parameters,
        x=x,
        y=y,
        learning_rate=0.1,
        epochs=20,
        l2=0.001,
    )

    probabilities = predict_proba(x, trained_parameters)

    print("Initial parameters:", initial_parameters)
    print("Trained parameters:", trained_parameters)
    print("Training loss:", loss)
    print("Predicted probabilities:", probabilities)

    assert trained_parameters.shape == (3,)
    assert probabilities.shape == (4,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert loss > 0.0

    print("Model module test passed.")


if __name__ == "__main__":
    main()