"""Small smoke test for AI4IFlowerClient; does not start a Flower server."""

from __future__ import annotations

import numpy as np

from iot_fl.flower_framework.flower_client import AI4IFlowerClient


def main() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 10))
    y = np.array([0] * 30 + [1] * 10, dtype=int)
    initial = np.zeros(x.shape[1] + 1, dtype=float)

    client = AI4IFlowerClient(
        client_name="factory_test",
        x_train=x,
        y_train=y,
        initial_parameters=initial,
    )

    initial_parameters = client.get_parameters({})
    updated_parameters, rows, fit_metrics = client.fit(
        initial_parameters,
        {"learning_rate": 0.05, "local_epochs": 2, "l2": 0.001},
    )
    loss, eval_rows, eval_metrics = client.evaluate(
        updated_parameters,
        {"threshold": 0.5},
    )

    assert rows == len(y)
    assert eval_rows == len(y)
    assert len(updated_parameters) == 1
    assert updated_parameters[0].shape == initial.shape
    assert np.isfinite(loss)

    print("Flower client smoke test passed")
    print("fit metrics:", fit_metrics)
    print("evaluation metrics:", eval_metrics)


if __name__ == "__main__":
    main()
