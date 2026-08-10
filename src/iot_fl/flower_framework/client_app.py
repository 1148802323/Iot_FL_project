from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from flwr.client import ClientApp
from flwr.common import Context

from iot_fl.flower_framework.flower_client import AI4IFlowerClient


def create_client_app(
    clients: Sequence[dict[str, Any]],
) -> ClientApp:

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        if partition_id < 0 or partition_id >= len(clients):
            raise IndexError(
                f"partition-id {partition_id} is outside the valid range "
                f"0 to {len(clients) - 1}"
            )
        client_data = clients[partition_id]
        x_train = np.asarray(client_data["x"], dtype=float)
        y_train = np.asarray(client_data["y"], dtype=int)
        initial_parameters = np.zeros(
            x_train.shape[1] + 1,
            dtype=float,
        )
        flower_client = AI4IFlowerClient(
            client_name=str(client_data["name"]),
            x_train=x_train,
            y_train=y_train,
            initial_parameters=initial_parameters,
        )

        return flower_client.to_client()

    return ClientApp(client_fn=client_fn)