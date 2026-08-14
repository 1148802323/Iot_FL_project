from iot_fl.flower_framework.config import FrameworkConfig

def test_valid_config():
    config = FrameworkConfig(
        aggregation="fedavg",
        num_clients=5,
        num_rounds=1,
    )

    assert config.aggregation == "fedavg"
    assert config.num_clients == 5
    assert config.num_rounds == 1


def test_invalid_num_clients():
    try:
        FrameworkConfig(
            aggregation="fedavg",
            num_clients=0,
            num_rounds=1,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for num_clients <= 0")

def test_invalid_num_rounds():
    try:
        FrameworkConfig(
            aggregation="fedavg",
            num_clients=5,
            num_rounds=0,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for num_rounds <= 0")