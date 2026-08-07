from dataclasses import dataclass


@dataclass
class FrameworkConfig:
    aggregation: str
    num_clients: int
    num_rounds: int

    def __post_init__(self) -> None:
        if self.num_clients <= 0:
            raise ValueError("num_clients must be greater than zero")

        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be greater than zero")