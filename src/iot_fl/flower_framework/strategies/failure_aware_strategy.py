from __future__ import annotations

from flwr.server.strategy import FedAvg

class FailureAwareStrategy(FedAvg):
    """Pluggable strategy shell for future failure-aware aggregation."""

    pass