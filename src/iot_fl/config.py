from __future__ import annotations


TARGET = "Machine failure"
ID_COL = "UDI"

FEATURES = [
    "air_temperature_k_z",
    "process_temperature_k_z",
    "rotational_speed_rpm_z",
    "torque_nm_z",
    "tool_wear_min_z",
    "temperature_gap_k_z",
    "power_proxy_z",
    "Type_H",
    "Type_L",
    "Type_M",
]

STRATEGIES = [
    "iid",
    "moderate_non_iid",
    "highly_non_iid",
]