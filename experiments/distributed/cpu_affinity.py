"""Compatibility imports for benchmark CPU affinity helpers."""

from .runtime.cpu_affinity import (
    available_logical_cpus,
    cpu_model_name,
    format_core_groups,
    physical_core_groups,
    pinned_to_cpus,
    set_current_process_affinity,
)

__all__ = [
    "available_logical_cpus",
    "cpu_model_name",
    "format_core_groups",
    "physical_core_groups",
    "pinned_to_cpus",
    "set_current_process_affinity",
]
