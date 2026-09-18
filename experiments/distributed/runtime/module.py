"""Bridge KernelLeaf modules and dense module gradients to parameter servers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .client import ParameterServerClient
from kernelleaf.backend import asnumpy, to_device


PARAMETER_PREFIX = "param:"


def _parameter_key(name: str, prefix: str = PARAMETER_PREFIX) -> str:
    return f"{prefix}{name}"


def collect_module_parameters(
    module, *, prefix: str = PARAMETER_PREFIX
) -> Dict[str, Tuple[float, ...]]:
    """Flatten named module parameters into sparse-server-compatible values."""
    values = {}
    for name, parameter in module.named_parameters():
        array = asnumpy(parameter.realize_cached_data()).reshape(-1)
        values[_parameter_key(name, prefix)] = tuple(float(item) for item in array)
    return values


def push_module_parameters(
    module,
    client: ParameterServerClient,
    *,
    clock: int = 0,
    prefix: str = PARAMETER_PREFIX,
):
    """Seed a parameter server with the current module parameter values."""
    return client.push(
        collect_module_parameters(module, prefix=prefix),
        clock=clock,
        op="set",
    )


def pull_module_parameters(
    module,
    client: ParameterServerClient,
    *,
    prefix: str = PARAMETER_PREFIX,
    strict: bool = True,
) -> None:
    """Copy flattened parameter-server values back into a KernelLeaf module."""
    parameters = dict(module.named_parameters())
    keys = [_parameter_key(name, prefix) for name in parameters]
    pulled = client.pull(keys)
    missing = [key for key in keys if key not in pulled]
    if strict and missing:
        raise KeyError(f"parameter server did not return keys: {missing}")
    for name, parameter in parameters.items():
        key = _parameter_key(name, prefix)
        if key not in pulled:
            continue
        values = np.asarray(pulled[key], dtype=parameter.dtype)
        if values.size != int(np.prod(parameter.shape)):
            raise ValueError(
                f"parameter {name!r} expected {np.prod(parameter.shape)} values, "
                f"got {values.size}"
            )
        parameter.cached_data = to_device(
            values.reshape(parameter.shape),
            parameter.device,
            dtype=parameter.dtype,
        )


def push_module_gradients(
    module,
    client: ParameterServerClient,
    *,
    clock: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    prefix: str = PARAMETER_PREFIX,
):
    """Push negative learning-rate-scaled dense grads as sparse keys."""
    updates: Dict[str, Tuple[float, ...]] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        gradient = asnumpy(parameter.grad.realize_cached_data())
        if weight_decay:
            data = asnumpy(parameter.realize_cached_data())
            gradient = gradient + float(weight_decay) * data
        delta = (-float(learning_rate) * gradient).reshape(-1)
        updates[_parameter_key(name, prefix)] = tuple(float(item) for item in delta)
    return client.push(updates, clock=clock, op="add")
