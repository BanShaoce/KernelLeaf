"""Physical-core discovery and process affinity helpers for benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import platform
import sys
from typing import Iterable, Optional, Sequence, Tuple


try:
    import psutil
except ImportError:  # pragma: no cover - exercised by the actionable error.
    psutil = None


def _require_psutil():
    if psutil is None:
        raise RuntimeError(
            "CPU affinity benchmarking requires psutil. Install with "
            "`pip install -e '.[benchmark]'`."
        )
    return psutil


def available_logical_cpus() -> Tuple[int, ...]:
    """Return logical CPUs currently available to this process."""
    process = _require_psutil().Process()
    try:
        return tuple(sorted(int(cpu) for cpu in process.cpu_affinity()))
    except (AttributeError, NotImplementedError):
        return tuple(range(os.cpu_count() or 1))


def _windows_core_groups(available: Iterable[int]) -> Tuple[Tuple[int, ...], ...]:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetLogicalProcessorInformationEx
    function.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    function.restype = wintypes.BOOL

    required = wintypes.DWORD(0)
    function(0, None, ctypes.byref(required))
    if not required.value:
        raise OSError(ctypes.get_last_error(), "cannot query logical processor info")
    buffer = ctypes.create_string_buffer(required.value)
    if not function(0, buffer, ctypes.byref(required)):
        raise ctypes.WinError(ctypes.get_last_error())

    raw = buffer.raw
    available_set = set(int(cpu) for cpu in available)
    groups = []
    offset = 0
    while offset < required.value:
        relationship = int.from_bytes(raw[offset : offset + 4], "little")
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        if size < 8 or offset + size > required.value:
            raise OSError("invalid GetLogicalProcessorInformationEx buffer")
        if relationship == 0:  # RelationProcessorCore
            group_count = int.from_bytes(raw[offset + 30 : offset + 32], "little")
            for group_index in range(group_count):
                base = offset + 32 + group_index * 16
                mask = int.from_bytes(raw[base : base + 8], "little")
                group_id = int.from_bytes(raw[base + 8 : base + 10], "little")
                cpus = tuple(
                    sorted(
                        group_id * 64 + bit
                        for bit in range(64)
                        if mask & (1 << bit)
                    )
                )
                allowed = tuple(cpu for cpu in cpus if cpu in available_set)
                if allowed:
                    groups.append(allowed)
        offset += size
    return tuple(groups)


def _parse_cpu_list(value: str) -> Tuple[int, ...]:
    cpus = []
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return tuple(sorted(set(cpus)))


def _linux_core_groups(available: Iterable[int]) -> Tuple[Tuple[int, ...], ...]:
    available_set = set(int(cpu) for cpu in available)
    groups = []
    seen = set()
    for cpu in sorted(available_set):
        topology = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        if not topology.exists():
            continue
        siblings = tuple(
            item
            for item in _parse_cpu_list(topology.read_text(encoding="ascii"))
            if item in available_set
        )
        key = tuple(sorted(siblings))
        if key and key not in seen:
            seen.add(key)
            groups.append(key)
    return tuple(groups)


def physical_core_groups() -> Tuple[Tuple[int, ...], ...]:
    """Return one CPU group per physical core, sorted deterministically."""
    available = available_logical_cpus()
    if not available:
        raise RuntimeError("no logical CPUs are available to the benchmark")
    if sys.platform == "win32":
        groups = _windows_core_groups(available)
    elif sys.platform.startswith("linux"):
        groups = _linux_core_groups(available)
    else:
        groups = ()
    if not groups:
        # Conservative fallback: one logical CPU per assignable group.
        groups = tuple((cpu,) for cpu in available)
    return tuple(sorted(groups, key=lambda group: (min(group), len(group))))


def format_core_groups(groups: Sequence[Sequence[int]]) -> str:
    return ",".join(
        "[" + "/".join(str(int(cpu)) for cpu in group) + "]" for group in groups
    )


def set_current_process_affinity(cpus: Optional[Sequence[int]]) -> None:
    """Pin the current process when an affinity is supplied."""
    if not cpus:
        return
    process = _require_psutil().Process()
    process.cpu_affinity([int(cpu) for cpu in cpus])


@contextmanager
def pinned_to_cpus(cpus: Optional[Sequence[int]]):
    """Temporarily pin the current process and restore its previous affinity."""
    if not cpus:
        yield
        return
    process = _require_psutil().Process()
    previous = process.cpu_affinity()
    process.cpu_affinity([int(cpu) for cpu in cpus])
    try:
        yield
    finally:
        process.cpu_affinity(previous)


def cpu_model_name() -> str:
    """Best-effort CPU model label for benchmark provenance."""
    return platform.processor() or "unknown CPU"
