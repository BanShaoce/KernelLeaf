"""Spawn-only local process supervision for KernelLeaf distributed jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
import queue
import time
import traceback


@dataclass(frozen=True)
class RoleSpec:
    role: str
    target: object
    args: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("role must be a non-empty string")
        if not callable(self.target):
            raise TypeError("target must be callable")


@dataclass(frozen=True)
class LaunchResult:
    exitcodes: dict
    elapsed: float


class DistributedProcessError(RuntimeError):
    """Raised after one role fails and the remaining roles are terminated."""

    def __init__(self, role, detail, exitcodes):
        super().__init__(f"distributed role {role!r} failed: {detail}")
        self.role = role
        self.exitcodes = exitcodes


def _role_entry(role, target, status_queue, stop_event, args):
    status_queue.put(("started", role, ""))
    try:
        target(stop_event, *args)
    except BaseException:
        status_queue.put(("failed", role, traceback.format_exc()))
        raise
    else:
        status_queue.put(("completed", role, ""))


class Launcher:
    """Launch roles with Windows-compatible spawn and supervise their lifetime."""

    def __init__(self, *, poll_interval=0.05, shutdown_timeout=3.0):
        if poll_interval <= 0 or shutdown_timeout <= 0:
            raise ValueError("launcher timeouts must be positive")
        self.context = mp.get_context("spawn")
        self.poll_interval = float(poll_interval)
        self.shutdown_timeout = float(shutdown_timeout)

    def queue(self):
        return self.context.Queue()

    def event(self):
        return self.context.Event()

    def run(self, specs, *, timeout=None):
        specs = tuple(specs)
        roles = [spec.role for spec in specs]
        if not specs or len(set(roles)) != len(roles):
            raise ValueError("role specs must be non-empty with unique role names")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
        status_queue = self.context.Queue()
        stop_event = self.context.Event()
        processes = {
            spec.role: self.context.Process(
                name=f"kernelleaf-{spec.role}",
                target=_role_entry,
                args=(spec.role, spec.target, status_queue, stop_event, spec.args),
            )
            for spec in specs
        }
        started = time.perf_counter()
        failure = None
        try:
            for process in processes.values():
                process.start()
            completed = set()
            while len(completed) < len(processes):
                if timeout is not None and time.perf_counter() - started > timeout:
                    failure = ("launcher", f"timed out after {timeout:.3f}s")
                    break
                try:
                    state, role, detail = status_queue.get(
                        timeout=self.poll_interval
                    )
                    if state == "failed":
                        failure = (role, detail)
                        break
                    if state == "completed":
                        completed.add(role)
                except queue.Empty:
                    pass
                for role, process in processes.items():
                    if role not in completed and process.exitcode not in (None, 0):
                        failure = (role, f"process exit code {process.exitcode}")
                        break
                if failure:
                    break
        finally:
            if failure:
                stop_event.set()
                for process in processes.values():
                    if process.is_alive():
                        process.terminate()
            for process in processes.values():
                process.join(timeout=self.shutdown_timeout)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=self.shutdown_timeout)
            exitcodes = {
                role: process.exitcode for role, process in processes.items()
            }
            status_queue.close()
        if failure:
            raise DistributedProcessError(failure[0], failure[1], exitcodes)
        nonzero = {role: code for role, code in exitcodes.items() if code != 0}
        if nonzero:
            role, code = next(iter(nonzero.items()))
            raise DistributedProcessError(role, f"process exit code {code}", exitcodes)
        return LaunchResult(exitcodes, time.perf_counter() - started)
