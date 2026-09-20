"""Shared synchronous training runtime helpers for distributed applications."""

from __future__ import annotations

import socket
import time

from .monitor import JsonlMonitor
from .protocol import MessageType
from .transport import receive_frame, send_frame


def connect_with_retry(host, port, timeout, stop_event):
    deadline = time.monotonic() + timeout
    while not stop_event.is_set():
        try:
            connection = socket.create_connection(
                (host, port), timeout=min(timeout, 1.0)
            )
            # The short timeout above only bounds connection attempts. Raw
            # timed frame I/O must use the configured training request timeout.
            connection.settimeout(timeout)
            return connection
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"could not connect to Parameter Server at {host}:{port}"
                )
            time.sleep(0.02)
    raise RuntimeError("distributed job was stopped before connection")


def reserve_local_port(host="127.0.0.1"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _receive_decoded(transport):
    started = time.perf_counter()
    payload = receive_frame(transport.socket, transport.max_frame_size)
    receive_time = time.perf_counter() - started
    decode_started = time.perf_counter()
    response = transport.codec.decode(payload)
    decode_time = time.perf_counter() - decode_started
    return response, len(payload) + 4, receive_time, decode_time


def push_gradients_and_wait(worker, transport, local_count, poll_interval):
    """Upload one local gradient and wait for the next parameter version."""
    request_started = time.perf_counter()
    request = worker.gradients_message(local_count)
    payload = transport.codec.encode(request)
    serialize_time = time.perf_counter() - request_started
    upload_started = time.perf_counter()
    upload_bytes = send_frame(
        transport.socket, payload, transport.max_frame_size
    )
    upload_time = time.perf_counter() - upload_started
    wait_time = 0.0
    while True:
        response, wire_bytes, receive_time, decode_time = _receive_decoded(
            transport
        )
        wait_time += receive_time
        if response.message_type is MessageType.PARAMETERS:
            accept_started = time.perf_counter()
            worker.handle_response(request, response)
            return {
                "gradient_serialize_time": serialize_time,
                "gradient_upload_time": upload_time,
                "parameter_wait_time": wait_time,
                "parameter_download_time": (
                    decode_time + time.perf_counter() - accept_started
                ),
                "optimizer_time": float(
                    response.payload.get("optimizer_time", 0.0)
                ),
                "upload_bytes": upload_bytes,
                "download_bytes": wire_bytes,
            }
        worker.handle_response(request, response)
        time.sleep(poll_interval)
        request = worker.pull_message(worker.step + 1)
        pull_payload = transport.codec.encode(request)
        send_frame(
            transport.socket, pull_payload, transport.max_frame_size
        )


def monitor_process(stop_event, output_path, metric_queue, expected_records):
    del stop_event
    JsonlMonitor(output_path).run(metric_queue, expected_records)
