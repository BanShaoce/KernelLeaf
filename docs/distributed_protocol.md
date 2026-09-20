# KernelLeaf V12.1 distributed protocol

V12.1 introduces only a communication foundation. It does not implement a
Parameter Server, distributed optimizer, worker training loop, launcher,
monitor, Ring AllReduce, gRPC, or NCCL.

## Package boundaries

- `protocol.py` defines the versioned message schema and message types.
- `serialization.py` defines the transport-independent `MessageCodec` API and
  the initial `JsonCodec`.
- `transport.py` implements length-prefixed framing over an already-connected
  socket. It does not know about JSON or arrays.

This separation allows V12.4 to introduce a binary ndarray codec without
changing socket framing or higher-level messages.

## Message schema

Protocol version 1 supports these message types:

```text
REGISTER
PULL_PARAMETERS
PUSH_GRADIENTS
PARAMETERS
HEARTBEAT
METRICS
SHUTDOWN
ERROR
```

Every message contains exactly these fields:

| Field | Type | Meaning |
|---|---|---|
| `protocol_version` | integer | Must equal `1` |
| `message_type` | enum string | One of the types above |
| `request_id` | non-empty string | Correlates requests and responses |
| `worker_id` | non-empty string | Stable sender identity |
| `step` | non-negative integer | Parameter/training version |
| `payload` | object | Message-specific data |

Unknown fields, missing fields, unknown message types, and mismatched protocol
versions fail explicitly instead of being guessed.

## JSON ndarray representation

The V12.1 JSON codec represents a numeric array as:

```json
{
  "__kernelleaf_ndarray__": true,
  "dtype": "<f4",
  "shape": [2, 2],
  "data": [1.0, 2.0, 3.0, 4.0]
}
```

The dtype uses NumPy's endian-aware dtype string. Shape and flattened data
length are validated during decoding. V12.1 supports boolean, integer,
unsigned-integer, and finite floating arrays. Object, string, complex, and
non-finite floating arrays are rejected by the JSON codec. A later binary codec
may broaden this contract deliberately.

NumPy arrays remain NumPy arrays after decoding. CuPy arrays are copied
explicitly to CPU only inside `JsonCodec.encode`, which is a network boundary,
not an operator hot path. Importing `kernelleaf.distributed` does not import
CuPy, so CPU-only installations remain supported.

## TCP framing

Each encoded message is one frame:

```text
+----------------------+-----------------------+
| uint32 payload bytes | codec payload         |
| network byte order   | exactly N bytes       |
+----------------------+-----------------------+
```

The receiver first reads exactly four bytes and then exactly the declared
payload length. This handles arbitrary TCP fragmentation and preserves message
boundaries when several frames arrive in one OS receive buffer. Empty frames
and frames above the configured maximum (64 MiB by default) are rejected before
payload allocation.

`PeerDisconnected`, `InvalidFrameLength`, and `TransportTimeout` distinguish
the principal failure cases. `FramedTransport(timeout=...)` temporarily applies
its timeout and restores the socket's previous timeout after each operation.

## Minimal example

```python
import socket
import numpy as np

from kernelleaf.distributed import (
    FramedTransport, JsonCodec, Message, MessageType,
)

sock = socket.create_connection(("127.0.0.1", 9000), timeout=5)
transport = FramedTransport(sock, JsonCodec(), timeout=5)
transport.send(Message(
    message_type=MessageType.PUSH_GRADIENTS,
    request_id="request-1",
    worker_id="worker-0",
    step=0,
    payload={"gradient": np.ones((2, 2), dtype=np.float32)},
))
```

For observability, `FramedTransport.send` returns the total number of bytes sent
(four-byte header plus codec payload). This is sufficient for V12.1 tests and
allows later training metrics to report upload/download volume without coupling
the codec to the transport.

## Verification

```powershell
python -m pytest tests/test_distributed_protocol.py -q
python -c "import sys, kernelleaf.distributed; print('cupy' in sys.modules)"
```
