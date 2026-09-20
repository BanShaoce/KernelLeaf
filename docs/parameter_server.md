# KernelLeaf V12.2 synchronous Parameter Server

V12.2 builds a synchronous Parameter Server state machine on the V12.1 message,
codec, and framed TCP layers. It intentionally does not provide a process
launcher, MNIST application, monitor, benchmark runner, Ring AllReduce, gRPC,
or NCCL. Those belong to later versions.

## Ownership model

The `ParameterServer` is the only component that owns the authoritative model
and optimizer. A `Worker` owns a local model replica only for forward/backward:

```text
Worker local forward/backward
        │
        ├─ PUSH_GRADIENTS(step=t, local_sample_count=n)
        ▼
ParameterServer waits for every configured Worker
        │
        ├─ sample-count weighted gradient average
        ├─ exactly one optimizer.step()
        ├─ parameter version t → t+1
        ▼
Workers pull PARAMETERS(step=t+1)
```

`Worker` deliberately has no optimizer argument and never calls
`optimizer.step()`. The server checks at construction that its optimizer owns
every model parameter exactly once.

## Stable parameter schema

Trainable parameters are sorted by their full `Module.named_parameters()` name.
Each peer uses a schema containing:

```json
{"name": "weight", "shape": [4, 3], "dtype": "<f4"}
```

Gradient and parameter payloads are ordered lists of `{name, value}` entries.
The receiver rejects missing, extra, duplicated, or reordered names rather than
depending on dictionary traversal order. V12.2 synchronizes trainable
parameters only; model buffers such as BatchNorm running statistics require an
explicit policy in a later training integration.

## Step protocol

1. A Worker sends `REGISTER` with its parameter schema.
2. The server validates the configured Worker ID and schema, then returns the
   current `PARAMETERS`.
3. Every Worker computes a local mean gradient for the same `step`.
4. Every Worker sends `PUSH_GRADIENTS` with `local_sample_count`.
5. Before all submissions arrive, the server returns `HEARTBEAT` with
   `status="waiting"`.
6. The final submission triggers a sample-count weighted average and one
   optimizer update. Its response contains the new parameters.
7. Waiting Workers issue `PULL_PARAMETERS` for `step+1`; they receive either a
   waiting heartbeat or the new parameters.

The V12.3 launcher/training loop owns polling cadence and process cleanup; see
`distributed_training.md`.

## Error responses

Expected request failures return a protocol `ERROR` instead of dropping the
connection. Codes include:

- `UNKNOWN_WORKER` / `NOT_REGISTERED`
- `SCHEMA_MISMATCH` / `PARAMETER_ORDER_MISMATCH`
- `DUPLICATE_GRADIENT`
- `STALE_STEP` / `FUTURE_STEP`
- `SHAPE_MISMATCH` / `DTYPE_MISMATCH`
- `NONFINITE_GRADIENT` / `INVALID_PAYLOAD`
- `WORKER_TIMEOUT`

When any registered Worker exceeds `heartbeat_timeout`, the synchronous job is
marked failed and pending gradients are discarded. Continuing with fewer
Workers would change the effective global batch and is therefore not done
silently.

## Device boundary

Wire values are CPU NumPy arrays. Worker gradients use an explicit GPU-to-CPU
copy while building `PUSH_GRADIENTS`. If eager scalar arithmetic promoted a
local gradient, the Worker explicitly normalizes it to the declared parameter
dtype at this boundary. The server still rejects any received dtype mismatch.
The server aggregates on CPU, then uses an
explicit CPU-to-parameter-device copy before its optimizer step. Parameter
downloads follow the reverse explicit boundary. No operator or autograd hot
path silently moves arrays between devices.

## Verification

```powershell
python -m pytest tests/test_distributed_protocol.py -q
python -m pytest tests/test_parameter_server.py -q
python -m pytest -q
```

The end-to-end test uses one server, two Workers, two real loopback TCP
connections, and different local sample counts. Its update is compared with a
single-process reference computed over the combined batch. A CUDA test, when
available, verifies a GPU-resident server model with CPU wire arrays.
