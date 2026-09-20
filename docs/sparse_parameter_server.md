# Sparse Parameter Server experiments

> Adapted from KernelLeaf PR #2 by BanShaoce. This is an experiment-local sparse KV runtime; the canonical dense Tensor training stack remains `kernelleaf.distributed`.

KernelLeaf's distributed layer follows the parameter-server design in
"Scaling Distributed Machine Learning with the Parameter Server" (Li et al.,
OSDI 2014): a sharded server group stores sparse parameters, worker nodes pull
the ranges they need and push sparse updates, and consistency is selected by
the workload rather than by the model implementation.

## Scope

The implementation has four layers:

1. `experiments.distributed.runtime.protocol` defines transport-neutral sparse entries,
   requests, responses, and add/set aggregation.
2. `experiments.distributed.runtime.server` owns one shard, parameter versions, pending
   worker clocks, sync barriers, and immediate async updates.
3. `experiments.distributed.runtime.socket_transport` and
   `experiments.distributed.runtime.grpc_transport` implement the same request/response
   contract over line-delimited Socket+JSON and gRPC+protobuf respectively.
4. `experiments.distributed.runtime.client` routes keys to stable shards and sends a
   heartbeat to every shard during sync rounds, including empty sparse updates.

`experiments.distributed.runtime.module` bridges ordinary KernelLeaf `Module` parameters
to the same protocol. Named parameters are flattened into sparse keys and can
be pulled, seeded, or updated with negative learning-rate-scaled gradients.

## Parameter model

```text
worker 0 ── pull(keys) / push(sparse entries) ──┐
worker 1 ── pull(keys) / push(sparse entries) ──┼── shard 0
worker 2 ── pull(keys) / push(sparse entries) ──┤
                                                  └── shard 1
```

Keys are arbitrary strings. A shard stores each key as a tuple of floating
point values, so scalar model weights and count buckets use the same protocol.
`crc32(key) % num_shards` provides stable range routing. A worker that has no
nonzero update for a shard still sends an empty push, which is required for a
correct synchronous barrier.

Each successful update increments the key version and the shard's global clock.
The server never broadcasts a full model: pull and push payloads contain only
the requested or changed ranges.

## Consistency modes

### Synchronous

Every worker has a non-negative `clock` (one epoch for sparse LR, one Gibbs
sweep for LDA). A push arrives at all shards and waits until every registered
worker has submitted the same clock. The shard then merges all updates and
advances the round. `barrier_timeout` prevents a failed worker from blocking a
shard forever.

### Asynchronous

Each push is applied immediately under the shard lock. Workers do not wait for
peers, so fast nodes may read a mixture of newer and older parameters. The
experiments include an optional `--worker-delay` to make this interleaving
visible without changing the algorithm.

### Runtime switching

`ParameterServerClient.set_mode("sync" | "async")` changes all shards.

- `sync -> async` flushes pending barrier updates before releasing waiters, so
  a mode switch does not discard committed work.
- `async -> sync` tracks the declared mode of each worker. A worker that has
  not reached the switch point can finish in-flight async pushes immediately;
  once it declares sync, its next round participates in the new barrier. This
  avoids treating an old async clock as a sync barrier.

## Socket+JSON

The default transport uses standard-library TCP sockets and UTF-8 JSON. Messages
are newline-delimited and capped at 16 MiB. Mutating push requests are not
automatically retried because duplicate additive updates would corrupt the
parameter version. Read-only requests may reconnect and retry once.

## gRPC+protobuf

The canonical schema is `kernelleaf/distributed/kernelleaf.proto`. Python builds
equivalent descriptors at runtime, so `protoc` is not needed for the built-in
experiments. Install the optional runtime with:

```bash
pip install -e ".[grpc,benchmark]"
```

The gRPC service exposes a unary `ParameterServer.Request` RPC and uses protobuf
for the request, sparse entries, response values, version, and metadata fields.

## Single-machine multi-node experiments

The experiments emulate a cluster with one OS process per parameter shard and
one OS process per worker. Communication always uses loopback TCP, so replacing
`127.0.0.1` with real hosts does not change the algorithm code.

Sparse logistic regression:

```bash
python -m experiments.distributed.run_experiments \
  --experiment lr --transport socket --mode sync \
  --workers 4 --shards 2 --epochs 20
```

Latent Dirichlet allocation:

```bash
python -m experiments.distributed.run_experiments \
  --experiment lda --transport socket --mode async \
  --workers 4 --shards 2 --sweeps 8
```

Switch consistency during a run:

```bash
python -m experiments.distributed.run_experiments \
  --experiment lr --transport grpc --mode sync \
  --switch-at 10 --switch-to async --workers 4 --shards 2 --epochs 20
```

Run both experiments with one command:

```bash
python -m experiments.distributed.run_experiments \
  --experiment all --transport socket --mode async
```

### Result-table measurement

The LR benchmark below runs the four configurations from the course table with
identical synthetic data, epochs, batch size, learning rate, and validation
set:

```bash
pip install -e ".[grpc,benchmark]"
python -X utf8 -m experiments.distributed.benchmark_table \
  --repeat 5 --warmup-epochs 2 --epochs 12 \
  --output output/distributed_ps_table.md
```

Rows produced:

```text
single machine baseline
Socket+JSON, 2 workers
gRPC+protobuf, 2 workers
Socket+JSON, 4 workers
```

The report columns use these definitions:

- `one_epoch_total_time`: the longest wall-clock epoch across all workers. This
  answers how long one global epoch takes.
- `compute_time`: sum of worker-local non-communication time in the same epoch.
- `communication_time`: sum of time blocked in pull, push, mode-switch, and
  barrier calls. A sync push therefore includes waiting for slower workers.
- `message_bytes`: sum of serialized request and response payload bytes. Socket
  counts UTF-8 JSON plus newline framing; gRPC counts protobuf payloads and does
  not count HTTP/2 headers or transport framing.
- `accuracy`: final validation accuracy from the same held-out synthetic task.
- `speedup`: `single-machine epoch time / configuration epoch time`.
- `ideal_speedup`: the worker count `N`; measured values should normally be
  below `N` because of communication, barrier waiting, server CPU use, and
  multi-core frequency scaling.

The first `--warmup-epochs` values are discarded, then the remaining epochs are
averaged for each configuration. `--repeat` repeats the complete configuration
and averages across repetitions. Run at least three repetitions for a course
report and keep the machine idle during measurement. Do not compare runs with
different data size, batch size, learning rate, epoch count, seed, or CPU load.

The benchmark defaults are deliberately compute-dense:

```text
samples=8192, features=512, nnz=128, batch-size=8192
```

With `batch_size >= local worker samples`, each worker performs one pull and one
push per epoch. This keeps the experiment focused on compute scaling; a tiny
mini-batch would make JSON serialization and synchronization dominate before
the compute speedup can appear. The baseline and every worker run the same
amount of local computations per assigned sample.

### CPU resource control and experiment process

The benchmark discovers physical cores rather than treating every logical
processor as an independent core. On a CPU with SMT, a physical-core group may
contain two logical processors. The allocation is:

```text
single baseline: worker core group 0
2 workers:       worker core groups 0, 1
4 workers:       worker core groups 0, 1, 2, 3
servers:         additional core groups after all workers
```

Consequently the baseline uses one physical core, two workers use two physical
cores, and each worker gets the same affinity and sample budget as the
baseline. Parameter-server processes are pinned away from worker cores so server
threads do not consume the resource being scaled.

The visible experiment process is:

1. Print the detected CPU model and physical-core groups.
2. Generate the same synthetic task and validation set for all configurations.
3. Run the baseline pinned to core group 0 for each repetition.
4. Run Socket+JSON with worker groups `0..N-1` and server groups after them.
5. Run gRPC+protobuf with the identical worker/server allocation.
6. Discard warmup epochs, average the remaining epoch measurements, and compute
   the measured speedup against the single-machine row.
7. Report the table to stdout and optionally write UTF-8 Markdown/CSV/JSON.

`--no-pin-cpu` disables affinity control for debugging, but its speedup values
are not suitable for the report. If the machine has fewer than six physical
cores (four workers plus two server shards), the benchmark stops rather than
silently overlapping resources.

Supported output formats are Markdown (default), CSV, and JSON:

```bash
python -X utf8 -m experiments.distributed.benchmark_table --format csv
python -X utf8 -m experiments.distributed.benchmark_table --format json
```

### Sparse logistic regression design

The synthetic task creates high-dimensional sparse rows. A small set of stable
signal features determines the label; noise features remain sparse. Each worker
owns a disjoint row partition, pulls only the union of features in its current
batch, and pushes feature-index and bias updates. Sync mode aggregates one
global epoch update per worker; async mode pushes after every mini-batch.
Validation uses a disjoint row sample but the same generating weights.

This is intentionally the lightweight logistic-SGD variant rather than a
reproduction of the paper's block-proximal l1 algorithm, KKT filter, or its
billion-feature production workload.

Reported metrics are validation accuracy and binary log loss. Server metadata
reports the number of touched sparse parameters and the applied worker clocks.

### LDA design

The corpus contains documents drawn from topic-specific word distributions.
Workers run collapsed Gibbs sampling on disjoint document partitions. Each
worker keeps document-topic counts locally and stores topic-word counts and
topic totals on the parameter server. A sparse delta contains only the word and
topic entries changed by a Gibbs move. Sync mode pushes once per sweep after all
documents in the local partition are sampled; async mode pushes after each
document so other workers can observe its count updates immediately.

Reported metrics are training perplexity, top words per topic, and overlap with
the known synthetic top words. The last two metrics are diagnostic smoke
signals, not a substitute for held-out topic-model evaluation.

The paper combines stochastic variational inference, collapsed Gibbs sampling,
and distributed gradients. This experiment implements only the collapsed-Gibbs
piece so the sparse count synchronization remains easy to inspect.

## Tests

```bash
python -m pytest tests/test_distributed.py -q
```

The suite covers JSON/protobuf-independent message round trips, sync barriers,
both switch directions including in-flight async updates, Socket end-to-end
traffic, the KernelLeaf module bridge, and a real gRPC round trip when `grpcio`
is installed.

## Limitations

- There is no replication, failover, or persistent parameter storage yet.
- Range compression is sparse-value elimination, not quantized communication.
- Bounded-delay/SSP and application-specific server-side user functions are not
  implemented; sync and async are the two supported consistency choices.
- The sync implementation assumes every registered worker eventually reaches
  the round and uses a timeout rather than elastic membership.
- The single-machine processes emulate nodes and network boundaries; they do
  not model real bandwidth, packet loss, or stragglers beyond delay injection.
