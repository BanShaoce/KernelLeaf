# Sparse distributed experiments

These sparse logistic-regression and collapsed-Gibbs LDA experiments were
adapted from [KernelLeaf PR #2](https://github.com/GuoGuo614/KernelLeaf/pull/2)
by BanShaoce (commits `0788189` and `3df3836`).

The PR was based on the V11 repository and introduced a sparse key/value
Parameter Server whose module names conflict with KernelLeaf's later V12–V16
dense Tensor training stack. To preserve both designs, the experiment runtime
is isolated under `experiments.distributed.runtime`; it is not exported from
`kernelleaf.distributed` and does not replace the canonical protocol,
transport, launcher, Parameter Server, or Worker.

Run the synthetic experiments from the repository root:

```powershell
python -m experiments.distributed.run_experiments `
  --experiment lr --transport socket --mode sync --workers 4 --shards 2

python -m experiments.distributed.run_experiments `
  --experiment lda --transport socket --mode async --workers 4 --shards 2
```

Generate the course comparison table:

```powershell
python -m experiments.distributed.benchmark_table `
  --repeat 3 --warmup-epochs 1 --epochs 12 `
  --skip-grpc --output runs/sparse_ps_table.md
```

gRPC remains optional. Install the `grpc` extra only when that comparison is
needed; Socket+JSON and all baseline tests require only NumPy.
