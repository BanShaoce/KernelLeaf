# KernelLeaf V12.3 单机同步 Parameter Server 训练

V12.3 在 V12.1 的 Socket+JSON 协议和 V12.2 的同步聚合状态机之上增加进程编排与 MNIST 训练，不改变 `TensorOp.compute / TensorOp.gradient` 自动微分结构，也不引入 PyTorch、gRPC、Ring AllReduce 或 NCCL。

## 四种角色

- **Parameter Server**：持有唯一的全局模型和优化器；收到全部 Worker 的梯度后按实际样本数加权，并且每个全局 step 只调用一次 `optimizer.step()`。
- **Worker**：持有模型副本，只执行数据读取、前向、反向以及梯度上传；不创建优化器。
- **Launcher**：强制使用 `multiprocessing.get_context("spawn")` 启动并监督全部角色。任一进程异常或作业超时会设置停止事件并终止其余进程，避免同步作业永久等待。
- **Monitor**：校验每条指标并以追加模式写入 JSONL。每写一行都会 flush；中途失败时已完成记录仍可读取。

所有可执行文件都有 `if __name__ == "__main__"` 保护，Windows 不依赖 `fork`。

## Strong scaling 与尾批

`global_batch_size` 在 1、2、4 Worker 间固定。每个 epoch 使用相同 seed 生成唯一全局乱序，随后每个全局 batch 通过 `numpy.array_split` 分成互不重叠的 Worker 分片。各 Worker 的局部 loss 是均值；PS 使用 `local_sample_count` 加权，因此不等长尾批与单进程全局均值梯度等价。

由于同步 PS 要求每一步所有 Worker 都提交正样本数，如果最后一个全局 batch 的样本数小于 Worker 数，程序会在启动进程前明确报错，不会复制或静默丢弃样本。

## 运行真实 MNIST

原始 gzip 文件应位于 `data/MNIST/raw/`。CPU 示例：

```powershell
python -m apps.distributed_mnist `
  --workers 2 `
  --epochs 5 `
  --global-batch-size 128 `
  --device cpu `
  --metrics runs/distributed_mnist/metrics.jsonl
```

CUDA 示例（每个本地进程使用当前实现的 `cuda:0`，需要可用的 CuPy/CUDA 环境）：

```powershell
python -m apps.distributed_mnist --workers 2 --epochs 5 --device cuda
```

CPU-only 安装始终可以导入并运行；CuPy 仅在选择 `--device cuda` 后延迟导入。Socket 边界上的梯度 D2H 与参数 H2D 拷贝仍由 V12.2 显式执行。

无需 MNIST 的快速 smoke：

```powershell
python -m apps.distributed_mnist `
  --workers 2 --epochs 2 --global-batch-size 20 `
  --synthetic-samples 40 --device cpu
```

## 指标语义

Monitor 每个 Worker、每个 step 写一条记录：`data_time`、`forward_time`、`backward_time`、`gradient_serialize_time`、`gradient_upload_time`、`parameter_wait_time`、`parameter_download_time`、`optimizer_time`、`total_step_time`、`samples_per_second`、`upload_bytes`、`download_bytes`、`loss`、`accuracy`、`worker_id` 和 `parameter_version`，并附带 `epoch`、`step`。

`parameter_wait_time` 是 Socket 等待响应累计时间；`parameter_download_time` 是最终参数响应的 JSON 解码与显式装载耗时。`upload_bytes`/`download_bytes` 包含 4 字节帧头。`optimizer_time` 由 PS 实测并随参数版本返回。每条记录的 `samples_per_second` 是 Worker 本地样本吞吐；汇总表使用全局样本总数除以包含 spawn、通信和监控的端到端时间，不据此声称线性加速。

## Benchmark

以下命令先运行一个无 PS、Socket、子进程和 Monitor 的纯单进程基线，
再依次实测 1、2、4 Worker，并打印汇总表：

```powershell
python -m benchmarks.bench_distributed `
  --model mlp --workers 1 2 4 --samples 512 `
  --global-batch-size 128 --device cpu
```

所有模式使用相同的 MLP、初始化 seed、合成数据、SGD、epoch 数和全局
batch。`speedup` 是实测吞吐相对于纯单进程基线的比值。分布式结果 JSONL
写入 `runs/distributed_benchmark/`（已被 Git 忽略）。进程启动和 JSON 数组
传输开销可能让多 Worker 比纯单进程更慢；脚本如实输出这种结果。

如果只想测量分布式 1/2/4 Worker 之间的 strong scaling，可以增加
`--skip-single-process`。

要增加计算量并复用 `apps.lenet5_mnist.Net` 的两层卷积模型，可运行：

```powershell
python -m benchmarks.bench_distributed `
  --model lenet5 --workers 1 2 4 --samples 1024 `
  --global-batch-size 256 --epochs 1 --device cpu
```

LeNet-5 模式约有 120 万参数，Socket+JSON 每一步的通信量也显著高于小型
MLP。它能提高卷积计算量，但不保证抵消文本序列化和完整参数广播成本；
应以汇总表和 JSONL 阶段计时为准。

## 验证

```powershell
python -m pytest tests/test_distributed_protocol.py -q
python -m pytest tests/test_parameter_server.py -q
python -m pytest tests/test_distributed_launcher.py -q
```

smoke test 使用 1 个 PS、2 个 Worker 和 1 个 Monitor 的真实 spawn 子进程，验证损失下降、指标完整、参数版本推进且所有进程正常退出。真实 MNIST 与多进程 CUDA 路径需要在相应硬件/数据环境中另外验证。
