# KernelLeaf AutoDrive 训练面板

前端页面沿用 `auto-drive(paddle)` 的四页布局与视觉样式，只将数据接口改接
KernelLeaf V11 的 run/JSONL 协议。训练监控页可以编辑 manifest、地图、设备和优化参数，
并启动、暂停、恢复或停止一个由后端管理的 KernelLeaf 训练进程。manifest 重新生成和
Grad-CAM 生成仍从命令行执行。

从页面启动训练时，manifest 必须位于项目的 `data/DonkeyCar/` 内，并且必须明确填写
一个地图名称。一次训练只会读取这个地图的数据，不接受空值、列表或逗号分隔的多个地图。
停止训练会终止当前进程，已完成 epoch 的 checkpoint 会保留。

## 开发模式

在项目根目录安装后端依赖，并生成一份不需要模型和数据集的演示记录：

```powershell
pip install -e ".[dashboard]"
python -m apps.autodrive dashboard-demo --run-id dashboard-demo
python -m apps.autodrive dashboard --runs-root runs
```

另开终端启动 Vue：

```powershell
cd apps/autodrive/dashboard
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。Vite 会把 `/api` 和 `/ws` 转发到
`http://127.0.0.1:8000`。

## 生产构建

```powershell
cd apps/autodrive/dashboard
npm run build
cd ../../..
python -m apps.autodrive dashboard --runs-root runs
```

构建后 FastAPI 会直接托管 `dashboard/dist`，访问 `http://127.0.0.1:8000` 即可。
`dist`、`node_modules` 和所有 run 日志均被 Git 忽略。

## 真实训练

训练程序默认创建 `runs/autodrive_<地图>_<时间>/`。为了让 Grad-CAM 和训练记录进入
同一实验，建议显式指定名称：

```powershell
python -m apps.autodrive train `
  --manifest data/DonkeyCar/collected_manifest.jsonl `
  --map warren-track `
  --device cuda `
  --epochs 10 `
  --run-name warren-v11

python -m apps.autodrive gradcam `
  --config checkpoints/autodrive_warren-track.json `
  --image data/DonkeyCar/collected/warren-track/具体RUN/00000000.png `
  --output runs/warren-v11/gradcam/steering.png `
  --head steering `
  --device cuda
```

不需要监控记录时增加 `--no-monitor-log`，训练、checkpoint 和推理 JSON 的行为不变。

## Run 文件协议

```text
runs/<run-id>/
├── run.json             # 配置、状态、开始/结束时间
├── metrics.jsonl        # 每个 epoch 一行，追加写入
├── latest.json          # 最新指标快照
├── validation.json      # 最新验证结果
├── data_summary.json    # 地图、划分和标签分布
└── gradcam/             # 可选解释图
```

`metrics.jsonl` 包含 train/val loss、steering/throttle loss、MAE、可选 accuracy、
学习率、step、epoch 耗时、累计耗时、CPU 内存和可用时的 GPU 显存。损坏的单行会被 API
隔离并报告，不会使整个 run 无法查看。

## 验证

```powershell
python -m pytest tests/test_autodrive_v11.py -q
cd apps/autodrive/dashboard
npm test
npm run build
npm audit --audit-level=moderate
```
