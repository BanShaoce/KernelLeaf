# 环境安装（Windows 教程）

下面的版本来自 2026-09-09 对本机 `donkey-env2` 的实际检查。KernelLeaf 的 CPU
后端只依赖 NumPy；CUDA、Triton、自动驾驶模拟器和 Dashboard 都是按需安装的可选功能。

### 1. 已验证的重要版本

| 类别 | 软件包 | 版本 | 用途 |
| --- | --- | --- | --- |
| 基础 | Python | `3.9.24` | 当前课程与模拟器环境 |
| 基础 | NumPy | `1.26.4` | CPU 张量和算子后端 |
| 测试 | pytest | `8.4.2` | 单元测试 |
| 测试 | pytest-cov | `7.1.0` | 覆盖率 |
| CUDA | CuPy (`cupy-cuda12x`) | `13.6.0` | CUDA 数组和自定义 RawKernel |
| CUDA | CUDA runtime wheel | `12.6.77` | CuPy CUDA 12 运行时组件 |
| CUDA | CUDA NVRTC wheel | `12.6.85` | 运行时编译 CUDA kernel |
| Triton | triton-windows | `3.2.0.post21` | Windows 上的融合算子和 Conv2d kernel |
| 图像 | Pillow | `11.1.0` | MNIST/AutoDrive 图片读取与增强 |
| 采集 | pygame | `2.6.1` | 自动驾驶键盘采集窗口 |
| 模拟器 | gym | `0.22.0` | DonkeyCar 环境接口 |
| 模拟器 | gym-donkeycar | `1.3.1` | DonkeyCar 模拟器适配 |
| Dashboard | FastAPI | `0.128.8` | 监控后端 |
| Dashboard | Uvicorn | `0.39.0` | ASGI 服务 |
| Dashboard | psutil | `7.1.3` | 资源监控、训练暂停与恢复 |
| Dashboard | Pydantic | `2.13.4` | API 数据模型 |
| Dashboard | HTTPX | `0.28.1` | 后端 API 测试 |
| Dashboard | websockets | `15.0.1` | 实时训练状态 |

本机前端验证使用 Node.js `24.14.1` 和 npm `11.11.0`。`npm ci` 会按照
`apps/autodrive/dashboard/package-lock.json` 安装锁定版本，目前主要包括 Vue
`3.5.42`、Vite `7.3.6`、ECharts `6.1.0` 和 TypeScript `5.9.3`。

本机 GPU 是 NVIDIA GeForce RTX 4060 Laptop GPU（8188 MiB），驱动版本为
`560.94`。这只是实测硬件，不要求其他机器必须使用同一型号；CUDA 路径必须使用
NVIDIA GPU，Apple Silicon 和纯 AMD 环境请使用 NumPy CPU 后端。

> **不要安装 PyTorch 或 Paddle 来实现 KernelLeaf。** 当前旧环境虽然还包含
> `torch`、`torchvision` 和 `paddlepaddle-gpu`，但它们是历史实验遗留项，不是
> KernelLeaf 的运行依赖，也不会被框架用于 Tensor、自动微分或算子实现。

### 2. 准备 Conda 和 PowerShell

确认 Conda 可以使用：

```powershell
conda --version
```

如果 PowerShell 提示无法识别 `conda`，执行：

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" init powershell
```

关闭并重新打开 PowerShell。只让当前窗口临时生效也可以执行：

```powershell
& "C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1"
```

### 3. 创建或进入环境

第一次安装时创建环境：

```powershell
conda create -n donkey-env2 python=3.9.24 pip=25.2 -y
conda activate donkey-env2
```

如果环境已经存在，只需：

```powershell
conda activate donkey-env2
cd C:\code\Mytorch
```

务必确认后续命令使用的是该环境的 Python：

```powershell
python --version
where.exe python
python -m pip --version
```

期望 `where.exe python` 的第一项类似：

```text
C:\ProgramData\anaconda3\envs\donkey-env2\python.exe
```

### 4. 安装 CPU、测试和 KernelLeaf

先安装固定版本的基础包，再以 editable 模式安装项目。`--no-deps` 可以避免 pip
重新选择与本教程不同的 NumPy 版本：

```powershell
python -m pip install "setuptools==80.9.0" "wheel==0.45.1"
python -m pip install "numpy==1.26.4" "pytest==8.4.2" "pytest-cov==7.1.0"
python -m pip install -e . --no-deps
```

只需要学习 CPU Tensor、自动微分和基础算子时，到这里就可以运行：

```powershell
python -m examples.device_smoke --cpu-only
python -m pytest -q
```

### 5. 安装 AutoDrive 图片、采集和模拟器依赖

```powershell
python -m pip install `
  "Pillow==11.1.0" `
  "pygame==2.6.1" `
  "gym==0.22.0" `
  "gym-donkeycar==1.3.1"
```

检查入口是否可以导入：

```powershell
python -m apps.autodrive --help
python -m apps.autodrive train --help
python -m apps.autodrive collect --help
```

运行采集或闭环驾驶前，还需要单独启动 DonkeyCar Unity 模拟器；Python 包不会自动
安装或启动 Unity 程序。

### 6. 安装 Dashboard 后端

```powershell
python -m pip install `
  "fastapi==0.128.8" `
  "starlette==0.49.3" `
  "pydantic==2.13.4" `
  "uvicorn==0.39.0" `
  "websockets==15.0.1" `
  "psutil==7.1.3" `
  "httpx==0.28.1"
```

启动已经构建好的 Dashboard：

```powershell
python -m apps.autodrive dashboard --runs-root runs
```

浏览器打开 `http://127.0.0.1:8000`。从页面启动的训练使用 `sys.executable` 创建
子进程，因此只要 Dashboard 是在 `donkey-env2` 中启动的，训练也会使用同一个环境。

### 7. 安装 Dashboard 前端

先安装 Node.js；本项目已验证 Node.js `24.14.1`，建议使用 Node 24：

```powershell
winget install --id OpenJS.NodeJS.LTS
```

安装后重新打开 PowerShell，然后执行：

```powershell
cd C:\code\Mytorch\apps\autodrive\dashboard
node --version
npm --version
npm ci
npm run build
npm test
cd C:\code\Mytorch
```

开发模式需要两个 PowerShell 窗口：

```powershell
# 窗口 1：donkey-env2
python -m apps.autodrive dashboard --runs-root runs
```

```powershell
# 窗口 2
cd C:\code\Mytorch\apps\autodrive\dashboard
npm run dev
```

执行过 `npm run build` 后，只启动后端并访问 `http://127.0.0.1:8000` 即可。

### 8. 安装 CUDA 12 / CuPy（可选）

先运行 `nvidia-smi`。如果系统找不到该命令或没有 NVIDIA GPU，请跳过本节并使用
`--device cpu`。

```powershell
nvidia-smi
python -m pip install `
  "cupy-cuda12x==13.6.0" `
  "nvidia-cuda-runtime-cu12==12.6.77" `
  "nvidia-cuda-nvrtc-cu12==12.6.85"
```

验证 CuPy 和 KernelLeaf CUDA：

```powershell
python -c "import cupy; print(cupy.__version__); print(cupy.cuda.is_available()); print(cupy.cuda.runtime.getDeviceCount())"
python -m examples.device_smoke
```

本机实测输出同时包含 `cpu:` 和 `cuda:0:`，两端 loss 都会下降。CuPy 有时会打印
`CUDA path could not be detected` 警告；如果 `cupy.cuda.is_available()` 为 `True`
且 smoke 通过，该警告不代表 CUDA 没有工作。只有 smoke 失败时才需要检查 NVIDIA
驱动、CUDA Toolkit 或 `CUDA_PATH`。

使用 pip 的 CUDA Runtime/NVRTC wheel 且没有安装完整 Toolkit 时，可以将路径常驻到
PowerShell Profile。先用 `python -m pip show nvidia-cuda-runtime-cu12` 确认安装位置，
再按实际 Python 版本调整下面的 `$cudaPackages`：

```powershell
$cudaPackages = "$env:APPDATA\Python\Python39\site-packages\nvidia"
$cudaRuntime = Join-Path $cudaPackages "cuda_runtime"
$env:CUDA_PATH = $cudaRuntime
$env:Path = "$(Join-Path $cudaRuntime 'bin');$(Join-Path $cudaPackages 'cuda_nvrtc\bin');$env:Path"
```

验证无误后，将这段加入 `$PROFILE`，新开的 PowerShell 会自动生效。不要把
`CUDA_PATH` 指向不存在的目录，也不要只设置 `CUDA_PATH` 而遗漏 NVRTC 的 `bin`。

不要同时安装 `cupy`、`cupy-cuda11x` 和 `cupy-cuda12x`，环境中只能保留一个
CuPy 发行包。

### 9. 安装 Triton（可选）

Windows 使用当前环境验证过的 `triton-windows`：

```powershell
python -m pip install "triton-windows==3.2.0.post21"
python -c "import triton; print(triton.__version__)"
python -m pytest tests/test_fused_ops.py -q
```

Linux 应安装官方 `triton`，不要在 Linux 安装 `triton-windows`。Triton 不存在或
输入形状不受支持时，KernelLeaf 的 `auto` 路径会回退到 CuPy 或 eager 实现。

### 10. 最终验收

```powershell
python -m examples.device_smoke
python -m pytest -q

cd C:\code\Mytorch\apps\autodrive\dashboard
npm ci
npm run build
npm test
npm audit --audit-level=moderate
```

本次环境审计时，KernelLeaf 全量测试为 `142 passed, 1 skipped`，CPU/CUDA device
smoke 均通过；这里的数字是本机快照，不应替代其他机器自己的测试结果。

### 11. 环境污染与常见问题

检查关键包到底安装在哪里：

```powershell
python -m pip show numpy cupy-cuda12x triton-windows fastapi pytest
python -c "import sys, site; print(sys.executable); print(site.getusersitepackages())"
```

理想情况下，`Location` 应位于：

```text
...\envs\donkey-env2\Lib\site-packages
```

本机 `donkey-env2` 的部分可选包实际来自用户级
`AppData\Roaming\Python\Python39\site-packages`。这会造成“换一个 Conda 环境仍然能
导入同一个包”的错觉。重建环境时应确保 pip 没有显示 `Defaulting to user
installation`；如果 Conda 安装在 `C:\ProgramData` 且当前用户没有写权限，请使用
管理员终端安装，或者在用户目录创建一个新的 Conda 环境。

当前旧环境执行 `python -m pip check` 还会报告 Paddle CUDA 依赖和 OpenCV/NumPy
版本冲突。这些包不属于 KernelLeaf，因此干净环境不要安装 Paddle、PyTorch 或
OpenCV。若必须共用旧环境，应以 KernelLeaf 的 smoke 和 pytest 结果为准，同时不要把
`pip check` 的冲突误认为 KernelLeaf 自身依赖错误。
