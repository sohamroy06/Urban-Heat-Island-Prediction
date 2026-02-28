# ShadowMap — AMD Deployment Guide

## Architecture Overview

ShadowMap is **hardware-agnostic** by design. The inference engine runs identically on:

| Platform | XGBoost Backend | ONNX Runtime EP |
|----------|----------------|----------------|
| CPU (any) | `device="cpu"` | `CPUExecutionProvider` |
| NVIDIA GPU | `device="cuda"` | `CUDAExecutionProvider` |
| AMD GPU (ROCm) | `device="cuda"` ¹ | `ROCMExecutionProvider` |

> ¹ XGBoost uses the `"cuda"` device string for ROCm when built with the ROCm backend. The API is identical.

---

## Running on AMD Instinct GPUs (ROCm)

### 1. Install ROCm

Follow the official AMD ROCm installation guide for your OS:

```bash
# Ubuntu 22.04 / 24.04
sudo apt update
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocminfo
rocm-smi
```

### 2. Install PyTorch for ROCm

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### 3. Install XGBoost with ROCm Support

XGBoost must be built from source with ROCm support:

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942"
make -j$(nproc)
cd ../python-package
pip install .
```

> **GPU architectures**: Use `gfx90a` for MI210/MI250, `gfx942` for MI300X.

### 4. Install ONNX Runtime with ROCm EP

```bash
pip install onnxruntime-rocm
```

Or build from source for latest features:

```bash
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --use_rocm --rocm_home /opt/rocm
pip install build/Linux/Release/dist/*.whl
```

---

## Deployment on AMD EPYC Server

### Recommended Hardware

| Component | Recommendation |
|-----------|---------------|
| CPU | AMD EPYC 9004 (Genoa) or 7003 (Milan) |
| GPU | AMD Instinct MI210 / MI250X / MI300X |
| RAM | 64 GB+ DDR5 |
| Storage | NVMe SSD (for model artifacts) |

### Server Setup

```bash
# 1. Clone the project
git clone <repo-url> shadowmap
cd shadowmap/backend

# 2. Create Python environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (auto-detects GPU)
python train_model.py

# 5. Export to ONNX
python export_onnx.py

# 6. Run benchmarks
python benchmark.py

# 7. Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Production Deployment with Gunicorn

```bash
pip install gunicorn
gunicorn main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

---

## Running Without AMD Hardware

The system runs on **any hardware** without modification:

```bash
# Standard setup (CPU or NVIDIA GPU auto-detected)
pip install -r requirements.txt
python train_model.py
python main.py
```

XGBoost automatically falls back to CPU if no GPU is available. ONNX Runtime uses `CPUExecutionProvider` as the default fallback.

---

## Hardware-Agnostic Design Principles

1. **No `.cuda()` calls** — all device selection is abstracted through XGBoost's `device` parameter
2. **No CUDA-only APIs** — no `torch.cuda.synchronize()`, no CUDA-specific memory management
3. **ONNX for portable inference** — ONNX Runtime selects the best EP at runtime
4. **`tree_method="hist"`** — the histogram-based tree method works across all backends
5. **Graceful fallback** — every GPU code path falls back to CPU cleanly

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCM_HOME` | ROCm installation path | `/opt/rocm` |
| `HIP_VISIBLE_DEVICES` | GPU device indices to use | all |
| `ORT_ROCM_DEVICE_ID` | ONNX Runtime ROCm device | `0` |

---

## Verifying GPU Usage

```python
# Check XGBoost device
from model import XGB_DEVICE
print(f"XGBoost device: {XGB_DEVICE}")

# Check ONNX Runtime providers
import onnxruntime as ort
print(f"Available providers: {ort.get_available_providers()}")

# Check PyTorch/ROCm
import torch
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```
