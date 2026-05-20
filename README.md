Model_Eye
=========

This repository contains the eye-based model service.

Contents to copy:
- `model_eye/` files: `eye_api.py`, `ResNet18.py`, `train.py`, `pretrained/`.

Quick start:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m eye_api
```

Notes:
- Configure `MODEL_PRETRAINED_PATH` to point to `pretrained/`.
- GPU: install CUDA PyTorch wheel and adjust Dockerfile.
