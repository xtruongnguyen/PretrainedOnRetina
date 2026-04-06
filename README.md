---
title: PretrainedOnRetina
sdk: docker
app_port: 8000
---

# Retina Disease Classification – FastAPI Inference Server

This repository provides a minimal FastAPI server to deploy your trained ViT model for retina OCT disease classification.

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-ffcc4d?logo=huggingface)](https://xtn11-pretrainedonretina.hf.space/)

Live app: https://xtn11-pretrainedonretina.hf.space/

Space page: https://huggingface.co/spaces/xtn11/PretrainedOnRetina

## Prepare the model

After training in your notebook/script, save artifacts to a directory, for example `my-trained-vit-model/`:

```python
model.save_pretrained('my-trained-vit-model')
processor.save_pretrained('my-trained-vit-model')
```

Ensure the directory contains files like `config.json`, `pytorch_model.bin` (or safetensors), `preprocessor_config.json`, etc.

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run locally

```bash
set MODEL_DIR=my-trained-vit-model  # Windows
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` to use the simple HTML upload form.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Predict with cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict?top_k=3" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg"
```

Sample JSON response:

```json
{
  "predictions": [
    {"label": "CNV", "score": 0.91},
    {"label": "DME", "score": 0.06},
    {"label": "DRUSEN", "score": 0.02}
  ]
}
```

## Notes
- The server first tries `MODEL_DIR` (defaults to `my-trained-vit-model` in the repo root).
- If that local path does not exist and `HF_MODEL_ID` is set, the server downloads model artifacts from that Hugging Face model repository at startup.
- For private model repositories, set `HF_TOKEN`.
- CPU inference by default. If you want GPU, move tensors to CUDA and ensure PyTorch with CUDA is installed.

## Deploy

These deployment options are written for anyone using this repository (local users, collaborators, and public GitHub viewers).

Deployment files included in this repository:
- `Dockerfile` for container deployment
- `.dockerignore` to keep images smaller
- `Procfile` for PaaS platforms that use process files

### Option A: Deploy with Docker

Build the image:

```bash
docker build -t retina-api .
```

Run with a mounted model directory:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_DIR=/models/my-trained-vit-model \
  -v $(pwd)/my-trained-vit-model:/models/my-trained-vit-model:ro \
  retina-api
```

PowerShell volume example:

```powershell
docker run --rm -p 8000:8000 -e MODEL_DIR=/models/my-trained-vit-model -v ${PWD}\my-trained-vit-model:/models/my-trained-vit-model:ro retina-api
```

### Option B: Deploy to PaaS (Render/Railway/Heroku-like)

Use these environment variables:
- `MODEL_DIR` (optional): absolute or app-relative local path to model artifacts
- `HF_MODEL_ID` (recommended): Hugging Face model repo id such as `xtn11/pretrained-on-retina-model`
- `HF_MODEL_REVISION` (optional): branch/tag/commit from model repo
- `HF_TOKEN` (optional): required only for private model repos
- `PORT` (provided by most platforms)

Start command (already in `Procfile`):

```text
web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Option C: Deploy to Hugging Face Spaces (Docker)

Recommended setup: keep this app code in your Space repo and keep model weights in a separate Hugging Face **Model** repo.

1. Create a Hugging Face **Model** repository (example: `xtn11/pretrained-on-retina-model`).
2. Upload model files to that model repo:
  - `config.json`
  - `model.safetensors`
  - `preprocessor_config.json`
3. Create a new Space with **SDK = Docker**.
4. At **Storage Bucket**, choose **Continue without bucket** (not required for this setup).
5. In the Space repository, ensure `README.md` starts with this frontmatter:

```yaml
---
title: PretrainedOnRetina
sdk: docker
app_port: 8000
---
```

6. In Space **Settings -> Variables**, set:

```text
HF_MODEL_ID=xtn11/pretrained-on-retina-model
```

Optional variables:
- `HF_MODEL_REVISION` (pin a specific model version)
- `HF_TOKEN` (only if the model repo is private)

7. Commit and push this app code to your Space:

```bash
git add .
git commit -m "Deploy Retina API to HF Space"
git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
git push hf main
```

8. Wait for build to finish, then verify:
  - `https://<your-space>.hf.space/health`
  - `https://<your-space>.hf.space/`

### Deployment checklist

- Choose one model source:
  - local path via `MODEL_DIR`, or
  - Hugging Face model repo via `HF_MODEL_ID`
- Set `HF_TOKEN` only if your model repo is private.
- Expose port `8000` locally, or use platform `PORT` in cloud deployment.
- Verify health at `/health` after deploy.
