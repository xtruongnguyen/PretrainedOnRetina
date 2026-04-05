---
title: PretrainedOnRetina
sdk: docker
app_port: 8000
---

# Retina Disease Classification – FastAPI Inference Server

This repository provides a minimal FastAPI server to deploy your trained ViT model for retina OCT disease classification.

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
- The server reads `MODEL_DIR` env var; defaults to `my-trained-vit-model` in the repo root.
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
- `MODEL_DIR` (required): absolute or app-relative path to model artifacts
- `PORT` (provided by most platforms)

Start command (already in `Procfile`):

```text
web: uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Option C: Deploy to Hugging Face Spaces (Docker)

This repo is already set up to include `my-trained-vit-model/` inside the Docker image.

1. Create a new Space with **SDK = Docker**.
2. At **Storage Bucket**, choose **Continue without bucket** (not required for this setup).
3. In the Space repository, ensure `README.md` starts with this frontmatter:

```yaml
---
title: Retina API
sdk: docker
app_port: 8000
---
```

4. From your local repo, track large model files (one-time):

```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
```

5. Commit and push to your Space:

```bash
git add .
git commit -m "Deploy Retina API to HF Space"
git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
git push hf main
```

6. In Space **Settings -> Variables**, optionally set:

```text
MODEL_DIR=/app/my-trained-vit-model
```

  The Docker image already sets this default, so this variable is optional.

7. Wait for build to finish, then verify:
  - `https://<your-space>.hf.space/health`
  - `https://<your-space>.hf.space/`

### Deployment checklist

- Upload or mount your trained model directory so the API can read it.
- Set `MODEL_DIR` to that path.
- Expose port `8000` locally, or use platform `PORT` in cloud deployment.
- Verify health at `/health` after deploy.

