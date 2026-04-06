from typing import List, Dict

import os
import io

import torch
from huggingface_hub import snapshot_download
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor


MODEL_DIR_ENV = "MODEL_DIR"
HF_MODEL_ID_ENV = "HF_MODEL_ID"
HF_MODEL_REVISION_ENV = "HF_MODEL_REVISION"
HF_TOKEN_ENV = "HF_TOKEN"
DEFAULT_MODEL_DIR = os.getenv(MODEL_DIR_ENV, "my-trained-vit-model")


def resolve_model_dir(default_model_dir: str) -> str:
    if os.path.isdir(default_model_dir):
        return default_model_dir

    hf_model_id = os.getenv(HF_MODEL_ID_ENV)
    if not hf_model_id:
        raise FileNotFoundError(
            f"Model directory not found: {default_model_dir}. "
            f"Set {MODEL_DIR_ENV} to a local model path or set {HF_MODEL_ID_ENV} "
            "to a Hugging Face model repository (for example: username/model-repo)."
        )

    hf_token = os.getenv(HF_TOKEN_ENV)
    hf_revision = os.getenv(HF_MODEL_REVISION_ENV)
    return snapshot_download(repo_id=hf_model_id, revision=hf_revision, token=hf_token)


def load_model_and_processor(model_dir: str):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)

    model.eval()
    return model, processor


def get_id2label(model) -> Dict[int, str]:
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        normalized: Dict[int, str] = {}
        for k, v in id2label.items():
            try:
                normalized[int(k)] = str(v)
            except Exception:
                continue
        if len(normalized) == len(id2label):
            # If labels look like default placeholders (e.g., LABEL_0), replace with domain labels
            looks_generic = all(str(v).upper().startswith("LABEL_") for v in normalized.values())
            if looks_generic and len(normalized) == 4:
                return {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}
            return normalized

    # Fallback for OCT 2017 
    return {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}


app = FastAPI(title="Retina Disease Classification API", version="1.0.0")


@app.on_event("startup")
def _startup():
    global model, processor, id2label
    model_dir = resolve_model_dir(DEFAULT_MODEL_DIR)
    model, processor = load_model_and_processor(model_dir)
    id2label = get_id2label(model)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/labels")
def labels() -> Dict[str, str]:
    # Expose the mapping for clients/UI
    return {str(k): v for k, v in id2label.items()}


@app.get("/")
def index_form() -> HTMLResponse:
    html = (
        """
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Retina Disease Classifier</title>
            <style>
              :root {
                --bg: #0f172a; /* slate-900 */
                --card: #111827; /* gray-900 */
                --muted: #94a3b8; /* slate-400 */
                --text: #e5e7eb; /* gray-200 */
                --primary: #22d3ee; /* cyan-400 */
                --primary-700: #0891b2; /* cyan-700 */
                --ring: rgba(34, 211, 238, 0.35);
                --danger: #ef4444;
              }
              * { box-sizing: border-box; }
              body {
                margin: 0; background: linear-gradient(180deg, #0b1220, #0f172a);
                color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif;
                min-height: 100vh; display: grid; place-items: center; padding: 24px;
              }
              .container {
                width: 100%; max-width: 920px; display: grid; gap: 16px;
              }
              .header {
                display: flex; align-items: center; gap: 12px; justify-content: center; flex-wrap: wrap; text-align: center;
              }
              .logo {
                width: 40px; height: 40px; border-radius: 10px; background: radial-gradient(120% 120% at 20% 20%, #22d3ee, #6366f1 60%, transparent 61%);
                box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset, 0 10px 30px rgba(34, 211, 238, 0.25);
              }
              .title { font-size: 24px; font-weight: 700; letter-spacing: 0.3px; }
              .subtitle { color: var(--muted); font-size: 14px; }

              .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
              @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

              .card {
                background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px; padding: 18px; box-shadow: 0 10px 40px rgba(0,0,0,0.35);
                backdrop-filter: blur(6px);
              }

              .dropzone {
                border: 2px dashed rgba(255,255,255,0.12); border-radius: 16px; padding: 22px; text-align: center; transition: 160ms ease;
              }
              .dropzone.dragover { border-color: var(--primary); background: rgba(34, 211, 238, 0.06); box-shadow: 0 0 0 4px var(--ring); }
              .hint { color: var(--muted); font-size: 13px; margin-top: 8px; }
              .actions { margin-top: 12px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
              .btn {
                background: var(--primary); color: #0a0a0a; font-weight: 700; border: none; border-radius: 12px; padding: 10px 14px; cursor: pointer;
                box-shadow: 0 10px 30px rgba(34, 211, 238, 0.3); transition: 160ms ease; font-size: 14px;
              }
              .btn:hover { transform: translateY(-1px); box-shadow: 0 14px 36px rgba(34, 211, 238, 0.4); }
              .btn.secondary { background: transparent; color: var(--text); border: 1px solid rgba(255,255,255,0.12); box-shadow: none; }

              .preview {
                margin-top: 14px; background: #0b1020; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 10px;
                display: grid; grid-template-columns: 130px 1fr; gap: 14px; align-items: center;
              }
              .preview img { width: 100%; height: 110px; object-fit: cover; border-radius: 10px; }
              .meta { font-size: 12px; color: var(--muted); }
              .meta b { color: var(--text); }

              .results h3 { margin: 0 0 10px 0; font-size: 16px; }
              .pill {
                display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-radius: 12px; margin-bottom: 8px;
                background: rgba(34, 211, 238, 0.06); border: 1px solid rgba(34, 211, 238, 0.25);
              }
              .label { font-weight: 700; letter-spacing: 0.3px; }
              .score { font-variant-numeric: tabular-nums; font-weight: 700; color: var(--primary); }
              .bar { height: 8px; background: rgba(255,255,255,0.09); border-radius: 999px; overflow: hidden; }
              .bar > div { height: 100%; background: linear-gradient(90deg, #22d3ee, #6366f1); }

              .loading { display: none; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; margin-top: 12px; }
              .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.2); border-top-color: var(--primary); border-radius: 50%; animation: spin 1s linear infinite; }
              @keyframes spin { to { transform: rotate(360deg); } }

              .error { display: none; color: var(--danger); font-size: 14px; margin-top: 10px; }
              footer { text-align: center; color: var(--muted); font-size: 12px; margin-top: 8px; }
            </style>
          </head>
          <body>
            <div class="container">
              <div class="header">
                <div class="logo"></div>
                <div>
                  <div class="title">Retina Disease Classifier</div>
                  <div class="subtitle">Upload an OCT image to predict CNV / DME / DRUSEN / NORMAL</div>
                </div>
              </div>

              <div class="grid">
                <div class="card">
                  <div id="dropzone" class="dropzone">
                    <div><b>Drag & drop</b> an image here or</div>
                    <div class="actions"><button id="browseBtn" class="btn">Choose image</button></div>
                    <div class="hint">Supported: JPG, PNG. Max ~10 MB (browser dependent).</div>
                    <input id="fileInput" type="file" accept="image/*" style="display:none" />
                  </div>

                  <div id="preview" class="preview" style="display:none">
                    <img id="previewImg" alt="preview" />
                    <div class="meta" id="fileMeta"></div>
                  </div>

                  <div class="loading" id="loading"><div class="spinner"></div><span>Running inference…</span></div>
                  <div class="error" id="error"></div>
                </div>

                <div class="card results">
                  <h3>Predictions</h3>
                  <div id="results">Upload an image to see predictions.</div>
                </div>
              </div>

            </div>

            <script>
              const dropzone = document.getElementById('dropzone');
              const browseBtn = document.getElementById('browseBtn');
              const fileInput = document.getElementById('fileInput');
              const preview = document.getElementById('preview');
              const previewImg = document.getElementById('previewImg');
              const fileMeta = document.getElementById('fileMeta');
              const loading = document.getElementById('loading');
              const errorBox = document.getElementById('error');
              const results = document.getElementById('results');

              function setLoading(state) { loading.style.display = state ? 'flex' : 'none'; }
              function setError(msg) { errorBox.textContent = msg || ''; errorBox.style.display = msg ? 'block' : 'none'; }
              function setResults(html) { results.innerHTML = html; }

              function bytesToSize(bytes) {
                const sizes = ['B', 'KB', 'MB', 'GB'];
                if (bytes === 0) return '0 B';
                const i = Math.floor(Math.log(bytes) / Math.log(1024));
                return (bytes / Math.pow(1024, i)).toFixed(2) + ' ' + sizes[i];
              }

              function previewFile(file) {
                const url = URL.createObjectURL(file);
                previewImg.src = url;
                fileMeta.innerHTML = `<b>${file.name}</b><br/>${file.type || 'image'} • ${bytesToSize(file.size)}`;
                preview.style.display = 'grid';
              }

              async function predict(file) {
                setError(''); setResults(''); setLoading(true);
                try {
                  const form = new FormData();
                  form.append('file', file, file.name || 'upload.jpg');
                  const resp = await fetch('/predict?top_k=4', { method: 'POST', body: form });
                  if (!resp.ok) throw new Error(`Request failed (${resp.status})`);
                  const data = await resp.json();
                  const list = (data.predictions || []).map(p => {
                    const pct = Math.max(0, Math.min(100, Math.round(p.score * 100)));
                    return `
                      <div class=\"pill\"> 
                        <div class=\"label\">${p.label}</div>
                        <div class=\"score\">${pct}%</div>
                      </div>
                      <div class=\"bar\"><div style=\"width:${pct}%\"></div></div>
                    `;
                  }).join('');
                  setResults(list || 'No predictions returned.');
                } catch (e) {
                  setError(e.message || 'Prediction failed');
                } finally {
                  setLoading(false);
                }
              }

              function handleFiles(files) {
                if (!files || !files.length) return;
                const file = files[0];
                previewFile(file);
                predict(file);
              }

              browseBtn.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });
              fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

              ;['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, (e) => {
                e.preventDefault(); e.stopPropagation(); dropzone.classList.add('dragover');
              }));
              ;['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, (e) => {
                e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('dragover');
              }));
              dropzone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
            </script>
          </body>
        </html>
        """
    )
    return HTMLResponse(content=html)


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def format_predictions(probs: torch.Tensor, top_k: int = 3) -> List[Dict[str, float]]:
    top_k = max(1, min(top_k, probs.shape[-1]))
    values, indices = torch.topk(probs, k=top_k)
    results: List[Dict[str, float]] = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        label = id2label.get(int(idx), str(idx))
        results.append({"label": label, "score": float(score)})
    return results


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        probs = softmax(outputs.logits.squeeze(0))

    preds = format_predictions(probs, top_k=top_k)
    return JSONResponse({"predictions": preds})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
