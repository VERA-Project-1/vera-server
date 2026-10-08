# vera-server

Lightweight FastAPI server for VERA emotion + VAD (valence/arousal/dominance) prediction,
serving the `hubert_heads_test_7_lora` model from vera-system-models with ONNX Runtime
(no PyTorch at runtime; ~190 MB idle, ~300 MB peak per prediction).

## Layout

```
app/
  main.py      FastAPI app, routes, model loaded once at startup
  config.py    paths & settings (env-overridable)
  audio.py     decoding + preprocessing; bit-identical to vera-system-models app.py
               (pydub -> Wav2Vec2Processor normalisation, reimplemented in numpy)
  model.py     ONNX Runtime session + label encoder, prediction logic
  memory.py    per-request memory readings for the logs
models/hubert_heads_test_7_lora/
  hubert_emotion_vad_30hrs.onnx    served model
  hubert_emotion_vad_30hrs_jit.pt  original TorchScript export (source for the .onnx)
  label_encoder.pkl
scripts/
  export_onnx.py   one-time .pt -> .onnx conversion with a parity check (needs torch + onnx)
```

## Setup

Requires Python 3.10+ and `ffmpeg` on `PATH`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

- `GET /health` — status, device, labels
- `POST /predict` — multipart `file` (`.webm`, `.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`; max 50 MB; only the first 8 s are analysed)

```bash
curl -F file=@sample.webm localhost:8000/predict
# {"emotion":"Neutral","emotion_confidence":0.36,"valence":0.76,"arousal":0.89,
#  "dominance":0.91,"trust":0.52,"confidence":0.82,"audio_seconds":5.03,"analyzed_seconds":5.03}
```

## Configuration

See `.env.example`: `MODEL_PATH`, `LABEL_ENCODER_PATH`, `INFERENCE_THREADS` (default `1`), `LOG_LEVEL` (`INFO`; `DEBUG` adds per-step audio detail), `CORS_ORIGINS` (comma-separated).

## Memory

Each prediction logs its memory use, e.g.
`Memory before=193 MB after=193 MB peak=255 MB (this request)`: the peak is reset at the start of
every request (Linux `/proc/self/clear_refs`), so it is that request's own high-water mark.

Inference runs one request at a time, so concurrent uploads queue instead of stacking memory, and
`app/__init__.py` tunes glibc's allocator at startup (`mallopt`, the same as `MALLOC_ARENA_MAX=2`,
`MALLOC_TRIM_THRESHOLD_=65536`, `MALLOC_MMAP_THRESHOLD_=65536`) so freed memory returns to the OS.

Measured locally on 16 clips (8 s worst case):

| | idle | peak per prediction |
|---|---|---|
| PyTorch + transformers (original) | 559 MB | 890 MB |
| ONNX Runtime, default allocator | 233 MB | 456 MB |
| ONNX Runtime, tuned allocator (current) | 188 MB | 302 MB |

## Re-exporting the model

The `.onnx` file is a format conversion of the `.pt` (same weights, no retraining):

```bash
.venv/bin/pip install torch onnx --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python scripts/export_onnx.py   # writes the .onnx and checks it against the .pt
```
