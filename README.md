# vera-server

Lightweight FastAPI server for VERA emotion + VAD (valence/arousal/dominance) prediction,
serving the `hubert_heads_test_7_lora` TorchScript model from vera-system-models.

## Layout

```
app/
  main.py      FastAPI app, routes, model loaded once at startup
  config.py    paths & settings (env-overridable)
  audio.py     preprocessing, same as vera-system-models app.py (pydub -> librosa -> Wav2Vec2Processor)
  model.py     TorchScript model + label encoder, prediction logic
models/hubert_heads_test_7_lora/
  hubert_emotion_vad_30hrs_jit.pt
  label_encoder.pkl
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

See `.env.example`: `MODEL_PATH`, `LABEL_ENCODER_PATH`, `DEVICE` (`cpu`/`cuda`), `PROCESSOR_NAME` (default `facebook/wav2vec2-base-960h`), `LOG_LEVEL` (`INFO`; `DEBUG` adds per-step audio detail), `CORS_ORIGINS` (comma-separated).
