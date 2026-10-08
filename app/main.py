import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app import config, memory
from app.audio import decode_audio, preprocess_audio
from app.logging_config import request_id_var, setup_logging
from app.model import EmotionModel

setup_logging(config.LOG_LEVEL)
logger = logging.getLogger("vera.api")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    start = time.perf_counter()
    app.state.model = EmotionModel(config.MODEL_PATH, config.LABEL_ENCODER_PATH)
    logger.info(
        f"Ready in {_ms(start) / 1000:.1f}s | memory {memory.fmt(memory.rss_mb())} "
        f"| onnxruntime {ort.__version__} ({config.INFERENCE_THREADS} thread) "
        f"| labels={list(app.state.model.idx_to_label.values())} | cors={config.CORS_ORIGINS}"
    )
    yield
    logger.info("Shutting down")


app = FastAPI(title="VERA Emotion Prediction API", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Reuse the caller's id (the Next.js proxy sends one) so both terminals can be correlated.
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    client = request.client.host if request.client else "-"
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"{request.method} {request.url.path} crashed after {_ms(start):.0f} ms")
        raise
    else:
        level = logging.ERROR if response.status_code >= 500 else logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(level, f"{request.method} {request.url.path} → {response.status_code} in {_ms(start):.0f} ms (client {client})")
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)


# Added last, so CORS is the outermost layer and answers preflight OPTIONS before the logger sees it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.get("/health")
def health(request: Request):
    model: EmotionModel = request.app.state.model
    return {"status": "ok", "device": model.device, "labels": list(model.idx_to_label.values())}


# One inference at a time: concurrent requests would each hold their own activations and
# multiply peak memory. Later requests wait for the lock instead.
_inference_lock = threading.Lock()


def _run_inference(model: EmotionModel, data: bytes, suffix: str) -> dict:
    t_wait = time.perf_counter()
    with _inference_lock:
        waited = _ms(t_wait)
        if waited > 50:
            logger.info(f"Waited {waited:.0f} ms for a previous request to finish")
        rss_before = memory.rss_mb()
        memory.reset_peak()

        t0 = time.perf_counter()
        samples, duration = decode_audio(data, suffix)
        t_decode = _ms(t0)
        t1 = time.perf_counter()
        input_values = preprocess_audio(samples)
        del samples
        t_preprocess = _ms(t1)
        t2 = time.perf_counter()
        result = model.predict(input_values)
        t_inference = _ms(t2)
        peak = memory.peak_mb()

    analyzed = min(duration, config.MAX_LENGTH / config.SAMPLE_RATE)
    truncated = f" (only first {analyzed:.0f}s analysed)" if analyzed < duration else ""
    logger.info(f"Audio {duration:.2f}s{truncated} → input {tuple(input_values.shape)}")
    result["audio_seconds"] = round(duration, 2)
    result["analyzed_seconds"] = round(analyzed, 2)
    logger.info(
        f"Result {result['emotion']} ({result['emotion_confidence']:.0%}) | "
        f"V={result['valence']:.2f} A={result['arousal']:.2f} D={result['dominance']:.2f} | "
        f"trust={result['trust']:.2f} confidence={result['confidence']:.2f}"
    )
    logger.info(f"Timings decode={t_decode:.0f}ms preprocess={t_preprocess:.0f}ms inference={t_inference:.0f}ms")
    logger.info(
        f"Memory before={memory.fmt(rss_before)} after={memory.fmt(memory.rss_mb())} "
        f"peak={memory.fmt(peak)} (this request)"
    )
    return result


@app.post("/predict")
async def predict_emotion(request: Request, file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    data = await file.read()
    logger.info(f"Upload '{filename}' ({file.content_type}, {len(data) / 1024:.1f} KB)")

    if suffix not in config.ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected '{filename}': extension '{suffix}' not in {sorted(config.ALLOWED_EXTENSIONS)}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}",
        )
    if len(data) > config.MAX_UPLOAD_BYTES:
        logger.warning(f"Rejected '{filename}': {len(data) / 1024**2:.1f} MB exceeds limit")
        raise HTTPException(status_code=413, detail=f"File is larger than {config.MAX_UPLOAD_BYTES // 1024**2} MB")
    if not data:
        logger.warning(f"Rejected '{filename}': empty upload")
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        # Decoding and inference are blocking; keep them off the event loop.
        return await run_in_threadpool(
            _run_inference, request.app.state.model, data, suffix
        )
    except Exception as e:
        logger.exception(f"Prediction failed for '{filename}': {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
