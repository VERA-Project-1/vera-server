import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "hubert_heads_test_7_lora"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_DIR / "hubert_emotion_vad_30hrs_jit.pt"))
LABEL_ENCODER_PATH = Path(os.environ.get("LABEL_ENCODER_PATH", DEFAULT_MODEL_DIR / "label_encoder.pkl"))
DEVICE = os.environ.get("DEVICE") or None  # None = auto (cuda if available)
PROCESSOR_NAME = os.environ.get("PROCESSOR_NAME", "facebook/wav2vec2-base-960h")

SAMPLE_RATE = 16000
MAX_LENGTH = SAMPLE_RATE * 8  # 8 second cap
ALLOWED_EXTENSIONS = {".webm", ".wav", ".mp3", ".ogg", ".m4a", ".flac"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

TRUST_EMOTION_PRIOR = {
    "neutral": 0.9,
    "happy": 0.85,
    "sad": 0.6,
    "angry": 0.2,
}
