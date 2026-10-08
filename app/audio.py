import logging
import os
import tempfile

import numpy as np
from pydub import AudioSegment

from app.config import MAX_LENGTH, SAMPLE_RATE

logger = logging.getLogger("vera.audio")


def decode_audio(data: bytes, suffix: str) -> tuple[np.ndarray, float]:
    """Decode an upload to mono 16 kHz float32 samples in [-1, 1), plus its duration in seconds."""
    logger.debug(f"Decoding {suffix} ({len(data)} bytes)")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        # webm is decoded exactly as in the original app.py; other containers are left to ffmpeg's probe
        # (e.g. "m4a" is not a valid ffmpeg input format name).
        audio = AudioSegment.from_file(tmp_path, format="webm" if suffix == ".webm" else None)
    finally:
        os.unlink(tmp_path)

    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    if audio.sample_width == 1:
        audio = audio.set_sample_width(2)
    # Same scaling as reading the exported PCM wav back with librosa/soundfile: int / 2^(bits-1).
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= float(1 << (8 * audio.sample_width - 1))
    duration = len(samples) / SAMPLE_RATE
    logger.debug(f"Decoded {duration:.2f}s ({audio.sample_width * 8}-bit)")
    return samples, duration


def preprocess_audio(samples: np.ndarray) -> np.ndarray:
    """Model input values for one clip.

    Replicates Wav2Vec2Processor (facebook/wav2vec2-base-960h, do_normalize=True) for a single clip:
    truncate to MAX_LENGTH, then zero-mean unit-variance normalisation.
    """
    if samples.size == 0:
        raise ValueError("Audio contains no samples")
    x = samples[:MAX_LENGTH]
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    input_values = np.ascontiguousarray(x, dtype=np.float32)[np.newaxis, :]
    logger.debug(f"Preprocessed tensor shape: {tuple(input_values.shape)}")
    return input_values
