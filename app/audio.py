import logging
import os
import tempfile
import time

import librosa
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2Processor

from app.config import MAX_LENGTH, PROCESSOR_NAME, SAMPLE_RATE

logger = logging.getLogger("vera.audio")


def load_processor() -> Wav2Vec2Processor:
    start = time.perf_counter()
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
    logger.info(f"Processor {PROCESSOR_NAME} loaded in {time.perf_counter() - start:.1f}s")
    return processor


def convert_to_wav(data: bytes, suffix: str) -> str:
    """Convert uploaded audio bytes to a temporary mono 16 kHz wav file path."""
    logger.debug(f"Converting {suffix} to wav ({len(data)} bytes)")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(data)
        tmp_in_path = tmp_in.name

    # Distinct name: for .wav uploads the input temp file already ends in ".wav".
    tmp_wav_path = os.path.splitext(tmp_in_path)[0] + "_16k.wav"
    try:
        # webm is decoded exactly as in the original app.py; other containers are left to ffmpeg's probe
        # (e.g. "m4a" is not a valid ffmpeg input format name).
        audio = AudioSegment.from_file(tmp_in_path, format="webm" if suffix == ".webm" else None)
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
        audio.export(tmp_wav_path, format="wav")
        logger.debug(f"Converted to wav: duration={len(audio) / 1000:.2f}s")
    finally:
        os.unlink(tmp_in_path)

    return tmp_wav_path


def preprocess_audio(processor: Wav2Vec2Processor, wav_path: str) -> tuple[torch.Tensor, float]:
    """Returns model input values and the clip duration in seconds (before truncation)."""
    speech_array, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    duration = len(speech_array) / SAMPLE_RATE
    logger.debug(f"Loaded audio: {duration:.2f}s ({len(speech_array)} samples)")
    inputs = processor(
        speech_array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    logger.debug(f"Preprocessed tensor shape: {inputs.input_values.shape}")
    return inputs.input_values, duration
