import logging
import pickle
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.config import INFERENCE_THREADS, TRUST_EMOTION_PRIOR

logger = logging.getLogger("vera.model")


class _LabelEncoderStub:
    """Stands in for sklearn's LabelEncoder so the pickle loads without scikit-learn."""


class _LabelEncoderUnpickler(pickle.Unpickler):

    _ALLOWED = {
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) == ("sklearn.preprocessing._label", "LabelEncoder"):
            return _LabelEncoderStub
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Disallowed global in label encoder pickle: {module}.{name}")


def load_labels(path: Path) -> dict[int, str]:
    with open(path, "rb") as f:
        encoder = _LabelEncoderUnpickler(f).load()
    return {idx: str(label) for idx, label in enumerate(encoder.classes_)}


class EmotionModel:
    def __init__(self, model_path: Path, label_encoder_path: Path):
        self.device = "cpu"
        start = time.perf_counter()
        options = ort.SessionOptions()
        options.intra_op_num_threads = INFERENCE_THREADS
        options.inter_op_num_threads = 1
        # The arena and memory patterns keep peak-sized buffers alive between requests;
        # without them memory drops back to ~230 MB after each prediction.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        size_mb = model_path.stat().st_size / 1024**2
        logger.info(f"Model {model_path.name} ({size_mb:.0f} MB) loaded in {time.perf_counter() - start:.1f}s")

        self.idx_to_label = load_labels(label_encoder_path)
        logger.info(f"Labels {self.idx_to_label}")

    def predict(self, input_values: np.ndarray) -> dict:
        emotion_logits, vad_outputs = self.session.run(None, {"input_values": input_values})

        logits = emotion_logits[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        emotion_pred_idx = int(np.argmax(probs))
        emotion_pred_label = self.idx_to_label[emotion_pred_idx]
        emotion_confidence = float(probs[emotion_pred_idx])

        valence = float(vad_outputs[0, 0])
        arousal = float(vad_outputs[0, 1])
        dominance = float(vad_outputs[0, 2])

        confidence = 0.5 * dominance + 0.3 * arousal + 0.2 * abs(valence - 0.5) * 2
        confidence = min(max(confidence, 0.0), 1.0)

        # Encoder classes are capitalised ("Happy"), prior keys are lowercase.
        emotion_prior = TRUST_EMOTION_PRIOR.get(emotion_pred_label.lower(), 0.5)
        trust = 0.4 * valence + 0.4 * dominance * (1 - arousal) + 0.2 * emotion_prior

        return {
            "emotion": emotion_pred_label,
            "emotion_confidence": round(emotion_confidence, 4),
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4),
            "trust": round(trust, 4),
            "confidence": round(confidence, 4),
        }
