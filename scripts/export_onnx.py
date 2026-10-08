"""One-time conversion of the TorchScript model to ONNX (no retraining: same weights, same graph).

Needs torch and onnx, which the server itself does not:
    pip install torch onnx onnxruntime --extra-index-url https://download.pytorch.org/whl/cpu
    python scripts/export_onnx.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hubert_heads_test_7_lora"
SRC = MODEL_DIR / "hubert_emotion_vad_30hrs_jit.pt"
DST = MODEL_DIR / "hubert_emotion_vad_30hrs.onnx"
SAMPLE_RATE = 16000


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = torch.jit.load(str(SRC), map_location="cpu").eval()
    example = torch.randn(1, SAMPLE_RATE * 8)

    # TorchScript modules go through the TorchScript-based exporter (dynamo=False).
    torch.onnx.export(
        model,
        (example,),
        str(DST),
        dynamo=False,
        input_names=["input_values"],
        output_names=["emotion_logits", "vad"],
        dynamic_axes={"input_values": {0: "batch", 1: "samples"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Wrote {DST.name} ({DST.stat().st_size / 1024**2:.0f} MB)")

    # Sanity check at a few lengths: ONNX Runtime must agree with TorchScript.
    session = ort.InferenceSession(str(DST), providers=["CPUExecutionProvider"])
    worst = 0.0
    for seconds in (1, 3.7, 8):
        x = torch.randn(1, int(SAMPLE_RATE * seconds))
        with torch.inference_mode():
            ref = [t.numpy() for t in model(x)]
        out = session.run(None, {"input_values": x.numpy()})
        diff = max(float(np.abs(r - o).max()) for r, o in zip(ref, out))
        worst = max(worst, diff)
        print(f"  {seconds:>4}s  max|Δ| = {diff:.2e}")
    if worst > 1e-3:
        sys.exit(f"Mismatch too large ({worst:.2e}); do not use this export.")
    print("OK")


if __name__ == "__main__":
    main()
