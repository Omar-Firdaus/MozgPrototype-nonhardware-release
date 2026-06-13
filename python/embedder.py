"""Face embeddings from arcface.onnx (CPU)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("pip install onnxruntime") from e

_DEFAULT_ONNX = Path(__file__).resolve().parent / "models" / "arcface.onnx"

_sess: Optional[ort.InferenceSession] = None
_in_name: Optional[str] = None
_out_name: Optional[str] = None
_input_nhwc: bool = False


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x) + 1e-12)
    return (x / n).astype(np.float32)


def _input_is_nhwc(shape) -> bool:
    if not shape or len(shape) < 4:
        return False
    s = list(shape)
    if isinstance(s[1], int) and s[1] == 3:
        return False
    if isinstance(s[3], int) and s[3] == 3:
        return True
    return False


def get_session() -> Tuple[ort.InferenceSession, str, str]:
    global _sess, _in_name, _out_name, _input_nhwc
    if _sess is not None and _in_name and _out_name:
        return _sess, _in_name, _out_name

    path = Path(
        os.environ.get(
            "FACE_EMBEDDER_ONNX",
            os.environ.get("MOBILEFACENET_ONNX", str(_DEFAULT_ONNX)),
        )
    ).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Face model not found: {path}\n"
            "Drop arcface.onnx in python/models/ or set FACE_EMBEDDER_ONNX."
        )
    _sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _in_name = _sess.get_inputs()[0].name
    _out_name = _sess.get_outputs()[0].name
    _input_nhwc = _input_is_nhwc(_sess.get_inputs()[0].shape)
    return _sess, _in_name, _out_name


def _prep(face_bgr: np.ndarray, nhwc: bool) -> np.ndarray:
    if face_bgr.size == 0:
        raise ValueError("empty face crop")
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) - 127.5) / 128.0
    if nhwc:
        return np.expand_dims(x, axis=0)
    return np.expand_dims(np.transpose(x, (2, 0, 1)), axis=0)


def get_embedding(face_image: np.ndarray) -> np.ndarray:
    sess, in_name, out_name = get_session()
    out = sess.run([out_name], {in_name: _prep(face_image, _input_nhwc)})[0]
    return _l2_normalize(np.asarray(out, dtype=np.float32).reshape(-1))
