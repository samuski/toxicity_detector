# main_module/il/il_infer.py
import os, threading
from pathlib import Path

import numpy as np
import joblib

_lock = threading.Lock()
_model = None
_vectorizer = None
_MODEL_DIR = None

def set_model_dir(path: str):
    """Call this from il_scan to choose iter_XXX explicitly."""
    global _MODEL_DIR, _model, _vectorizer
    _MODEL_DIR = str(path)
    _model = None
    _vectorizer = None

def _resolve_dir() -> Path:
    # default: env var or iter_001
    base = Path(os.environ.get("IL_MODEL_DIR", "/app/media/il/iter_001/artifacts"))
    if _MODEL_DIR:
        base = Path(_MODEL_DIR)
    return base

def _load():
    global _model, _vectorizer
    if _model is not None and _vectorizer is not None:
        return

    with _lock:
        if _model is not None and _vectorizer is not None:
            return

        d = _resolve_dir()
        m = d / "model.joblib"
        v = d / "vectorizer.joblib"

        if not (m.exists() and v.exists()):
            raise RuntimeError(f"IL artifacts not found (need model.joblib + vectorizer.joblib): {d}")

        _model = joblib.load(m)
        _vectorizer = joblib.load(v)
        print(f"[IL] Loaded sklearn IL model from {d}")

def il_score_text(text: str) -> float:
    """Return p(toxic) in [0,1]."""
    _load()
    X = _vectorizer.transform([text])

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)[0]
        # assume class 1 = toxic; if only one column, take last
        return float(proba[1] if len(proba) > 1 else proba[-1])

    if hasattr(_model, "decision_function"):
        z = float(_model.decision_function(X)[0])
        return float(1.0 / (1.0 + np.exp(-z)))  # sigmoid

    # fallback: hard prediction
    y = int(_model.predict(X)[0])
    return float(y)
