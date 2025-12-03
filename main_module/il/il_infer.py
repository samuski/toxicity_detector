from pathlib import Path
from django.conf import settings
import joblib

_VEC = None
_MODEL = None
_LOADED_ITER = None

def _latest_iter_dir() -> Path | None:
    base = Path(settings.IL_DIR)
    if not base.exists():
        return None
    iters = sorted([p for p in base.glob("iter_*") if p.is_dir()])
    return iters[-1] if iters else None

def reload_il(iter_num: int | None = None):
    global _VEC, _MODEL, _LOADED_ITER
    if iter_num is None:
        d = _latest_iter_dir()
        if d is None:
            raise FileNotFoundError("No IL iterations found under media/il/")
        iter_num = int(d.name.split("_")[1])
        iter_dir = d
    else:
        iter_dir = Path(settings.IL_DIR) / f"iter_{iter_num:03d}"

    art = iter_dir / "artifacts"
    _VEC = joblib.load(art / "vectorizer.joblib")
    _MODEL = joblib.load(art / "model.joblib")
    _LOADED_ITER = iter_num

def il_score_text(text: str) -> float:
    if _VEC is None or _MODEL is None:
        reload_il()  # loads latest by default
    X = _VEC.transform([text])
    return float(_MODEL.predict_proba(X)[0, 1])

def il_loaded_iter() -> int | None:
    return _LOADED_ITER
