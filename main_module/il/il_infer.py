# main_module/il/il_infer.py
import os, threading, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_lock = threading.Lock()

MODEL_DIR = os.environ.get("IL_MODEL_DIR", "/artifacts/il/active")

def set_model_dir(path: str):
    """Switch IL checkpoint directory at runtime (and clear cache)."""
    global MODEL_DIR, _model, _tokenizer
    MODEL_DIR = path
    _model = None
    _tokenizer = None
    print(f"[IL] Using model dir: {MODEL_DIR}")

def _load():
    global _model, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                if not os.path.isdir(MODEL_DIR):
                    raise RuntimeError(f"[IL] model dir missing: {MODEL_DIR}")
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
                _model.to(_device).eval()

def il_score_text(text: str, max_len: int = 256) -> float:
    _load()
    with torch.no_grad():
        t = _tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(_device)
        logits = _model(**t).logits
        z = logits[0, 0]
        return float(torch.sigmoid(z).item())
