# backend/app/moderation/sl_infer.py
import os, threading, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_lock = threading.Lock()

def reload_model():
    """Hot-reload the currently active model from disk."""
    global _model, _tokenizer
    with _lock:
        _model = None
        _tokenizer = None
    _load()
    return True

MODEL_DIR = os.path.join(os.environ.get("ARTIFACT_DIR", "/artifacts"), "sl", "active")

def _load():
    """Load model+tokenizer once, on first use (safe under Gunicorn workers)."""
    global _model, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
                _model.to(_device).eval()

def score_text(text: str, max_len: int = 256) -> float:
    """
    Return p(toxic) in [0, 1].

    - If the model has a single logit (num_labels=1 / BCE), use sigmoid.
    - If the model has two logits (num_labels=2), use softmax and take class 1.
    """
    _load()
    with torch.no_grad():
        t = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(_device)

        logits = _model(**t).logits  # shape [batch, C]

        if logits.dim() == 1:
            # rare, but just in case
            logits = logits.unsqueeze(0)

        if logits.size(-1) == 1:
            # Single-logit (BCE-with-logits on soft labels)
            logit = logits.squeeze(-1)[0]          # scalar
            p1 = torch.sigmoid(logit).item()
        else:
            # Two-class (old setup)
            logit_vec = logits[0]                  # [2]
            p1 = torch.softmax(logit_vec, dim=-1)[1].item()

        return float(p1)

def estimate_uncertainty(p: float) -> float:
    """Quick uncertainty proxy (use MC-Dropout later if needed)."""
    return p * (1 - p)

if __name__ == "__main__":
    os.environ.setdefault("ARTIFACT_DIR", "/artifacts")
    reload_model()
    print("Interactive toxicity scoring. Type text and press Enter. Ctrl-D/Ctrl-Z to quit.")
    while True:
        try:
            s = input("> ")
        except EOFError:
            break
        if not s.strip():
            continue
        p = score_text(s)
        u = estimate_uncertainty(p)
        print(f"p(toxic)={p:.4f}  uncertainty={u:.4f}")
