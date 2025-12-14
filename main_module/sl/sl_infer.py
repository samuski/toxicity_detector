# backend/app/moderation/sl_infer.py
import os, threading, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
import numpy as np
import torch

_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_lock = threading.Lock()

MODEL_DIR = os.environ.get("SL_MODEL_DIR", "/artifacts/sl/baseline")

def reload_model():
    global _tok, _model
    if not os.path.isdir(MODEL_DIR):
        print(f"[SL] model dir missing: {MODEL_DIR} (SL disabled until trained)")
        _tok, _model = None, None
        return

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
    _load()
    if _model is None:
        raise RuntimeError("SL model not available yet. Train it or mount /artifacts/sl/active.")

    _model.eval()
    with torch.no_grad():
        t = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(_device)

        logits = _model(**t).logits  # [1, N]

        if logits.shape[-1] == 1:
            z = logits[0, 0]
            p = torch.sigmoid(z)
            return float(p.item())
        else:
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0, 1].item())

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

@lru_cache(maxsize=8)
def _load_sl_dir(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()
    return tok, mdl

@torch.no_grad()
def score_text_with_dir(model_dir: str, text: str, max_len: int = 256) -> float:
    tok, mdl = _load_sl_dir(model_dir)
    enc = tok([text], truncation=True, max_length=max_len, return_tensors="pt")
    out = mdl(**enc)
    z = out.logits.view(-1).cpu().numpy()[0]
    p = 1.0 / (1.0 + np.exp(-z))
    return float(p)