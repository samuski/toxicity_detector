#!/usr/bin/env python3
import argparse, os, glob, time
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ---------------- Defaults ----------------
MODEL = "roberta-base"
MODEL_DIR = os.path.join(os.environ.get("ARTIFACT_DIR", "/artifacts"), "sl", "active")
DATA_CSV = os.getenv("DATA_CSV", "/app/data/test.csv")
OUT_DIR  = os.getenv("EVAL_OUT", "/artifacts/eval")
TEXT_COL = os.getenv("TEXT_COL", "text")
LABEL_COL = os.getenv("LABEL_COL", "label_score")

# This is ONLY for binarizing label_score -> y_true, and for the manual eval row.
LABEL_THRESHOLD_DEFAULT = float(os.getenv("LABEL_THRESHOLD", "0.5"))


# ---------------- Data ----------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


def collate_fn(batch, tokenizer, max_length: int):
    texts = [b["text"] for b in batch]
    labels = [b.get("label", None) for b in batch]
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    # we don't actually use labels in the forward pass, so this is optional
    if labels[0] is not None:
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc


# ---------------- Inference ----------------
@torch.no_grad()
def infer_logits(
    model, dataloader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    logits_list = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ("input_ids", "attention_mask")}
        out = model(**batch)
        logits = out.logits.detach().cpu().numpy()
        logits_list.append(logits)
    return np.concatenate(logits_list, axis=0)


def softmax(z, axis=1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------- Metrics ----------------
def compute_binary_metrics(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    roc = np.nan
    prauc = np.nan
    if len(np.unique(y_true)) == 2:
        try:
            roc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
        try:
            prauc = average_precision_score(y_true, y_prob) 
        except Exception:
            pass

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f_macro,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f_w,
        "roc_auc": roc,
        "pr_auc": prauc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }, cm


def best_threshold_by_f1(y_true, y_prob, grid: np.ndarray):
    best = None
    best_cm = None
    for t in grid:
        m, cm = compute_binary_metrics(y_true, y_prob, float(t))
        if (best is None) or (m["f1_macro"] > best["f1_macro"]):
            best = m
            best_cm = cm
    return best, best_cm


# ---------------- Plotting ----------------
def save_confusion_matrix_png(cm: np.ndarray, labels: List[str], outpath: Path, title: str):
    fig = plt.figure(figsize=(4.5, 4.5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    # annotate
    vmax = cm.max()
    thresh = vmax / 2.0 if vmax > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------- Core eval ----------------
def evaluate_one_weight(
    model_name_or_path: str,
    weight_path: str,
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    outdir: Path,
    threshold_grid: np.ndarray,
    id_col: Optional[str] = None,
    label_threshold: float = LABEL_THRESHOLD_DEFAULT,
):
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 1) Ground truth: float -> binary using label_threshold ---
    df = df.copy()
    scores = pd.to_numeric(df[label_col], errors="coerce")
    bad = scores.isna()
    if bad.any():
        dropped = int(bad.sum())
        df = df.loc[~bad]
        scores = scores[~bad]
        print(f"[eval] dropped {dropped} rows with invalid {label_col}")

    y_true = (scores >= label_threshold).astype(int).values
    texts = df[text_col].astype(str).tolist()
    ids = df[id_col].tolist() if id_col and id_col in df.columns else list(range(len(df)))

    # --- 2) Load tokenizer & model ---
    weight_path = Path(weight_path)
    if weight_path.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(weight_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(weight_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
        )
        state = torch.load(str(weight_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[WARN] Unexpected keys in state_dict: {unexpected}")
        if missing:
            print(f"[WARN] Missing keys in state_dict: {missing}")

    model.to(device)

    # --- 3) Inference ---
    ds = TextDataset(texts, y_true)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    logits = infer_logits(model, dl, device=device)
    if logits.shape[1] == 1:
        # binary logits -> probability
        z = logits.reshape(-1)
        y_prob = 1.0 / (1.0 + np.exp(-z))   # sigmoid
    else:
        y_prob = softmax(logits, axis=1)[:, 1]

    # --- 4) Metrics: manual threshold vs auto best-F1 ---
    # manual: use label_threshold as the policy threshold
    manual_metrics, manual_cm = compute_binary_metrics(y_true, y_prob, label_threshold)

    # auto: search over threshold_grid
    best_metrics, best_cm = best_threshold_by_f1(y_true, y_prob, threshold_grid)

    # --- 5) Save per-weight outputs ---
    tag = weight_path.stem
    preds_csv = outdir / f"{tag}__preds.csv"
    pd.DataFrame(
        {
            "id": ids,
            "text": texts,
            "label_true": y_true,
            "prob_toxic": y_prob,
            f"pred@{label_threshold:.2f}": (y_prob >= label_threshold).astype(int),
            f"pred@best({best_metrics['threshold']:.3f})": (
                y_prob >= best_metrics["threshold"]
            ).astype(int),
        }
    ).to_csv(preds_csv, index=False)

    # confusion matrices
    manual_cm_csv = outdir / f"{tag}__cm_t{label_threshold:.2f}.csv"
    best_cm_csv = outdir / f"{tag}__cm_t{best_metrics['threshold']:.3f}.csv"
    pd.DataFrame(manual_cm, index=[0, 1], columns=[0, 1]).to_csv(manual_cm_csv)
    pd.DataFrame(best_cm, index=[0, 1], columns=[0, 1]).to_csv(best_cm_csv)

    # PNG plots
    save_confusion_matrix_png(
        manual_cm,
        ["clean(0)", "toxic(1)"],
        outdir / f"{tag}__cm_t{label_threshold:.2f}.png",
        f"Confusion Matrix @{label_threshold:.2f} — {tag}",
    )
    save_confusion_matrix_png(
        best_cm,
        ["clean(0)", "toxic(1)"],
        outdir / f"{tag}__cm_t{best_metrics['threshold']:.3f}.png",
        f"Confusion Matrix @{best_metrics['threshold']:.3f} — {tag}",
    )

    # rows for summary
    row_manual = {"weight": tag, "mode": f"t={label_threshold:.2f}", **manual_metrics}
    row_best = {"weight": tag, "mode": "t=bestF1", **best_metrics}
    return row_manual, row_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL,
                    help="HF model id or local dir (e.g., roberta-base or ./roberta)")
    ap.add_argument("--weights", default=MODEL_DIR,
                    help="Path or glob to weight files (e.g., 'weights/*.bin' or model dir)")
    ap.add_argument("--data_csv", default=DATA_CSV,
                    help="CSV file with at least columns: text,label_score")
    ap.add_argument("--text_col", default=TEXT_COL)
    ap.add_argument("--label_col", default=LABEL_COL)
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--outdir", default=OUT_DIR)
    ap.add_argument(
        "--thresholds",
        default="0.05:0.95:0.01",
        help="start:stop:step for prediction threshold sweep "
             "(inclusive start, exclusive stop)",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--label_threshold",
        type=float,
        default=LABEL_THRESHOLD_DEFAULT,
        help="Threshold on label_score to define ground-truth toxicity.",
    )

    args = ap.parse_args()

    # threshold grid for predictions
    s, e, st = map(float, args.thresholds.split(":"))
    grid = np.arange(s, e, st)
    if e == 1.0 and grid[-1] < 0.999:
        grid = np.append(grid, 0.999)

    # device
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    df = pd.read_csv(args.data_csv)
    assert args.text_col in df.columns and args.label_col in df.columns, \
        "CSV must have text & label columns."

    # gather weights
    if any(ch in args.weights for ch in ["*", "?", "["]):
        weight_paths = sorted(glob.glob(args.weights))
    else:
        weight_paths = [args.weights]
    if not weight_paths:
        raise FileNotFoundError(f"No weight files match: {args.weights}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for w in weight_paths:
        print(f"\n=== Evaluating: {w} ===")
        t0 = time.time()
        row_manual, row_best = evaluate_one_weight(
            model_name_or_path=args.model,
            weight_path=w,
            df=df,
            label_col=args.label_col,
            text_col=args.text_col,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=dev,
            outdir=outdir,
            threshold_grid=grid,
            id_col=args.id_col,
            label_threshold=args.label_threshold,
        )
        dt = time.time() - t0
        row_manual["seconds"] = dt
        row_best["seconds"] = dt
        summary_rows.extend([row_manual, row_best])

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(by=["mode", "f1_macro"], ascending=[True, False])
    summary_csv = outdir / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nWrote summary to: {summary_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
