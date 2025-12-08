# main_module/eval/train_sl.py
import os, argparse, numpy as np
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

from eval import accuracy_score, precision_recall_fscore_support
DEFAULT_TRAIN_CSV = "data/train.csv"
DEFAULT_VAL_CSV   = "data/val.csv"
DEFAULT_MODEL     = "roberta-base"
DEFAULT_EPOCHS    = 3
DEFAULT_BSZ       = 8
DEFAULT_MAX_LEN   = 256

# LABEL = "label_score"
LABEL_BIN = "label_bin" # for already-binary labels


def _clean_df(df: pd.DataFrame, name: str, label_col: str) -> pd.DataFrame:
    if "text" not in df.columns or label_col not in df.columns:
        raise ValueError(f"{name}.csv must have columns: text,{label_col}")

    df = df.copy()
    df["text"] = df["text"].astype("string").fillna("")
    df["text"] = (
        df["text"].str.normalize("NFKC")
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip()
    )

    y = pd.to_numeric(df[label_col], errors="coerce")
    bad = y.isna()
    if bad.any():
        df = df.loc[~bad].copy()
        y = y[~bad]
        print(f"[{name}] dropped {int(bad.sum())} rows with invalid/missing label")

    # force 0/1
    y = y.astype("int64")
    y = y.clip(0, 1)

    df[LABEL_BIN] = y

    empty = (df["text"].str.len() == 0)
    if empty.any():
        df = df.loc[~empty].copy()
        print(f"[{name}] dropped {int(empty.sum())} empty-text rows")

    return df.reset_index(drop=True)


def build_ds_from_csv(train_csv: str, val_csv: str, label_col: str, threshold: float, augment_csv: str | None) -> DatasetDict:
    read_opts = dict(low_memory=False, keep_default_na=False, dtype={label_col: "object"})

    df_tr = pd.read_csv(train_csv, **read_opts)
    df_va = pd.read_csv(val_csv,   **read_opts)

    df_tr = _clean_df(df_tr, "train", label_col)
    df_va = _clean_df(df_va, "val",   label_col)

    if augment_csv:
        df_aug = pd.read_csv(augment_csv, **read_opts)
        df_aug = _clean_df(df_aug, "augment", label_col)
        df_tr = pd.concat([df_tr, df_aug], ignore_index=True)
        print(f"[train] appended augment rows: +{len(df_aug)} (total train={len(df_tr)})")

    return DatasetDict({
        "train": Dataset.from_pandas(df_tr, preserve_index=False),
        "validation": Dataset.from_pandas(df_va, preserve_index=False),
    })


def build_ds_from_hf(name: str, config: str | None, label_col: str, threshold: float) -> DatasetDict:
    raw = load_dataset(name, config) if config else load_dataset(name)

    text_key = "text" if "text" in raw["train"].features else (
        "comment_text" if "comment_text" in raw["train"].features else None
    )
    if text_key is None or label_col not in raw["train"].features:
        raise ValueError(f"HF dataset must expose '{text_key}' and '{label_col}'.")

    def to_bin(example):
        y = float(example[label_col])
        example[LABEL_BIN] = 1 if y >= float(threshold) else 0
        return example

    def rename(split):
        ds = raw[split].rename_columns({text_key: "text"})
        ds = ds.map(to_bin)
        return ds

    val_split = "validation" if "validation" in raw else ("test" if "test" in raw else "train")
    return DatasetDict({"train": rename("train"), "validation": rename(val_split)})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits).squeeze(-1)
    labels = np.asarray(labels).astype(int)

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision_toxic": float(prec), "recall_toxic": float(rec), "f1_toxic": float(f1)}



class WeightedBCETrainer(Trainer):
    def __init__(self, *args, pos_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = float(pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)

        # sample weight: w = y*pos_weight + (1-y)*1
        w = labels * self.pos_weight + (1.0 - labels) * 1.0

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, weight=w, reduction="mean"
        )
        return (loss, outputs) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SL_MODEL_NAME", DEFAULT_MODEL))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("SL_EPOCHS", DEFAULT_EPOCHS)))
    ap.add_argument("--bsz", type=int, default=int(os.environ.get("SL_BSZ", DEFAULT_BSZ)))
    ap.add_argument("--max_len", type=int, default=int(os.environ.get("SL_MAX_LEN", DEFAULT_MAX_LEN)))

    ap.add_argument("--train_csv", default=os.environ.get("TRAIN_CSV"))
    ap.add_argument("--val_csv",   default=os.environ.get("VAL_CSV"))
    ap.add_argument("--augment_csv", default=os.environ.get("AUGMENT_CSV"))

    ap.add_argument("--hf_name",   default=os.environ.get("HF_NAME"))
    ap.add_argument("--hf_config", default=os.environ.get("HF_CONFIG"))

    ap.add_argument("--tag", default=os.environ.get("SL_TAG", "active"))

    ap.add_argument("--label_col", default=os.environ.get("LABEL_COL", "label_bin"))
    ap.add_argument("--label_threshold", type=float, default=float(os.environ.get("LABEL_TH", 0.5)))
    ap.add_argument("--pos_weight", default=os.environ.get("POS_WEIGHT", "auto"),
                    help="Use 'auto' or a float (e.g., 6.0).")

    args = ap.parse_args()

    art_dir = os.environ.get("ARTIFACT_DIR", "/artifacts")
    out_dir = os.path.join(art_dir, "sl", args.tag)
    tmp_dir = os.path.join(art_dir, "sl", "tmp")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv or DEFAULT_TRAIN_CSV
    val_csv   = args.val_csv   or DEFAULT_VAL_CSV

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        ds = build_ds_from_csv(train_csv, val_csv, args.label_col, args.label_threshold, args.augment_csv)
        print(f"Using local CSVs: {train_csv} / {val_csv}" + (f" + {args.augment_csv}" if args.augment_csv else ""))
    elif args.hf_name:
        ds = build_ds_from_hf(args.hf_name, args.hf_config, args.label_col, args.label_threshold)
        print(f"Using HF dataset: {args.hf_name} ({args.hf_config or 'default'})")
    else:
        raise SystemExit(
            "Provide CSVs (TRAIN_CSV/VAL_CSV) or HF_NAME.\n"
            f"Missing defaults:\n  {train_csv}\n  {val_csv}"
        )

    # Class balance -> pos_weight
    y_train = np.array(ds["train"][LABEL_BIN], dtype=np.int64)
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        raise SystemExit("No positive (toxic) samples in training data.")

    pos_weight = (neg / max(pos, 1)) if args.pos_weight == "auto" else float(args.pos_weight)
    print(f"[class balance] neg={neg} pos={pos} pos_weight={pos_weight:.4f}")

    tok = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    def tfm(batch):
        t = tok(batch["text"], truncation=True, max_length=args.max_len)
        t["labels"] = [int(x) for x in batch[LABEL_BIN]]
        return t

    ds = DatasetDict({
        "train": ds["train"].map(tfm, batched=True),
        "validation": ds["validation"].map(tfm, batched=True),
    })

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)

    targs = TrainingArguments(
        output_dir=tmp_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_total_limit=1,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=16,
        load_best_model_at_end=True,
        metric_for_best_model="f1_toxic",
        greater_is_better=True,
    )

    trainer = WeightedBCETrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
    )

    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved SL model to {out_dir}")


if __name__ == "__main__":
    main()
