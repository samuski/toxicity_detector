# main_module/eval/train_sl.py
import os, argparse, numpy as np
import torch
from pathlib import Path
from datasets import (
    load_dataset, 
    Dataset, 
    DatasetDict,
    Value
)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)

from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score,
    mean_squared_error
)
import pandas as pd

DEFAULT_TRAIN_CSV = "data/train.csv"
DEFAULT_VAL_CSV   = "data/val.csv"
DEFAULT_MODEL     = "roberta-base"
DEFAULT_EPOCHS    = 3
DEFAULT_BSZ       = 8
DEFAULT_MAX_LEN   = 256 # Actual max is around 300 tokens but these are very few and usually toxic token appears before this limit.
LABEL = "label_score"

def build_ds_from_csv(train_csv: str, val_csv: str):
    read_opts = dict(
        low_memory=False,          # no chunked guessing
        keep_default_na=False,     # keep literal "NA" as text
        dtype={LABEL: "object"},    # read label as object then force type later
    )
    df_tr = pd.read_csv(train_csv, **read_opts)
    df_va = pd.read_csv(val_csv,   **read_opts)

    # Ensure required columns exist
    for df, name in [(df_tr, "train"), (df_va, "val")]:
        if "text" not in df or LABEL not in df:
            raise ValueError(f"{name}.csv must have columns: text,label_bin")

        # Force string text; fill blanks
        df["text"] = df["text"].astype("string").fillna("")

        df["text"] = (
            df["text"].str.normalize("NFKC")
                      .str.replace(r"\s+", " ", regex=True)
                      .str.strip()
        )

        # Coerce label to 0/1
        df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce").astype("float32")
        bad = df[LABEL].isna()
        if bad.any():
            dropped = int(bad.sum())
            df.dropna(subset=[LABEL], inplace=True)
            print(f"[{name}] dropped {dropped} rows with invalid/missing label")

        df[LABEL] = df[LABEL].astype("float32")
        empty = (df["text"].str.len() == 0)
        if empty.any():
            df.drop(index=df.index[empty], inplace=True)
            print(f"[{name}] dropped {int(empty.sum())} empty-text rows")

    from datasets import Dataset, DatasetDict
    return DatasetDict({
        "train": Dataset.from_pandas(df_tr.reset_index(drop=True), preserve_index=False),
        "validation": Dataset.from_pandas(df_va.reset_index(drop=True), preserve_index=False),
    })

def build_ds_from_hf(name: str, config: str|None) -> DatasetDict:
    raw = load_dataset(name, config) if config else load_dataset(name)
    text_key = "text" if "text" in raw["train"].features else (
        "comment_text" if "comment_text" in raw["train"].features else None
    )
    if text_key is None or LABEL not in raw["train"].features:
        raise ValueError("Dataset must expose 'text' (or 'comment_text') and binary 'label'.")
    def rename(split):
        return raw[split].rename_columns({text_key: "text"})
    val_split = "validation" if "validation" in raw else "test"
    return DatasetDict({"train": rename("train"), "validation": rename(val_split)})

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.flatten() 
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    return {
        "mse": mse,
        "rmse": rmse
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SL_MODEL_NAME", DEFAULT_MODEL))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("SL_EPOCHS", DEFAULT_EPOCHS)))
    ap.add_argument("--bsz", type=int, default=int(os.environ.get("SL_BSZ", DEFAULT_BSZ)))
    ap.add_argument("--max_len", type=int, default=int(os.environ.get("SL_MAX_LEN", DEFAULT_MAX_LEN)))
    ap.add_argument("--train_csv", default=os.environ.get("TRAIN_CSV"))
    ap.add_argument("--val_csv",   default=os.environ.get("VAL_CSV"))
    ap.add_argument("--hf_name",   default=os.environ.get("HF_NAME"))
    ap.add_argument("--hf_config", default=os.environ.get("HF_CONFIG"))
    ap.add_argument(
        "--tag",
        default=os.environ.get("SL_TAG", "active"),
        help="Subfolder under ARTIFACT_DIR/sl/ to save model (e.g. baseline, oracle, active).",
    )
    args = ap.parse_args()

    art_dir = os.environ.get("ARTIFACT_DIR", "/artifacts")
    out_dir = os.path.join(art_dir, "sl", args.tag)
    tmp_dir = os.path.join(art_dir, "sl", "tmp")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv or DEFAULT_TRAIN_CSV
    val_csv   = args.val_csv   or DEFAULT_VAL_CSV

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        ds = build_ds_from_csv(train_csv, val_csv)
        print(f"Using local CSVs: {train_csv} / {val_csv}")
    elif args.hf_name:
        ds = build_ds_from_hf(args.hf_name, args.hf_config)
        print(f"Using HF dataset: {args.hf_name} ({args.hf_config or 'default'})")
    else:
        raise SystemExit(
            "Provide CSVs (create data/train.csv, data/val.csv) or set HF_NAME. "
            "Current defaults not found:\n"
            f"  {train_csv}\n  {val_csv}"
        )

    tok = AutoTokenizer.from_pretrained(args.model) # tokenizer is BPE-based
    data_collator = DataCollatorWithPadding(tokenizer=tok)  # pads per-batch to max length

    def tfm(batch):
        t = tok(batch["text"], truncation=True, max_length=args.max_len)
        t["labels"] = batch[LABEL]
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
        metric_for_best_model="mse",
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model, 
        args=targs, 
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved SL model to {out_dir}")

if __name__ == "__main__":
    main()
