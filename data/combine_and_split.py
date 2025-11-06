import pandas as pd, numpy as np, hashlib
import chardet, pathlib

# def shastr(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()


def read_with_detect(path):
    raw = pathlib.Path(path).read_bytes()
    enc = chardet.detect(raw)["encoding"] or "cp1252"
    return pd.read_csv(path, encoding=enc)

j = read_with_detect("jigsaw.csv")
h = read_with_detect("hatexplain.csv")

J = pd.DataFrame({
    "original_id": j["id"].astype(str),
    "source": "jigsaw",
    "text": j["comment_text"],
    "label_bin": j["is_toxic"].astype(int),
    "label_score": j["toxicity"].astype(float),
    "ann1": np.nan, "ann2": np.nan, "ann3": np.nan,
    "policy": "given"
})

H = pd.DataFrame({
    "original_id": h["id"].astype(str),
    "source": "hatexplain",
    "text": h["comment_text"],
    "label_bin": h["is_toxic"].astype(int),
    "label_score": np.nan,
    "ann1": h["annotator1"], "ann2": h["annotator2"], "ann3": h["annotator3"],
    "policy": "given"
})

df = pd.concat([J, H], ignore_index=True)
# df["text_sha256"] = df["text"].map(shastr)


import re, unicodedata

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 1) Strongly coerce to pandas' nullable string and fill NaN
df["text"] = df["text"].astype("string").fillna("")

# 2) Normalize (optional but good for stable hashing & dedup)
df["text_norm"] = df["text"].apply(normalize_text)

# 3) Drop truly empty rows (rare but safer)
empty_mask = df["text_norm"].str.len() == 0
if empty_mask.any():
    print("Dropping empty text rows:", int(empty_mask.sum()))
    df = df.loc[~empty_mask].copy()

# 4) Hash on the normalized text
df["text_sha256"] = df["text_norm"].apply(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())

df.to_csv("merged_data.csv", index=False, encoding="utf-8")
