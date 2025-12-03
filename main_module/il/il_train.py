import json
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_il(jsonl_path: str, out_dir: str = "il_artifacts"):
    p = Path(jsonl_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    texts, y = [], []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            y.append(int(r["label"]))

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)

    joblib.dump(vec, out / "vectorizer.joblib")
    joblib.dump(clf, out / "model.joblib")
    return str(out)
