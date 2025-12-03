# your_app/il/ingest.py

import csv
from pathlib import Path
from django.db import transaction

from ..models import ModerationItem
from main_module.sl.sl_infer import score_text, estimate_uncertainty


@transaction.atomic
def ingest_csv(
    path: str,
    source: str = "jigsaw",
    label_col: str | None = "label_score",
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> int:
    """
    Ingest a CSV of new items and store them as ModerationItem rows.

    - score_text(text) → toxicity score in [0,1]
    - estimate_uncertainty(score) → scalar uncertainty heuristic
    - If low/high thresholds are provided:
        score <= low_threshold  -> CERTAIN + ALLOW
        score >= high_threshold -> CERTAIN + BLOCK
        else                    -> UNCERTAIN
    - If thresholds are None, everything is UNCERTAIN.
    Returns the number of rows ingested.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    created = 0

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = row.get("text", "")
            if not text:
                continue

            # SL model output
            score = float(score_text(text))
            unc = float(estimate_uncertainty(score))

            # Optional ground-truth label if it's a labeled dataset
            true_label = None
            if label_col and label_col in row and row[label_col] not in ("", None):
                try:
                    true_label = float(row[label_col])
                except ValueError:
                    true_label = None

            status = ModerationItem.Status.UNCERTAIN
            final_action = ModerationItem.FinalAction.NONE
            decision_source = None

            if low_threshold is not None and high_threshold is not None:
                # Simple auto-routing based on score
                if score <= low_threshold:
                    status = ModerationItem.Status.CERTAIN
                    final_action = ModerationItem.FinalAction.ALLOW
                    decision_source = "SL"
                elif score >= high_threshold:
                    status = ModerationItem.Status.CERTAIN
                    final_action = ModerationItem.FinalAction.BLOCK
                    decision_source = "SL"
                else:
                    status = ModerationItem.Status.UNCERTAIN
                    final_action = ModerationItem.FinalAction.NONE
                    decision_source = None

            ModerationItem.objects.create(
                external_id=row.get("id") or row.get("comment_id"),
                source=source,
                text=text,
                sl_score=score,
                sl_uncertainty=unc,
                true_label=true_label,
                status=status,
                final_action=final_action,
                decision_source=decision_source,
            )
            created += 1

    return created
