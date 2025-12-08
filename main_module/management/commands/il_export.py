import json
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings

from main_module.models import ModerationItem


class Command(BaseCommand):
    help = "Export HUMAN-labeled items to JSONL for a given iteration."

    def add_arguments(self, parser):
        parser.add_argument("iter", type=int, help="Iteration number (e.g., 1)")
        parser.add_argument("--limit", type=int, default=4000)

    def handle(self, *args, **opts):
        k = int(opts["iter"])
        limit = int(opts["limit"]) * k

        out_dir = Path(settings.IL_DIR) / f"iter_{k:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "labels.jsonl"

        # simplest: newest HUMAN-labeled items up to limit
        qs = (ModerationItem.objects
              .filter(decision_source="HUMAN", final_action__in=["ALLOW", "BLOCK"])
              .order_by("-updated_at", "-id")[:limit])

        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for item in qs:
                y = 1 if item.final_action == "BLOCK" else 0
                f.write(json.dumps({"text": item.text, "label": y}, ensure_ascii=False) + "\n")
                n += 1

        self.stdout.write(self.style.SUCCESS(f"[iter {k}] Exported {n} -> {out_path}"))

# Usage: python manage.py il_export 1
# Exports up to 4000 HUMAN-labeled items with final_action ALLOW or BLOCK
# to the specified JSONL file for IL training.