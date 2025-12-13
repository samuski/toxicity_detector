import json
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings

from main_module.models import ModerationItem


class Command(BaseCommand):
    help = "Export HUMAN-labeled items to JSONL for a given iteration."

    def add_arguments(self, parser):
        parser.add_argument("iter", type=int, help="Iteration number (e.g., 1)")
        # default=None so we can detect “not passed”
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Total number of HUMAN-labeled items to export. "
                 "If omitted, defaults to 750 * iter.",
        )

    def handle(self, *args, **opts):
        k = int(opts["iter"])

        # If user gave --limit, use it as-is.
        # Otherwise default to 750 * k (iteration-scaled).
        if opts["limit"] is None:
            limit = 750 * k
        else:
            limit = int(opts["limit"])

        out_dir = Path(settings.IL_DIR) / f"iter_{k:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "labels.jsonl"

        qs = (
            ModerationItem.objects
            .filter(decision_source="HUMAN", final_action__in=["ALLOW", "BLOCK"])
            .order_by("-updated_at", "-id")[:limit]
        )

        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for item in qs:
                y = 1 if item.final_action == "BLOCK" else 0
                f.write(
                    json.dumps(
                        {"text": item.text, "label": y},
                        ensure_ascii=False,
                    ) + "\n"
                )
                n += 1

        self.stdout.write(
            self.style.SUCCESS(f"[iter {k}] Exported {n} -> {out_path}")
        )
