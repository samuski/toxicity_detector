from pathlib import Path
from django.core.management.base import BaseCommand, CommandError

from main_module.il.ingest import ingest_csv


class Command(BaseCommand):
    help = "Ingest a CSV into ModerationItem rows (computes SL scores and routes certain/uncertain)."

    def add_arguments(self, parser):
        parser.add_argument("csv_path", type=str, help="Path to CSV (inside container). Must include a 'text' column.")
        parser.add_argument("--source", default="augment", help="Source tag stored on ModerationItem.source")
        parser.add_argument("--label-col", default="label_score", help="Optional label column name (or empty to ignore)")
        parser.add_argument("--low", type=float, default=None, help="If set with --high, score<=low becomes CERTAIN/ALLOW")
        parser.add_argument("--high", type=float, default=None, help="If set with --low, score>=high becomes CERTAIN/BLOCK")

    def handle(self, *args, **opts):
        csv_path = opts["csv_path"]
        p = Path(csv_path)
        if not p.exists():
            raise CommandError(f"CSV not found: {csv_path}")

        label_col = opts["label_col"] or None
        low = opts["low"]
        high = opts["high"]

        if (low is None) ^ (high is None):
            raise CommandError("Provide both --low and --high, or neither.")

        n = ingest_csv(
            csv_path,
            source=opts["source"],
            label_col=label_col,
            low_threshold=low,
            high_threshold=high,
        )
        self.stdout.write(self.style.SUCCESS(f"Ingested {n} rows from {csv_path}"))
