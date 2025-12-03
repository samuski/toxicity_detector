from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings

from main_module.il.il_train import train_il


class Command(BaseCommand):
    help = "Train IL model for a given iteration from exported labels.jsonl."

    def add_arguments(self, parser):
        parser.add_argument("iter", type=int, help="Iteration number (e.g., 1)")

    def handle(self, *args, **opts):
        k = int(opts["iter"])
        out_dir = Path(settings.IL_DIR) / f"iter_{k:03d}"
        labels_path = out_dir / "labels.jsonl"
        artifacts_dir = out_dir / "artifacts"

        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels: {labels_path}. Run: python manage.py il_export {k}")

        train_il(str(labels_path), out_dir=str(artifacts_dir))
        self.stdout.write(self.style.SUCCESS(f"[iter {k}] Trained -> {artifacts_dir}"))

# Usage: python manage.py il_train 1
# Trains IL model from il/iter_001/labels.jsonl and saves artifacts to il/iter_001/artifacts