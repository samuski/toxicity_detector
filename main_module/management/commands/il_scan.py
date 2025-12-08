# main_module/management/commands/il_scan.py
from pathlib import Path

from django.core.management.base import BaseCommand

from main_module.models import ModerationItem
import main_module.il.il_infer as il_infer  # import module, not function


class Command(BaseCommand):
    help = "Scan DB rows with SL decisions and flag strong SL↔IL disagreements for human review."

    def add_arguments(self, parser):
        parser.add_argument("--source", default=None)
        parser.add_argument("--sl_low", type=float, default=0.2)
        parser.add_argument("--sl_high", type=float, default=0.8)
        parser.add_argument("--il_low", type=float, default=0.2)
        parser.add_argument("--il_high", type=float, default=0.8)
        parser.add_argument("--limit", type=int, default=0)

        # Choose which IL iteration/artifacts to use
        parser.add_argument("--il_iter", type=int, default=1)
        parser.add_argument(
            "--il_dir",
            default="",
            help="Explicit IL artifacts dir (overrides --il_iter).",
        )

        # Optional: use a different decision threshold for IL suggested action
        parser.add_argument("--il_decision_th", type=float, default=0.5)

    def handle(self, *args, **opts):
        source = opts["source"]
        sl_low, sl_high = float(opts["sl_low"]), float(opts["sl_high"])
        il_low, il_high = float(opts["il_low"]), float(opts["il_high"])
        limit = int(opts["limit"]) if opts["limit"] else 0
        il_iter = int(opts["il_iter"])
        il_dir = (opts["il_dir"] or "").strip()
        il_decision_th = float(opts["il_decision_th"])

        # --- Pick IL model dir (joblib artifacts) ---
        chosen = None
        if il_dir:
            chosen = Path(il_dir)
        elif il_iter > 0:
            chosen = Path(f"/app/media/il/iter_{il_iter:03d}/artifacts")

        if chosen is not None:
            if not chosen.is_dir():
                raise SystemExit(f"IL model dir not found: {chosen}")
            il_infer.set_model_dir(str(chosen))
            self.stdout.write(self.style.WARNING(f"[IL] Using model dir: {chosen}"))

        # --- Query: scan SL-certain items decided by SL ---
        qs = ModerationItem.objects.filter(
            decision_source="SL",
            status=ModerationItem.Status.CERTAIN,
        )
        if source:
            qs = qs.filter(source=source)

        qs = qs.order_by("created_at", "id")
        if limit and limit > 0:
            qs = qs[:limit]

        flagged = 0
        scanned = 0

        for item in qs.iterator(chunk_size=500):
            scanned += 1

            # IL probability
            p_il = float(il_infer.il_score_text(item.text))
            il_action = "BLOCK" if p_il >= il_decision_th else "ALLOW"

            # Store IL info (do NOT overwrite SL decision)
            item.il_score = p_il
            item.il_suggested_action = il_action

            # SL action derived from what SL already set
            sl_action = item.final_action  # "ALLOW" or "BLOCK"

            # "strong disagreement" only when both are confident
            sl_conf_allow = (
                item.sl_score is not None and item.sl_score <= sl_low and sl_action == "ALLOW"
            )
            sl_conf_block = (
                item.sl_score is not None and item.sl_score >= sl_high and sl_action == "BLOCK"
            )
            il_conf_allow = (p_il <= il_low)
            il_conf_block = (p_il >= il_high)

            disagree = (sl_conf_allow and il_conf_block) or (sl_conf_block and il_conf_allow)

            # IMPORTANT: clear old flags when no longer disagreeing
            item.needs_review = bool(disagree)
            if disagree:
                flagged += 1

            item.save(update_fields=["il_score", "il_suggested_action", "needs_review"])

        self.stdout.write(self.style.SUCCESS(
            f"Scanned {scanned} rows. Flagged {flagged} for human review."
        ))
