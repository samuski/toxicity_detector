from django.core.management.base import BaseCommand
from django.db import transaction

from main_module.models import ModerationItem
from main_module.il.il_infer import il_score_text

class Command(BaseCommand):
    help = "Scan DB rows with SL decisions and flag strong SL↔IL disagreements for human review."

    def add_arguments(self, parser):
        parser.add_argument("--source", default=None)
        parser.add_argument("--sl_low", type=float, default=0.2)
        parser.add_argument("--sl_high", type=float, default=0.8)
        parser.add_argument("--il_low", type=float, default=0.2)
        parser.add_argument("--il_high", type=float, default=0.8)
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **opts):
        source = opts["source"]
        sl_low, sl_high = opts["sl_low"], opts["sl_high"]
        il_low, il_high = opts["il_low"], opts["il_high"]
        limit = opts["limit"]

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
            p_il = float(il_score_text(item.text))
            il_action = "BLOCK" if p_il >= 0.5 else "ALLOW"

            # store IL info (no overwrite of SL decision)
            item.il_score = p_il
            item.il_suggested_action = il_action

            # SL action derived from what SL already set
            sl_action = item.final_action  # "ALLOW" or "BLOCK"

            # "strong disagreement" only when both are confident
            sl_conf_allow = (item.sl_score is not None and item.sl_score <= sl_low and sl_action == "ALLOW")
            sl_conf_block = (item.sl_score is not None and item.sl_score >= sl_high and sl_action == "BLOCK")
            il_conf_allow = (p_il <= il_low)
            il_conf_block = (p_il >= il_high)

            disagree = (
                (sl_conf_allow and il_conf_block) or
                (sl_conf_block and il_conf_allow)
            )

            if disagree:
                item.needs_review = True
                flagged += 1

            item.save(update_fields=["il_score", "il_suggested_action", "needs_review"])

        self.stdout.write(self.style.SUCCESS(
            f"Scanned {scanned} rows. Flagged {flagged} for human review."
        ))
