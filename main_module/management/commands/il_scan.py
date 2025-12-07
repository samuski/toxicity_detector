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

        # NEW:
        parser.add_argument("--il_iter", type=int, default=0,
                            help="IL iteration to use (e.g. 1,2,3). 0 = do not override.")
        parser.add_argument("--il_dir", default="",
                            help="Explicit IL artifacts dir (overrides --il_iter).")

    def handle(self, *args, **opts):
        source = opts["source"]
        sl_low, sl_high = opts["sl_low"], opts["sl_high"]
        il_low, il_high = opts["il_low"], opts["il_high"]
        limit = opts["limit"]

        il_iter = int(opts["il_iter"])
        il_dir = (opts["il_dir"] or "").strip()

        # Choose IL model dir
        if il_dir:
            chosen = Path(il_dir)
        elif il_iter > 0:
            chosen = Path(f"/app/media/il/iter_{il_iter:03d}/artifacts")
        else:
            # fall back to whatever il_infer defaults to (env IL_MODEL_DIR, etc.)
            chosen = None

        if chosen is not None:
            if not chosen.is_dir():
                raise SystemExit(f"IL model dir not found: {chosen}")
            il_infer.set_model_dir(str(chosen))

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
            p_il = float(il_infer.il_score_text(item.text))
            il_action = "BLOCK" if p_il >= 0.5 else "ALLOW"

            item.il_score = p_il
            item.il_suggested_action = il_action

            sl_action = item.final_action  # "ALLOW" or "BLOCK"

            sl_conf_allow = (item.sl_score is not None and item.sl_score <= sl_low and sl_action == "ALLOW")
            sl_conf_block = (item.sl_score is not None and item.sl_score >= sl_high and sl_action == "BLOCK")
            il_conf_allow = (p_il <= il_low)
            il_conf_block = (p_il >= il_high)

            disagree = ((sl_conf_allow and il_conf_block) or (sl_conf_block and il_conf_allow))

            # IMPORTANT: also clear old flags when no longer disagreeing
            item.needs_review = bool(disagree)
            if disagree:
                flagged += 1

            item.save(update_fields=["il_score", "il_suggested_action", "needs_review"])

        self.stdout.write(self.style.SUCCESS(
            f"Scanned {scanned} rows. Flagged {flagged} for human review."
        ))
