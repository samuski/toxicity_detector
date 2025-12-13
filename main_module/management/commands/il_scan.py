# main_module/management/commands/il_scan.py
from pathlib import Path

from django.core.management.base import BaseCommand

from main_module.models import ModerationItem
import main_module.il.il_infer as il_infer  # import module, not function


class Command(BaseCommand):
    help = "Scan DB rows with SL decisions and let IL re-define disagreements & uncertain items."

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

        # Optional: decision threshold for IL suggested action
        parser.add_argument("--il_decision_th", type=float, default=0.5)

        # By default we also sweep UNCERTAIN; this flag disables that behaviour
        parser.add_argument(
            "--no-sweep-uncertain",
            action="store_true",
            help="Do NOT run IL over UNCERTAIN items (only scan SL-CERTAIN).",
        )

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
            chosen = Path(f"/app/artifacts/il/iter_{il_iter:03d}/artifacts")

        if chosen is not None:
            if not chosen.is_dir():
                raise SystemExit(f"IL model dir not found: {chosen}")
            il_infer.set_model_dir(str(chosen))
            self.stdout.write(self.style.WARNING(f"[IL] Using model dir: {chosen}"))

        # ------------------------------------------------------------------
        # 1) Run IL on SL-CERTAIN items and *downgrade disagreements* to UNCERTAIN
        # ------------------------------------------------------------------
        qs = ModerationItem.objects.filter(
            decision_source="SL",
            status=ModerationItem.Status.CERTAIN,
        )
        if source:
            qs = qs.filter(source=source)

        qs = qs.order_by("created_at", "id")
        if limit and limit > 0:
            qs = qs[:limit]

        flagged = 0      # number of items downgraded to UNCERTAIN
        scanned = 0

        for item in qs.iterator(chunk_size=500):
            scanned += 1

            # IL probability
            p_il = float(il_infer.il_score_text(item.text))
            il_action = "BLOCK" if p_il >= il_decision_th else "ALLOW"

            # Always store IL info
            item.il_score = p_il
            item.il_suggested_action = il_action

            # SL action derived from what SL already set
            sl_action = item.final_action  # "ALLOW" or "BLOCK"

            # "strong disagreement" only when both are confident
            sl_conf_allow = (
                item.sl_score is not None
                and item.sl_score <= sl_low
                and sl_action == "ALLOW"
            )
            sl_conf_block = (
                item.sl_score is not None
                and item.sl_score >= sl_high
                and sl_action == "BLOCK"
            )
            il_conf_allow = (p_il <= il_low)
            il_conf_block = (p_il >= il_high)

            disagree = (sl_conf_allow and il_conf_block) or (sl_conf_block and il_conf_allow)

            if disagree:
                # Downgrade: IL says "this confident SL decision looks wrong"
                item.status = ModerationItem.Status.UNCERTAIN
                # Optionally tag this source; or leave as None if you prefer
                item.decision_source = None
                item.needs_review = True
                flagged += 1
                update_fields = [
                    "il_score",
                    "il_suggested_action",
                    "status",
                    "decision_source",
                    "needs_review",
                ]
            else:
                # No strong disagreement → keep SL decision, clear any stale flags
                item.needs_review = False
                update_fields = [
                    "il_score",
                    "il_suggested_action",
                    "needs_review",
                ]

            item.save(update_fields=update_fields)

        self.stdout.write(self.style.SUCCESS(
            f"[IL] Scanned {scanned} SL-CERTAIN rows. Downgraded {flagged} to UNCERTAIN for human/IL review."
        ))

        # ------------------------------------------------------------------
        # 2) Sweep UNCERTAIN items: let IL auto-promote very confident ones
        # ------------------------------------------------------------------
        if not opts.get("no_sweep-uncertain"):
            qs_unc = ModerationItem.objects.filter(
                status=ModerationItem.Status.UNCERTAIN,
            ).exclude(decision_source="HUMAN")  # keep human decisions sacred

            if source:
                qs_unc = qs_unc.filter(source=source)

            total_unc = qs_unc.count()
            promoted = 0

            self.stdout.write(
                self.style.WARNING(
                    f"[IL] Sweeping {total_unc} UNCERTAIN items "
                    f"(using il_low={il_low}, il_high={il_high})"
                )
            )

            for item in qs_unc.iterator(chunk_size=500):
                p_il = float(il_infer.il_score_text(item.text))

                new_action = None
                if p_il <= il_low:
                    new_action = "ALLOW"
                elif p_il >= il_high:
                    new_action = "BLOCK"

                if not new_action:
                    # Still ambiguous → remain UNCERTAIN, just store IL score
                    item.il_score = p_il
                    item.save(update_fields=["il_score"])
                    continue

                # Auto-promote this item to CERTAIN(IL)
                item.final_action = new_action
                item.status = ModerationItem.Status.CERTAIN
                item.decision_source = "IL"
                item.il_score = p_il
                item.il_suggested_action = new_action
                item.needs_review = False
                item.save(
                    update_fields=[
                        "final_action",
                        "status",
                        "decision_source",
                        "il_score",
                        "il_suggested_action",
                        "needs_review",
                    ]
                )
                promoted += 1

            self.stdout.write(
                self.style.SUCCESS(
                    f"[IL] UNCERTAIN sweep done. Auto-promoted {promoted}/{total_unc} items."
                )
            )
