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

        # Decision threshold for IL suggested action (for BLOCK vs ALLOW label)
        parser.add_argument("--il_decision_th", type=float, default=0.5)

        # By default we also sweep UNCERTAIN; this flag disables that behaviour
        parser.add_argument(
            "--no-sweep-uncertain",
            action="store_true",
            help="Do NOT run IL over UNCERTAIN items (only scan SL-CERTAIN).",
        )

        # NEW: optionally re-check IL-CERTAIN items using the *newest* IL model
        parser.add_argument(
            "--review-confident-items",
            action="store_true",
            help=(
                "Re-score items already decided by IL (CERTAIN+IL) and "
                "downgrade to UNCERTAIN if the new IL model is no longer confident "
                "or strongly disagrees with the previous IL score."
            ),
        )

    def handle(self, *args, **opts):
        source = opts["source"]
        sl_low, sl_high = float(opts["sl_low"]), float(opts["sl_high"])
        il_low, il_high = float(opts["il_low"]), float(opts["il_high"])
        limit = int(opts["limit"]) if opts["limit"] else 0
        il_iter = int(opts["il_iter"])
        il_dir = (opts["il_dir"] or "").strip()
        il_decision_th = float(opts["il_decision_th"])
        review_confident = bool(opts.get("review_confident_items"))

        # --- Pick IL model dir (joblib artifacts) ---
        chosen = None
        if il_dir:
            chosen = Path(il_dir)
        elif il_iter > 0:
            # adjust this root to match your actual IL_DIR if needed
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

        scanned = 0
        downgraded_from_sl = 0

        for item in qs.iterator(chunk_size=500):
            scanned += 1

            # New IL probability
            p_il = float(il_infer.il_score_text(item.text))
            il_action = "BLOCK" if p_il >= il_decision_th else "ALLOW"

            # Always store latest IL info
            item.il_score = p_il
            item.il_suggested_action = il_action

            # SL action already stored as final_action
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
                item.decision_source = None
                item.needs_review = True
                downgraded_from_sl += 1
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

        self.stdout.write(
            self.style.SUCCESS(
                f"[IL] Scanned {scanned} SL-CERTAIN rows. "
                f"Downgraded {downgraded_from_sl} to UNCERTAIN for review."
            )
        )

        # ------------------------------------------------------------------
        # 2) Sweep UNCERTAIN items: let IL auto-promote very confident ones
        # ------------------------------------------------------------------
        if not opts.get("no_sweep_uncertain"):
            qs_unc = ModerationItem.objects.filter(
                status=ModerationItem.Status.UNCERTAIN,
            ).exclude(decision_source="HUMAN")  # keep HUMAN decisions untouched

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

        # ------------------------------------------------------------------
        # 3) OPTIONAL: dredge IL-CERTAIN items using the *new* IL model
        # ------------------------------------------------------------------
        if review_confident:
            qs_conf = ModerationItem.objects.filter(
                status=ModerationItem.Status.CERTAIN,
                decision_source="IL",
            )
            if source:
                qs_conf = qs_conf.filter(source=source)

            total_conf = qs_conf.count()
            redowngraded = 0

            self.stdout.write(
                self.style.WARNING(
                    f"[IL] Re-checking {total_conf} IL-CERTAIN items "
                    f"(review_confident_items, il_low={il_low}, il_high={il_high})"
                )
            )

            for item in qs_conf.iterator(chunk_size=500):
                prev_il = item.il_score  # score from previous IL iteration
                p_il = float(il_infer.il_score_text(item.text))
                item.il_score = p_il  # always store latest score

                # Old IL confidence side
                prev_conf_allow = prev_il is not None and prev_il <= il_low
                prev_conf_block = prev_il is not None and prev_il >= il_high

                # New IL confidence side
                new_conf_allow = p_il <= il_low
                new_conf_block = p_il >= il_high

                # If the *side* of decision flips while both are confident,
                # or the new IL falls into the ambiguous middle region,
                # we downgrade to UNCERTAIN again.
                flip = (
                    (prev_conf_allow and new_conf_block) or
                    (prev_conf_block and new_conf_allow)
                )
                now_ambiguous = not (new_conf_allow or new_conf_block)

                if flip or now_ambiguous:
                    item.status = ModerationItem.Status.UNCERTAIN
                    item.decision_source = None
                    item.needs_review = True
                    item.il_suggested_action = None
                    redowngraded += 1
                    item.save(
                        update_fields=[
                            "status",
                            "decision_source",
                            "needs_review",
                            "il_score",
                            "il_suggested_action",
                        ]
                    )
                else:
                    # Still confident and consistent → keep as IL decision
                    item.needs_review = False
                    item.il_suggested_action = (
                        "BLOCK" if p_il >= il_decision_th else "ALLOW"
                    )
                    item.save(
                        update_fields=[
                            "needs_review",
                            "il_score",
                            "il_suggested_action",
                        ]
                    )

            self.stdout.write(
                self.style.SUCCESS(
                    f"[IL] review-confident-items: downgraded {redowngraded}/{total_conf} IL-CERTAIN items to UNCERTAIN."
                )
            )
