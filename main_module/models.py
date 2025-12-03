from django.db import models


class ModerationItem(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"          # just ingested, not routed yet (optional)
        UNCERTAIN = "UNCERTAIN", "Uncertain"    # needs human/IL
        CERTAIN = "CERTAIN", "Certain"          # SL/IL confident enough

    class FinalAction(models.TextChoices):
        NONE = "NONE", "None"                   # no final decision yet
        ALLOW = "ALLOW", "Allow"
        BLOCK = "BLOCK", "Block"
        ESCALATE = "ESCALATE", "Escalate"       # for more complex flows, optional

    # Raw content
    external_id = models.CharField(
        max_length=128,
        blank=True,
        null=True,
        help_text="Original ID from source CSV / dataset (if any).",
    )
    source = models.CharField(
        max_length=64,
        blank=True,
        null=True,
        help_text="Source tag (e.g., 'jigsaw', 'hatexplain', 'live').",
    )
    text = models.TextField()

    # SL model outputs at ingestion time
    sl_score = models.FloatField(
        help_text="Supervised model score in [0,1] at time of ingestion.",
    )
    sl_uncertainty = models.FloatField(
        blank=True,
        null=True,
        help_text="Uncertainty estimate (e.g., p*(1-p)).",
    )

    # Optional: ground truth if this is from a labeled dataset
    true_label = models.FloatField(
        blank=True,
        null=True,
        help_text="Optional ground-truth label (e.g., regression score or 0/1).",
    )

    # Pipeline state
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.UNCERTAIN,
        db_index=True,   # important for queue-like queries
    )
    final_action = models.CharField(
        max_length=16,
        choices=FinalAction.choices,
        default=FinalAction.NONE,
        db_index=True,
    )

    # Audit
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    # For later IL: store which policy decided this (SL threshold, IL model, human, etc.)
    decision_source = models.CharField(
        max_length=32,
        blank=True,
        null=True,
        help_text="Who made the final decision: SL, IL, HUMAN, RULE, etc.",
    )

    class Meta:
        indexes = [
            # Fast “queue” access: find uncertain items in time order
            models.Index(fields=["status", "created_at"]),
        ]

    def __str__(self):
        return f"[{self.pk}] {self.source or 'src'} :: {self.text[:80]}..."


class ModerationDecision(models.Model):
    class Action(models.TextChoices):
        ALLOW = "ALLOW", "Allow"
        BLOCK = "BLOCK", "Block"
        ESCALATE = "ESCALATE", "Escalate"

    class Source(models.TextChoices):
        HUMAN = "HUMAN", "Human"
        IL = "IL", "Imitation policy"
        SL = "SL", "Supervised baseline"
        RULE = "RULE", "Hand-coded rule"
        SYSTEM = "SYSTEM", "System default"

    item = models.ForeignKey(
        ModerationItem,
        on_delete=models.CASCADE,
        related_name="decisions",
    )

    action = models.CharField(
        max_length=16,
        choices=Action.choices,
    )
    source = models.CharField(
        max_length=16,
        choices=Source.choices,
    )

    # Optional: who the human was (if applicable)
    actor_id = models.CharField(
        max_length=64,
        blank=True,
        null=True,
        help_text="Moderator/user ID if the decision was human.",
    )

    # Snapshot of model state when the decision was made
    sl_score_at_decision = models.FloatField(
        blank=True,
        null=True,
        help_text="SL score at time of decision (for auditing IL vs SL).",
    )
    il_score_at_decision = models.FloatField(
        blank=True,
        null=True,
        help_text="Optional IL confidence / logit.",
    )

    comment = models.TextField(
        blank=True,
        null=True,
        help_text="Optional reasoning or notes for this decision.",
    )

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.source} {self.action} on item {self.item_id}"
