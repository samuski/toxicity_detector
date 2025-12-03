from django.contrib import admin
from .models import ModerationItem, ModerationDecision


@admin.register(ModerationItem)
class ModerationItemAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "sl_score", "status", "final_action", "created_at")
    list_filter = ("source", "status", "final_action")
    search_fields = ("text", "external_id")


@admin.register(ModerationDecision)
class ModerationDecisionAdmin(admin.ModelAdmin):
    list_display = ("id", "item", "action", "source", "actor_id", "created_at")
    list_filter = ("action", "source")
    search_fields = ("item__text", "actor_id")
