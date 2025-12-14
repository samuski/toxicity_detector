# main_module/urls.py
from django.contrib import admin
from django.urls import path
from django.views.generic import RedirectView

from main_module.views.moderation import (
    moderation_dashboard,
    api_score,
    api_batch_score,
    api_eval,
    api_il_next,
    api_il_decide,
    api_score_multi
)

from main_module.views.moderation import api_il_score

urlpatterns = [
    path("", RedirectView.as_view(url="/moderation/", permanent=False)),
    path("admin/", admin.site.urls),

    path("moderation/", moderation_dashboard, name="moderation_dashboard"),
    path("api/moderation/score_multi", api_score_multi, name="api_score_multi"),
    path("api/moderation/batch", api_batch_score, name="api_batch_score"),

    path("api/moderation/eval", api_eval, name="api_eval"),
    path("api/moderation/il/next", api_il_next, name="api_il_next"),
    path("api/moderation/il/decide", api_il_decide, name="api_il_decide"),
    path("api/moderation/il/score", api_il_score, name="api_il_score"),

]
