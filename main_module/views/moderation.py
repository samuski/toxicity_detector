# views/moderation.py
import io, csv, json
from dataclasses import dataclass
from typing import List
from django.http import JsonResponse, HttpResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from main_module.eval.sl_infer import reload_model, score_text, estimate_uncertainty

# Warm model once when the module loads (safe with Gunicorn workers)
reload_model()

# In-memory ring buffer for recent results (simple; swap to DB later if needed)
MAX_HISTORY = 200
_history: List[dict] = []

def _push_history(row: dict):
    _history.append(row)
    if len(_history) > MAX_HISTORY:
        del _history[: len(_history) - MAX_HISTORY]

def moderation_dashboard(request: HttpRequest):
    # Show the page & recent results
    return render(request, "moderation/dashboard.html", {"history": list(reversed(_history))})

@require_http_methods(["POST"])
def api_score(request: HttpRequest):
    """
    JSON in: {"text": "your string", "max_len": 256}
    JSON out: {"p_toxic": float, "uncertainty": float}
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
        text = (payload.get("text") or "").strip()
        max_len = int(payload.get("max_len") or 256)
        if not text:
            return JsonResponse({"error": "text is required"}, status=400)
        p = float(score_text(text, max_len=max_len))
        u = float(estimate_uncertainty(p))
        row = {"text": text, "p_toxic": p, "uncertainty": u}
        _push_history(row)
        return JsonResponse(row)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt  # if you call from non-browser tools; remove if you’ll send CSRF tokens
@require_http_methods(["POST"])
def api_batch_score(request: HttpRequest):
    """
    Multipart form: file=<csv with a 'text' column>
    Returns a CSV with columns: text,p_toxic,uncertainty,pred_label
    """
    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"error": "file is required"}, status=400)

    try:
        data = f.read().decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(data))
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["text", "p_toxic", "uncertainty", "pred_label"])

        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            p = float(score_text(text))
            u = float(estimate_uncertainty(p))
            y = 1 if p >= 0.5 else 0
            w.writerow([text, f"{p:.6f}", f"{u:.6f}", y])

        out.seek(0)
        return HttpResponse(
            out.read(),
            content_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="scored.csv"'},
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
