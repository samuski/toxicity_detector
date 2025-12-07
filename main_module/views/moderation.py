# views/moderation.py
import io, csv, json
from dataclasses import dataclass
from typing import List

from django.http import JsonResponse, HttpResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction

from main_module.sl.sl_infer import reload_model, score_text, estimate_uncertainty
from main_module.il.il_infer import il_score_text

from ..models import ModerationItem, ModerationDecision

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

@csrf_exempt
@require_POST
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

@csrf_exempt
@require_POST
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

@csrf_exempt
@require_GET
def api_il_next(request):
    qs_review = ModerationItem.objects.filter(needs_review=True)
    qs_uncertain = ModerationItem.objects.filter(
        status=ModerationItem.Status.UNCERTAIN,
        final_action=ModerationItem.FinalAction.NONE,
        needs_review=False,
    )

    remaining_review = qs_review.count()
    remaining_uncertain = qs_uncertain.count()

    item = (qs_review.order_by("created_at", "id").first()
            or qs_uncertain.order_by("created_at", "id").first())

    if not item:
        return JsonResponse({"remaining_review": 0, "remaining_uncertain": 0, "item": None})

    return JsonResponse({
        "remaining_review": remaining_review,
        "remaining_uncertain": remaining_uncertain,
        "item": {
            "id": item.id,
            "text": item.text,
            "source": item.source,
            "created_at": item.created_at.isoformat(),
            "sl_score": item.sl_score,
            "sl_uncertainty": item.sl_uncertainty,
            "sl_action": item.final_action,          # shows SL’s auto decision if any
            "il_score": item.il_score,
            "il_action": item.il_suggested_action,
            "needs_review": item.needs_review,
            "decision_source": item.decision_source,
        }
    })



@csrf_exempt
@require_POST
def api_il_decide(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    item_id = payload.get("item_id")
    action = payload.get("action")  # "ALLOW" | "BLOCK"
    if not item_id or action not in ("ALLOW", "BLOCK"):
        return JsonResponse({"error": "Need item_id and action in {ALLOW,BLOCK}."}, status=400)

    actor = str(getattr(request.user, "id", None)) if getattr(request, "user", None) and request.user.is_authenticated else None

    with transaction.atomic():
        item = (ModerationItem.objects
            .select_for_update()
            .get(id=item_id)
        )

        # Guard: if already decided, don’t double-label
        if item.final_action != ModerationItem.FinalAction.NONE and not item.needs_review:
            return JsonResponse({"error": "Item already decided."}, status=409)

        ModerationDecision.objects.create(
            item=item,
            action=action,
            source=ModerationDecision.Source.HUMAN,
            actor_id=actor,
            sl_score_at_decision=item.sl_score,
        )

        item.final_action = action
        item.status = ModerationItem.Status.CERTAIN
        item.decision_source = ModerationDecision.Source.HUMAN
        item.needs_review = False
        item.save(update_fields=["final_action", "status", "decision_source", "needs_review", "updated_at"])


    return JsonResponse({"ok": True})

@csrf_exempt
@require_POST
def api_eval(request: HttpRequest):
    f = request.FILES.get("file")
    if not f:
        return JsonResponse({"error": "file is required"}, status=400)

    use_baseline = request.POST.get("sl_baseline") == "1"
    use_il = request.POST.get("il") == "1"

    if not use_baseline and not use_il:
        return JsonResponse({"error": "Select at least one pipeline."}, status=400)

    data = f.read().decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(data))

    if not reader.fieldnames or "text" not in reader.fieldnames:
        return JsonResponse({"error": "CSV must include a text column."}, status=400)

    # accept label or label_score
    label_col = "label" if "label" in reader.fieldnames else ("label_score" if "label_score" in reader.fieldnames else None)
    if not label_col:
        return JsonResponse({"error": "CSV must include label or label_score column."}, status=400)

    texts, y_true = [], []
    for row in reader:
        t = (row.get("text") or "").strip()
        if not t:
            continue
        try:
            y = float(row.get(label_col))
        except:
            continue
        y_true.append(1 if y >= 0.5 else 0)
        texts.append(t)

    if not texts:
        return JsonResponse({"error": "No valid rows found."}, status=400)

    # metric helpers: you can reuse the ones I sent earlier or keep it minimal
    def confusion(y_true, y_pred):
        tn=fp=fn=tp=0
        for yt, yp in zip(y_true, y_pred):
            if yt==0 and yp==0: tn+=1
            elif yt==0 and yp==1: fp+=1
            elif yt==1 and yp==0: fn+=1
            else: tp+=1
        return tn, fp, fn, tp

    def macro_f1(tn, fp, fn, tp):
        # class 1
        p1 = tp / (tp+fp) if (tp+fp) else 0.0
        r1 = tp / (tp+fn) if (tp+fn) else 0.0
        f1_1 = (2*p1*r1/(p1+r1)) if (p1+r1) else 0.0
        # class 0
        p0 = tn / (tn+fn) if (tn+fn) else 0.0
        r0 = tn / (tn+fp) if (tn+fp) else 0.0
        f1_0 = (2*p0*r0/(p0+r0)) if (p0+r0) else 0.0
        return (f1_0 + f1_1) / 2.0, (p0+p1)/2.0, (r0+r1)/2.0

    pipelines = []

    if use_baseline:
        scores = [float(score_text(t)) for t in texts]
        pred = [1 if s >= 0.5 else 0 for s in scores]
        tn, fp, fn, tp = confusion(y_true, pred)
        f1m, pm, rm = macro_f1(tn, fp, fn, tp)
        acc = (tp+tn)/len(y_true)
        pipelines.append({
            "name": "SL baseline",
            "metrics": {"accuracy": acc, "f1_macro": f1m, "precision_macro": pm, "recall_macro": rm, "roc_auc": None, "pr_auc": None},
            "cm": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        })

    if use_il:
        from main_module.il.il_infer import il_score_text
        scores = [float(il_score_text(t)) for t in texts]
        pred = [1 if s >= 0.5 else 0 for s in scores]
        tn, fp, fn, tp = confusion(y_true, pred)
        f1m, pm, rm = macro_f1(tn, fp, fn, tp)
        acc = (tp+tn)/len(y_true)
        pipelines.append({
            "name": "IL",
            "metrics": {"accuracy": acc, "f1_macro": f1m, "precision_macro": pm, "recall_macro": rm, "roc_auc": None, "pr_auc": None},
            "cm": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        })

    return JsonResponse({"pipelines": pipelines, "n": len(texts)})

@csrf_exempt
@require_POST
def api_il_score(request: HttpRequest):
    try:
        payload = json.loads(request.body.decode("utf-8"))
        text = (payload.get("text") or "").strip()
        if not text:
            return JsonResponse({"error": "text is required"}, status=400)
        p = float(il_score_text(text))
        return JsonResponse({"p_toxic": p})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
