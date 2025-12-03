import os
from pathlib import Path
from django.http import JsonResponse

ART_DIR = os.environ.get("ARTIFACT_DIR", "/artifacts")
SL_BASELINE_DIR = Path(ART_DIR) / "sl" / "baseline"
SL_ORACLE_DIR   = Path(ART_DIR) / "sl" / "oracle"

def have_model_dir(path: Path) -> bool:
    # light sanity check: HF dir should at least have config + model weights
    return (path / "config.json").exists() and (
        (path / "pytorch_model.bin").exists() or
        any(p.name.endswith(".safetensors") for p in path.glob("*.safetensors"))
    )

def eval_view(request):
    # ... parse CSV, etc. ...
    pipelines = []

    if request.POST.get("sl_baseline") == "1":
        if have_model_dir(SL_BASELINE_DIR):
            pipelines.append(("SL baseline", str(SL_BASELINE_DIR)))
        else:
            # optional: include an error entry instead of silently skipping
            return JsonResponse(
                {"error": "SL baseline model not found at sl/baseline"},
                status=400,
            )

    if request.POST.get("sl_oracle") == "1":
        if have_model_dir(SL_ORACLE_DIR):
            pipelines.append(("SL oracle", str(SL_ORACLE_DIR)))
        else:
            return JsonResponse(
                {"error": "SL oracle model not found at sl/oracle"},
                status=400,
            )

    # ... plus IL / LLM if selected ...

    results = []
    for name, model_dir in pipelines:
        metrics, cm = evaluate_model_on_df(
            model_name_or_path=model_dir,
            df=df,
            text_col=text_col,
            label_col=label_col,
            # reuse your eval.py logic here
        )
        results.append({
            "name": name,
            "metrics": metrics,
            "cm": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
        })

    return JsonResponse({"pipelines": results})
