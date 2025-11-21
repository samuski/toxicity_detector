import os
from main_module.eval.sl_infer import score_text, reload_model as _reload
# we call your existing trainer script via function import or subprocess
# simplest: subprocess to keep deps isolated

def train_sl(cfg: dict) -> str:
    env = os.environ.copy()
    env["ARTIFACT_DIR"] = cfg["artifacts_dir"]
    env["SL_MODEL_NAME"] = cfg["model"]
    env["SL_MAX_LEN"] = str(cfg["max_length"])
    env["SL_BSZ"] = str(cfg["batch_size"])
    env["SL_EPOCHS"] = str(cfg["epochs"])
    data = cfg.get("data", {})
    if "csv" in data:
        env["TRAIN_CSV"] = data["csv"]["train"]
        env["VAL_CSV"]   = data["csv"]["val"]
    elif "hf" in data:
        env["HF_NAME"]   = data["hf"]["name"]
        if "config" in data["hf"]:
            env["HF_CONFIG"] = data["hf"]["config"]
    else:
        raise SystemExit("No dataset configured in data.yaml")

    import subprocess, sys
    cmd = [sys.executable, "main_module/eval/sl_train.py"]
    subprocess.run(cmd, check=True, env=env)
    return os.path.join(cfg["artifacts_dir"], "sl", "active")

def predict_one(text: str, cfg: dict) -> float:
    return score_text(text, max_len=cfg["max_length"])

def serve_reload():
    _reload()
