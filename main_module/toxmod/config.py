from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import yaml

class Thresholds(BaseModel):
    low: float = 0.3
    high: float = 0.7

class ToxModCfg(BaseModel):
    model: str = "roberta-base"
    artifacts_dir: str = "/artifacts"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 1
    thresholds: Thresholds = Thresholds()
    data: Dict[str, Any] | None = None  # injected from data.yaml

def load_config(cfg_path: str, data_cfg: Optional[str] = None) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if data_cfg:
        with open(data_cfg, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg["data"] = data
    return ToxModCfg(**cfg).model_dump()
