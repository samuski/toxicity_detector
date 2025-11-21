import typer, json, os
from .config import load_config
from .pipeline import train_sl, predict_one, serve_reload

app = typer.Typer(add_completion=False)

@app.command()
def train(cfg_path: str = "toxmod.yaml", data_cfg: str = "data.yaml"):
    cfg = load_config(cfg_path, data_cfg)
    out = train_sl(cfg)  # returns save path
    typer.echo(json.dumps({"saved_to": out}))

@app.command()
def predict(text: str, cfg_path: str = "toxmod.yaml"):
    cfg = load_config(cfg_path)
    p = predict_one(text, cfg)
    # ternary mapping using thresholds
    low, high = cfg["thresholds"]["low"], cfg["thresholds"]["high"]
    label = "non-toxic" if p <= low else "toxic" if p >= high else "unsure"
    typer.echo(json.dumps({"score": round(p, 3), "label": label}))

@app.command()
def reload_model():
    serve_reload()
    typer.echo("reloaded")

if __name__ == "__main__":
    app()
