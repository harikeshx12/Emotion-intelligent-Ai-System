import os
import yaml
from pathlib import Path


def load_config(path="config/default.yaml"):
with open(path, "r") as f:
return yaml.safe_load(f)




def ensure_dirs(cfg):
Path(cfg["paths"]["model_dir"]).mkdir(parents=True, exist_ok=True)
Path(cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)
