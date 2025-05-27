# src/config.py
from pathlib import Path
import yaml, pkg_resources  # built-in yaml if you pip-installed pyyaml

_cfg_file = Path(__file__).parents[1] / "config.yaml"
if not _cfg_file.exists():                         # CI fallback
    _cfg_file = _cfg_file.with_name("config_example.yaml")

with _cfg_file.open() as f:
    _cfg = yaml.safe_load(f)

DATA_DIR    = Path(_cfg["data_dir"]).expanduser().resolve()
RESULTS_DIR = Path(_cfg["results_dir"]).expanduser().resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
