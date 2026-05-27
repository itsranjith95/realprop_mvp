from functools import lru_cache
from pathlib import Path
from src.core.utils import read_yaml

ROOT = Path.cwd()
CONFIG_PATH = ROOT / "config" / "appconfig.yaml" or ROOT / "config" / "app_config.yml"


@lru_cache
def get_settings() -> dict:
    return read_yaml(CONFIG_PATH)