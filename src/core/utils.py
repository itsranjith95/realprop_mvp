import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def read_yaml(path: str | Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def sha256_for_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: str | Path, dst: str | Path) -> None:
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)


def sanitize_filename(filename: str) -> str:
    keep = []
    for ch in filename:
        if ch.isalnum() or ch in {".", "_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)