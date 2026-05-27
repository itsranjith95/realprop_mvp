from pathlib import Path
import shutil, uuid

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def new_id(prefix: str = "doc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def save_upload(upload_file, dest_path: str | Path) -> str:
    dest = Path(dest_path)
    ensure_dir(dest.parent)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return str(dest)