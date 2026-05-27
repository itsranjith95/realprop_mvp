from pathlib import Path

ROOT = Path.cwd()

DIRS = [
    "config",
    "data/raw/motherdeed",
    "data/raw/khata",
    "data/interim",
    "data/ocr",
    "data/extracted",
    "data/reports",
    "models",
    "docker",
    "src",
    "src/app",
    "src/core",
    "src/services",
    "src/pipelines",
    "tests",
    "tests/unit",
    "tests/integration",
    "notebooks",
]

FILES = {
    "README.md": """# RealProp MVP

AI-assisted property legal due-diligence MVP for Bengaluru.
Scope: Mother Deed + Khata only.
Frontend: Streamlit
Backend: Python service modules
Storage: SQLite + local filesystem
Data versioning: DVC
""",
    ".gitignore": """__pycache__/
*.pyc
.venv/
env/
.db
.sqlite3
.DS_Store
dvc.lock
.dvc/cache/
data/raw/**
!data/raw/.gitkeep
!data/raw/motherdeed/.gitkeep
!data/raw/khata/.gitkeep
""",
    "requirements.txt": """streamlit
pydantic
pyyaml
python-dateutil
rapidfuzz
sqlalchemy
""",
    "pyproject.toml": """[project]
name = "realprop-mvp"
version = "0.1.0"
description = "AI-assisted property due diligence MVP for Bengaluru"
requires-python = ">=3.10"

[tool.setuptools]
package-dir = {"" = "src"}
""",
    "dvc.yaml": """stages: {}
""",
    "config/appconfig.yaml": """app:
  name: realprop-mvp
  city: Bengaluru
  environment: dev
  storage_root: data
  db_path: data/app.db
""",
    "config/ocrconfig.yaml": """ocr:
  engine: paddleocr
  language: en
  max_pages: 50
""",
    "config/extractionconfig.yaml": """extraction:
  supported_documents:
    - motherdeed
    - khata
""",
    "config/rulesconfig.yaml": """rules:
  version: v0
  thresholds:
    low_confidence: 0.80
""",
    "config/loggingconfig.yaml": """logging:
  level: INFO
  file: data/app.log
""",
    "src/__init__.py": "",
    "src/app/__init__.py": "",
    "src/app/main_app.py": '''import streamlit as st

st.set_page_config(page_title="RealProp MVP", layout="wide")
st.title("RealProp MVP")
st.text_input("Case ID")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.button("Mother Deed")
col2.button("Khata")
col3.button("Upload")
col4.button("Cancel")
col5.button("Process")
col6.button("Download Report", disabled=True)

st.info("Phase 0 scaffold is ready.")
''',
    "src/app/components.py": '''def section_header(title: str) -> str:
    return f"## {title}"
''',
    "src/core/__init__.py": "",
    "src/core/enums.py": '''from enum import Enum

class DocumentType(str, Enum):
    MOTHER_DEED = "motherdeed"
    KHATA = "khata"

class CaseStatus(str, Enum):
    DRAFT = "draft"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    REVIEW_REQUIRED = "review_required"
    REVIEWED = "reviewed"

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    OCR_DONE = "ocr_done"
    EXTRACTED = "extracted"
''',
    "src/core/models.py": '''from pydantic import BaseModel
from datetime import datetime
from .enums import CaseStatus, DocumentStatus, DocumentType

class Case(BaseModel):
    id: str
    created_at: datetime
    status: CaseStatus = CaseStatus.DRAFT
    city: str = "Bengaluru"
    property_description: str | None = None

class Document(BaseModel):
    id: str
    case_id: str
    doc_type: DocumentType
    path: str
    status: DocumentStatus = DocumentStatus.UPLOADED
''',
    "src/core/utils.py": '''from pathlib import Path
from datetime import datetime
import uuid

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def utc_now():
    return datetime.utcnow()
''',
    "src/services/__init__.py": "",
    "src/services/case_service.py": '''from src.core.models import Case
from src.core.utils import utc_now

def create_case(case_id: str, property_description: str | None = None) -> Case:
    return Case(
        id=case_id,
        created_at=utc_now(),
        property_description=property_description,
    )
''',
    "src/services/document_service.py": '''from pathlib import Path
from src.core.models import Document
from src.core.enums import DocumentType
from src.core.utils import generate_id, ensure_dir

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

def validate_file(filename: str):
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {suffix}"
    return True, ""

def build_storage_path(case_id: str, doc_type: str, filename: str) -> Path:
    folder = ensure_dir(Path("data/raw") / case_id / doc_type)
    return folder / filename

def create_document_record(case_id: str, doc_type: str, path: str) -> Document:
    return Document(
        id=generate_id("doc"),
        case_id=case_id,
        doc_type=DocumentType(doc_type),
        path=path,
    )
''',
    "src/services/report_service.py": '''def build_report(case_id: str) -> str:
    return f"Report generation placeholder for case: {case_id}"
''',
    "src/pipelines/__init__.py": "",
    "src/pipelines/intake_pipeline.py": '''from src.services.case_service import create_case

def run_intake(case_id: str, property_description: str | None = None):
    return create_case(case_id=case_id, property_description=property_description)
''',
    "src/pipelines/ingestion_pipeline.py": '''from src.services.document_service import validate_file, build_storage_path, create_document_record

def run_ingestion(case_id: str, doc_type: str, filename: str):
    ok, err = validate_file(filename)
    if not ok:
        raise ValueError(err)

    path = build_storage_path(case_id, doc_type, filename)
    return create_document_record(case_id=case_id, doc_type=doc_type, path=str(path))
''',
    "tests/unit/test_placeholder.py": '''def test_placeholder():
    assert True
''',
    "tests/integration/test_placeholder.py": '''def test_integration_placeholder():
    assert True
''',
    "notebooks/.gitkeep": "",
    "data/raw/.gitkeep": "",
    "data/raw/motherdeed/.gitkeep": "",
    "data/raw/khata/.gitkeep": "",
    "data/interim/.gitkeep": "",
    "data/ocr/.gitkeep": "",
    "data/extracted/.gitkeep": "",
    "data/reports/.gitkeep": "",
    "models/.gitkeep": "",
    "docker/Dockerfile.app": """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "src/app/main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
""",
}

def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        print(f"[CREATED] {path.relative_to(ROOT)}")
    else:
        print(f"[SKIPPED] {path.relative_to(ROOT)} already exists")

def main():
    print(f"Scaffolding inside current folder: {ROOT}")

    for d in DIRS:
        dir_path = ROOT / d
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] {dir_path.relative_to(ROOT)}")

    for relative_path, content in FILES.items():
        write_file(ROOT / relative_path, content)

    print("\\nScaffold setup complete.")

if __name__ == "__main__":
    main()