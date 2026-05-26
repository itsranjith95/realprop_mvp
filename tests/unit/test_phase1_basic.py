from pathlib import Path
from src.core.utils import sanitize_filename


def test_sanitize_filename():
    assert sanitize_filename("my file.pdf") == "my_file.pdf"


def test_sample_dirs_exist():
    assert Path("data/raw/motherdeed").exists()
    assert Path("data/raw/khata").exists()