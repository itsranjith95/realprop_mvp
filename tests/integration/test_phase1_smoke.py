from src.core.init_db import init_db


def test_db_init_smoke():
    init_db()
    assert True