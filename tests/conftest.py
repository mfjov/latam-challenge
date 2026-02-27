import os
import pytest


@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch):
    """Cambia el CWD a tests/ para que las rutas relativas como ../data/data.csv resuelvan bien."""
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__)))
