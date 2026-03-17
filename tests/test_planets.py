from __future__ import annotations

import pytest

from inferagni import planets


@pytest.mark.unit
def test_import_exoatlas_silently_suppresses_print(monkeypatch, capsys):
    sentinel = object()

    def fake_import_module(name: str):
        assert name == "exoatlas"
        print("import spam")
        return sentinel

    monkeypatch.setattr(planets.importlib, "import_module", fake_import_module)

    got = planets._import_exoatlas_silently()

    assert got is sentinel
    captured = capsys.readouterr()
    assert captured.out == ""
