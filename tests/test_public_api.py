"""Guards on the public API surface.

These tests are deliberately about *contracts*, not behaviour: if a symbol is
removed from ``__all__`` or stops being importable from the top level, that is
a breaking change and should fail loudly here.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import kernellib


# Every non-package module directly under ``kernellib``. A new module must be
# added here (and given an API doc page) for ``test_no_unexpected_submodules``
# to pass.
SUBMODULES: list[str] = []


def test_version_is_a_dotted_string() -> None:
    assert isinstance(kernellib.__version__, str)
    major, minor, patch = kernellib.__version__.split(".")[:3]
    assert all(part.isdigit() for part in (major, minor, patch))


def test_all_is_sorted_and_unique() -> None:
    assert kernellib.__all__ == sorted(kernellib.__all__)
    assert len(kernellib.__all__) == len(set(kernellib.__all__))


@pytest.mark.parametrize("name", kernellib.__all__)
def test_every_exported_name_is_importable(name: str) -> None:
    assert hasattr(kernellib, name), f"{name} is in __all__ but not defined"


def test_no_unexpected_submodules() -> None:
    found = {
        info.name for info in pkgutil.iter_modules(kernellib.__path__) if not info.ispkg
    }
    assert found == set(SUBMODULES)


def test_submodules_import_cleanly_and_declare_all() -> None:
    for module in SUBMODULES:
        imported = importlib.import_module(f"kernellib.{module}")
        assert hasattr(imported, "__all__"), f"kernellib.{module} is missing __all__"
        assert imported.__all__ == sorted(imported.__all__)


def test_functional_all_is_sorted_and_importable() -> None:
    assert kernellib.functional.__all__ == sorted(kernellib.functional.__all__)
    for name in kernellib.functional.__all__:
        assert hasattr(kernellib.functional, name), name


def test_package_is_typed() -> None:
    """PEP 561 marker must ship so downstream type checkers see annotations."""
    marker = Path(kernellib.__path__[0]) / "py.typed"
    assert marker.is_file(), f"missing PEP 561 marker at {marker}"
