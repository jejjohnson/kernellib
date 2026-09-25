"""Dependency boundary guards.

kernellib sits below the probabilistic-modelling layer (pyrox-gp) and must be
usable without it. Importing the package in a fresh interpreter must not pull
in NumPyro or scikit-learn; the modelling and legacy dependencies stay out.
"""

from __future__ import annotations

import subprocess
import sys


FORBIDDEN_ON_IMPORT = ("numpyro", "sklearn")


def test_import_does_not_load_modelling_dependencies() -> None:
    code = (
        "import sys, kernellib; "
        f"loaded = sorted(set({FORBIDDEN_ON_IMPORT!r}) & set(sys.modules)); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
