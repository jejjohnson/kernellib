"""Dependency boundary guards.

kernellib sits below the probabilistic-modelling layer (pyrox-gp) and must be
usable without it. Importing the package in a fresh interpreter must not pull
in NumPyro or scikit-learn; the modelling and legacy dependencies stay out, and
so do the docs-only training tools (pipekit, pipekit-train, optax).
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# pipekit / pipekit-train and optax are docs-group dependencies for the
# notebooks' training loops (roadmap decision 7); the library never imports them.
FORBIDDEN_ON_IMPORT = (
    "numpyro",
    "sklearn",
    "pynndescent",
    "numba",
    "pipekit",
    "pipekit_train",
    "optax",
)


@pytest.mark.slow
def test_sklearn_adapter_is_opt_in() -> None:
    # The adapter package loads scikit-learn only when imported explicitly.
    code = (
        "import sys, kernellib; "
        "assert 'kernellib.sklearn' not in sys.modules; "
        "import kernellib.sklearn; "
        "assert 'sklearn' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.slow
def test_import_does_not_load_modelling_dependencies() -> None:
    code = (
        "import sys, kernellib; "
        f"loaded = sorted(set({FORBIDDEN_ON_IMPORT!r}) & set(sys.modules)); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
