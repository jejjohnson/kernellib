"""Shared pytest configuration.

float64 is enabled for the whole suite, as in gaussx: the moved operator
tests compare against dense references at float64 tolerances.
"""

from __future__ import annotations

import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def getkey() -> eqxi.GetKey:
    """Fresh PRNG keys, seeded from ``EQX_GETKEY_SEED`` when set."""
    return eqxi.GetKey()
