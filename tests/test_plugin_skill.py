"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/kernellib/skills/kernel-methods-with-kernellib/SKILL.md`` is what
agents in downstream projects read before writing kernel methods on
kernellib, so a stale name or a broken example there teaches them the wrong
API. These tests run its worked example and check that every ``kl.X`` /
``kernellib.X`` / ``kernellib.functional.X`` / ``kernellib.sklearn.X`` /
``gaussx.X`` it and the plugin's reviewer name is a current public name.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import jax.numpy as jnp
import pytest


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "kernellib"
SKILL = PLUGIN / "skills" / "kernel-methods-with-kernellib" / "SKILL.md"
AGENT = PLUGIN / "agents" / "kernellib-reuse-reviewer.md"
if not SKILL.is_file():
    # A built distribution ships tests/ without the repository root.
    pytest.skip("the plugin is not present", allow_module_level=True)

_ALIASES = {"kl": "kernellib", "gx": "gaussx"}
_NAME = re.compile(
    r"(?<![\w./])(kl|gx|gaussx|kernellib\.functional|kernellib\.sklearn|kernellib)"
    r"\.([A-Za-z_]\w*)"
)
_BLOCKS = re.compile(r"```python\n(.*?)```", re.S)


@pytest.mark.slow
def test_worked_example_runs_and_its_claims_hold():
    (block,) = [b for b in _BLOCKS.findall(SKILL.read_text()) if "to_operator" in b]
    ns: dict = {}
    exec(block, ns)
    f_test = ns["f_test"]
    rmse = float(jnp.sqrt(jnp.mean((ns["mean"] - f_test) ** 2)))
    # The noise sd is 0.1 and the median-heuristic RBF over-smooths a little;
    # the prose says "about 0.10" (0.102 under x64), and 0.15 still catches a
    # broken solve, which gives an RMSE near the signal's own sd (~0.5).
    assert rmse < 0.15
    assert bool(ns["krr"].converged)
    # The prose says the median-heuristic lengthscale over-smooths.
    assert float(ns["grad"]) > 0
    # Each entry of Phi Phiᵀ - K has sd at most sqrt(2 / R) ≈ 0.0625 for
    # R = 512 cosine features with unit variance; the maximum over the
    # 160,000 entries stays within about 5 sd.
    assert float(ns["gram_err"]) < 5 * (2 / 512) ** 0.5
    # y depends on x₁, so no permutation beats the observed pairing:
    # p = 1 / (199 + 1).
    assert float(ns["test"].p_value) == pytest.approx(0.005)


@pytest.mark.parametrize("path", [SKILL, AGENT], ids=lambda p: p.name)
def test_named_api_is_current(path: Path):
    stale = []
    for prefix, attr in set(_NAME.findall(path.read_text())):
        module_name = _ALIASES.get(prefix, prefix)
        if module_name == "kernellib.sklearn":
            pytest.importorskip("sklearn")
        module = importlib.import_module(module_name)
        if attr in getattr(module, "__all__", dir(module)):
            continue
        try:  # a submodule path such as ``kernellib.functional``
            importlib.import_module(f"{module_name}.{attr}")
        except ImportError:
            stale.append(f"{module_name}.{attr}")
    assert not stale, f"{path.name} names objects that do not exist: {sorted(stale)}"
