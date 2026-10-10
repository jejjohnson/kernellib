---
name: add-notebook
description: Add or update an executed example notebook in kernellib's docs — jupytext percent .py while drafting, an executed .ipynb as the committed source, a toc entry in docs/myst.yml, xref links into the API reference, ruff on the notebook cells, and a strict docs build. Use when asked to write, extend, re-run or fix a tutorial / example notebook under docs/notebooks.
---

# Add or update a notebook

The full standard is `.github/instructions/docs-examples.instructions.md`;
read it first. This is the checklist.

## 1. Plan it

- One question per notebook; check the existing ones in `docs/notebooks/`
  (and `docs/myst.yml`'s toc) so you extend rather than duplicate.
- The basename must be unique across `docs/`, `docs/guide/` and
  `docs/notebooks/`: mystmd builds flat URLs from basenames, underscores
  becoming hyphens (`kernel_approximations.ipynb` →
  `/kernel-approximations/`), and a clash gets an unstable `-1` suffix.
- Use the public API (`import kernellib as kl`, `import gaussx as gx`) the
  way a user would; no private names. Training loops may use the docs
  group's optax / pipekit-train; the library never does.

## 2. Draft in `.py`, commit the `.ipynb`

1. Write `docs/notebooks/<name>.py` in jupytext percent format (header in
   the instructions file); first markdown cell: title + Colab badge.
2. Smoke-run: `uv run --group docs python docs/notebooks/<name>.py`.
3. Convert and execute:

   ```bash
   uv run --group docs jupytext --to notebook docs/notebooks/<name>.py
   uv run --group docs jupyter nbconvert --to notebook --execute \
     docs/notebooks/<name>.ipynb --inplace --ExecutePreprocessor.timeout=180
   ```

4. Delete the `.py`; the executed `.ipynb` is the source of truth. To fix
   something later, regenerate the `.py` (`jupytext --to py:percent`),
   edit, re-convert and **re-execute**; never hand-edit committed source
   cells.

## 3. Content rules

- Figures inline with `plt.show()`: no `savefig`, no committed PNGs; keep
  the notebook under the pre-commit 1 MB limit (fewer, smaller figures;
  `dpi` around 100).
- Link API names with `[`kl.RBF`](xref:api#kernellib.RBF)` — a missing
  target fails the strict build. Cite with the `bib/*.bib` files.
- Random draws from explicit keys so a re-run reproduces the outputs.
- Code cells ≤ 88 characters (ruff lints the `.ipynb` cells, not the
  `.py`).

## 4. Wire it in and verify

- Add `- file: notebooks/<name>.ipynb` under the right section of the
  `toc` in `docs/myst.yml`.
- `uv run --group lint ruff format docs/notebooks/` and
  `uv run --group lint ruff check .`
- `make docs-check` (strict validation of both halves) and `make docs`
  (builds, assembles and checks every internal link) both need the mystmd
  CLI (`npm install -g mystmd`); if the theme download is blocked, see
  `docs/README.md`. Without mystmd, run `make docs-api` and say in the PR
  that the prose half was not checked.
