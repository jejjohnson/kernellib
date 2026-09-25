---
applyTo: "docs/**/*.ipynb,docs/**/*.py,docs/**/*.md,notebooks/**/*.ipynb"
---

# Documentation Examples — Standards & Workflow

## Overview

Example notebooks live in `docs/notebooks/` as **executed `.ipynb` files**. The committed `.ipynb` carries both source cells and rendered cell outputs (including matplotlib figures as embedded PNGs). **mystmd** renders them without re-executing, so the committed outputs are what users see in the docs.

Every notebook is **Google Colab compatible** — the first cell detects Colab and `pip install`s the right dependencies so users can click "Open in Colab" and run end-to-end without touching the host environment.

## Directory Layout

```
docs/
├── notebooks/
│   ├── demo_foo.ipynb
│   ├── tutorial_bar.ipynb
│   └── ...
└── api/
    └── ...
```

No separate `images/` directory — figures live inside the `.ipynb` cell outputs.

## Authoring Workflow

**Develop in jupytext percent format**, then convert and execute for the final commit. Plain `.py` diffs are vastly easier to review than raw `.ipynb` JSON, so do the substantive editing on the `.py` side.

1. Create `docs/notebooks/foo.py` in jupytext percent format (header below).
2. Iterate — edit, smoke-run via `uv run --group docs python docs/notebooks/foo.py`, repeat.
3. When satisfied, convert to `.ipynb`:

   ```bash
   uv run --group docs jupytext --to notebook docs/notebooks/foo.py
   ```

4. Execute in place so cell outputs are embedded:

   ```bash
   uv run --group docs jupyter nbconvert --to notebook \
     --execute docs/notebooks/foo.ipynb \
     --inplace \
     --ExecutePreprocessor.timeout=180
   ```

5. Lint the notebook, **not** the `.py`:

   ```bash
   uv run --group lint ruff format docs/notebooks/
   uv run --group lint ruff check docs/notebooks/
   ```

6. Delete the `.py` — the `.ipynb` is the committed source of truth.
7. Confirm the docs still build strictly, since a new notebook needs a nav
   entry:

   ```bash
   make docs
   ```

   This builds both halves of the site, assembles them, and verifies that
   every internal link resolves — including links from the notebook prose
   into the API reference.

8. Commit the `.ipynb` and add it to the `toc` in `docs/myst.yml`.

## Notebook Basenames Must Be Unique

mystmd derives a page's URL from its **basename**, ignoring the directory. A
notebook named `quickstart.ipynb` therefore collides with a guide page named
`quickstart.md`, and mystmd silently disambiguates them with an
order-dependent `-1` suffix — an unstable URL that changes when files are
added. Frontmatter `slug:` is ignored, so the only fix is a distinct basename.

Keep basenames unique across `docs/`, `docs/guide/`, and `docs/notebooks/`.

## Ruff Lints Notebook Code Cells

`ruff check .` — what CI runs — includes the **code cells** of every
`docs/notebooks/*.ipynb`. Two consequences:

- **Code cell lines must be ≤ 88 characters**, same as the rest of the repo.
  Long `ax.plot(...)` calls are the usual offender; bind an intermediate
  variable rather than letting the line run.
- **Lint the `.ipynb`, never the jupytext `.py`.** The `.py` will report E501
  on every markdown cell, because the standard below requires each markdown
  paragraph to be one long line. It will also report `I001`, because ruff sees
  the cell-separated import blocks as one unsorted block. Both are artefacts
  of the flat `.py` view and neither applies to the notebook, where ruff lints
  each cell independently. This is a large part of why step 6 deletes the
  `.py`.

If you need to fix a lint error after executing, edit the `.py` (regenerate it
with `jupytext --to py:percent foo.ipynb`), re-convert, and **re-execute** —
never hand-edit a committed `.ipynb`'s source cells, or the outputs stop
matching the code that supposedly produced them.

## Jupytext Header (dev only)

While developing in `.py`, start the file with:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Cell Markers (dev only)

- **Code cells**: `# %%`
- **Markdown cells**: `# %% [markdown]` followed by `#`-prefixed lines

```python
# %% [markdown]
# # Title
#
# Some explanation with LaTeX: $\nabla^2 \psi = f$

# %%
import numpy as np
```

## First Markdown Cell — Title + Colab Badge

Every notebook opens with a `#`-level title and a Colab badge pointing at its `main`-branch URL. Replace `OWNER/REPO` below with the actual GitHub owner and repository name (e.g. from `repo_url` in `mkdocs.yml`):

```markdown
# Demo — Feature Overview

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/docs/notebooks/demo_foo.ipynb)

<one-paragraph summary>

**What you'll learn:**

1. ...
2. ...
3. ...
```

## First Code Cell — Colab Detection + Install

Detect Colab, install the package only when needed (substitute `OWNER/REPO` with the actual GitHub owner and repository name):

```python
import subprocess
import sys

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "kernellib @ git+https://github.com/OWNER/REPO@main",
        ],
        check=True,
    )
```

Local / CI users with the environment set up skip the install and go straight to imports.

## Imports + Warnings

```python
import warnings

warnings.filterwarnings("ignore", message=r".*IProgress.*")

import numpy as np
import matplotlib.pyplot as plt
# ... other imports
```

- Suppress the IProgress warning from ipywidgets so the output is clean.

## Reproducibility — `watermark`

After imports, print a version readout. The cell uses `get_ipython()` and an `importlib.util.find_spec` check so a plain `python foo.py` smoke run during dev — and a local `nbconvert --execute` on a machine without `watermark` installed — both no-op cleanly instead of raising `UsageError: Line magic function %load_ext not found`:

```python
import importlib.util

try:
    from IPython import get_ipython

    ipython = get_ipython()
except ImportError:
    ipython = None

if ipython is not None and importlib.util.find_spec("watermark") is not None:
    ipython.run_line_magic("load_ext", "watermark")
    ipython.run_line_magic(
        "watermark",
        "-v -m -p numpy,matplotlib,kernellib",
    )
else:
    print("watermark extension not installed; skipping reproducibility readout.")
```

The `[docs]` dependency group pulls in `watermark`, so under the documented authoring workflow (`uv run --group docs jupyter nbconvert --execute ...`) the readout actually prints. Users reproducing the notebook later see exactly which package versions generated the committed outputs.

## Notebook Structure

1. **Title + Colab badge** (markdown)
2. **Background** (markdown) — motivation, math, what the user will learn
3. **Setup** — Colab detection + install
4. **Imports + config + watermark**
5. **Problem setup** — data, grids, initial conditions
6. **Core demonstration(s)** — alternating markdown and code
7. **Visualizations** — figures are embedded directly as cell outputs
8. **Summary / takeaways**

## Matplotlib Style

**Defaults only** — no `plt.style.use` and no `rcParams` tweaks:

- `C0`, `C1`, `C2` (matplotlib defaults) for main series.
- `"k--"` for truth / reference lines.
- `figsize=(12, 5)` for single plots, `(18, 5)` for 1×3 comparison grids.
- `ax.scatter(...)` for data points.
- `ax.fill_between(..., alpha=0.2)` for uncertainty bands.

## Markdown Paragraph Wrapping

**Each paragraph in a `# %% [markdown]` block must be a single long line.** Do not soft-wrap paragraph text across multiple `#` lines. jupytext preserves source newlines as soft breaks, which mkdocs-jupyter renders as awkward visual breaks.

Right:

```python
# %% [markdown]
# This notebook demonstrates the key features of the package. We walk through all the main patterns using a simple example so the only thing that differs is how the parameters are configured.
```

Wrong:

```python
# %% [markdown]
# This notebook demonstrates the key features of the package. We walk
# through all the main patterns using a simple example so the only
# thing that differs is how the parameters are configured.
```

Lines that *must* stay on their own (do not join):

- Headings: `# # Title`, `# ## Section`
- Display math: `# $$...$$` block (one expression per line)
- Table rows: `# | col | col |`
- List item heads: `# - item` or `# 1. item`
- Code-fence delimiters: `# ``` ` and contents inside the fence
- Blockquotes: `# > quote`

## Math in Markdown Cells

Inline: `$\|x - x'\|^2$`.
Display:

```markdown
$$f(x) = \sum_{i=1}^{N} w_i \phi_i(x)$$
```

MathJax is configured in `mkdocs.yml` — both inline and display math render in the docs.

## Checklist for New Notebooks

- [ ] Authored in jupytext `.py` percent format during development
- [ ] First markdown cell: `#`-level title + Colab badge
- [ ] Second markdown cell: background + math + "What you'll learn"
- [ ] Setup cell: Colab detection + package install via `subprocess`
- [ ] `warnings.filterwarnings(..., IProgress, ...)`
- [ ] `%watermark` version readout
- [ ] Matplotlib defaults only (no `style.use`, no `rcParams`)
- [ ] Converted to `.ipynb` and executed in place
- [ ] `ruff check docs/notebooks/` passes on the `.ipynb` (code cells ≤ 88 chars)
- [ ] No cell output has `output_type: error`
- [ ] `.py` deleted; `.ipynb` with embedded outputs committed
- [ ] Listed in the `toc` in `docs/myst.yml`
- [ ] Basename unique across `docs/`
- [ ] `make docs` passes (both halves build, all links resolve)
- [ ] Every numeric claim in the prose matches the executed output — if a cell
      prints a table, the paragraph describing it must agree with the numbers
      actually printed, not the ones you expected
