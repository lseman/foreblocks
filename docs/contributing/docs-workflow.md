---
title: Documentation Workflow
description: How to write, edit, and maintain docs for foreblocks.
editLink: true
---

# Documentation Workflow

This repository uses in-repo, versioned documentation instead of a separate GitHub Wiki.

## Why this approach

- docs stay aligned with the code
- documentation changes can be reviewed in pull requests
- examples and notebooks can evolve alongside implementation
- contributors do not need a separate editing surface

## Structure

The documentation system is organized as:

- [`docs/index.md`](https://github.com/lseman/foreblocks/blob/main/docs/index): docs home
- `docs/tutorials/`: runnable onboarding and practical workflows
- `docs/architecture/`: internal structure and design pages
- `docs/reference/`: API and repository reference
- root-level guide pages in `docs/`: subsystem-focused guides
- [`site/landing/index.html`](https://github.com/lseman/foreblocks/blob/main/site/landing/index.html): static landing page source for the site root
- [`.github/scripts/compile_docs.py`](https://github.com/lseman/foreblocks/blob/main/.github/scripts/compile_docs.py): Python compiler and navigation for the `/docs/` site

## When to update docs

Update documentation whenever you:

- add or remove public exports
- change constructor contracts or tensor-shape expectations
- modify a training workflow
- introduce a new major subsystem
- add a new example notebook worth surfacing

## Minimum documentation rule

For any meaningful user-facing change, update:

1. [`README.md`](https://github.com/lseman/foreblocks/blob/main/README) if the landing page should change
2. one relevant page under `docs/`
3. one runnable example or notebook when the change affects workflows

## Authoring guidance

- prefer accurate, minimal runnable examples
- document the public API before internal details
- avoid promising workflows that have not been validated in the repository
- link across pages so documentation behaves like a wiki, not isolated files
- prefer fenced code blocks with an explicit language like `python` or `bash`
- keep configuration docs aligned with `foreblocks/models/config.py` and `foreblocks/training/config.py`

## Local preview

Build the documentation from the repository root using Python:

```bash
python -m pip install -r .github/scripts/docs-requirements.txt
python .github/scripts/compile_docs.py
```

Preview the generated site locally:

```bash
python -m http.server 8080 --directory site
```

Open `http://localhost:8080/docs/`. Re-run the compiler after editing Markdown.
The documentation build does not require Node.js. The separately built frontend
playground under `apps/studio/` has its own dependencies.
