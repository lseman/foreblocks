# Documentation Workflow

This repository uses in-repo, versioned documentation instead of a separate GitHub Wiki.

## Why this approach

- docs stay aligned with the code
- documentation changes can be reviewed in pull requests
- examples and notebooks can evolve alongside implementation
- contributors do not need a separate editing surface

## Structure

The documentation system is organized as:

- [`docs/index.md`](https://github.com/lseman/foreblocks/blob/main/docs/index.md): docs home
- `docs/tutorials/`: runnable onboarding and practical workflows
- `docs/architecture/`: internal structure and design pages
- `docs/reference/`: API and repository reference
- root-level guide pages in `docs/`: subsystem-focused guides
- [`web/index.html`](https://github.com/lseman/foreblocks/blob/main/web/index.html): static landing page source for the site root
- [`mkdocs.yml`](https://github.com/lseman/foreblocks/blob/main/mkdocs.yml): navigation and site structure for the `/docs/` site

## When to update docs

Update documentation whenever you:

- add or remove public exports
- change constructor contracts or tensor-shape expectations
- modify a training workflow
- introduce a new major subsystem
- add a new example notebook worth surfacing

## Minimum documentation rule

For any meaningful user-facing change, update:

1. [`README.md`](https://github.com/lseman/foreblocks/blob/main/README.md) if the landing page should change
2. one relevant page under `docs/`
3. one runnable example or notebook when the change affects workflows

## Authoring guidance

- prefer accurate, minimal runnable examples
- document the public API before internal details
- avoid promising workflows that have not been validated in the repository
- link across pages so documentation behaves like a wiki, not isolated files
- prefer fenced code blocks with an explicit language like `python` or `bash`
- keep configuration docs aligned with `foreblocks/aux/config.py`

## Local preview

This repository now uses `mkdocs-material` with `pymdownx` Markdown extensions. Install it first:

```bash
pip install mkdocs-material
```

Then serve the wiki locally:

```bash
mkdocs serve
```

Or build a static site:

```bash
mkdocs build
```

## Related pages

- [Home](../index.md)
- [Overview](../overview.md)
- [Repository Map](../reference/repository-map.md)
