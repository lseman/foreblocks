# Architecture

The canonical, versioned description of this repository's structure lives in the
docs site, not here — see:

- [Repository Map](docs/reference/repository-map.md) — top-level areas, `src/foreblocks/`
  subpackage table, the tiered layout convention, and `foretools`/`projects/`.
- [System Overview](docs/architecture/system-overview.md) — subsystem roles, dependency
  graph, and the public API boundary.
- [Public API](docs/reference/public-api.md) — the stable top-level import surface and
  where each export resolves to.
- [Reorg Migration Map](docs/reference/reorg-migration.md) — old → new import path history.

If you move a package, update those pages in the same change (they are Markdown files,
edit them directly — no site build needed to keep them accurate).

## Packages under `src/`

`src/` holds `foreblocks` (the main library), `darts` (standalone NAS), `foretools`
(companion tooling), and `mltracker` (experiment tracking) — all four are packaged
in `pyproject.toml`. Two standalone, non-packaged sub-projects live at `projects/`
(`projects/tree`, a C++/CUDA library; `projects/scheduling`, an unrelated RL side-project)
— see [`projects/README.md`](projects/README.md).
