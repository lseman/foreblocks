# projects/

Standalone sub-projects that live in this repository but are **not** part of
the `foreblocks` distribution — they have no entry in `pyproject.toml` and are
not importable as `foreblocks.*`. Each is self-contained with its own README
and dependencies.

| Project | What it is |
| --- | --- |
| [`tree/`](tree/) (ForeTree) | A standalone C++23 tree-model library (histogram splitting, CUDA) with pybind bindings and its own CMake build. Not a Python package. |
| [`scheduling/`](scheduling/) | Neural schedulers (RL/GNN) for the Offline Nanosatellite Task Scheduling (ONTS) problem. Unrelated to time-series forecasting. |
