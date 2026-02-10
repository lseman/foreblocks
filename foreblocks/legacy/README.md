# Legacy Modules

This folder contains archived internal modules kept for reference only.

- These files are **not** part of the public API.
- New code should import from the organized packages under:
  - `foreblocks.training`
  - `foreblocks.evaluation`
  - `foreblocks.blocks`
  - `foreblocks.tf`
  - `foreblocks.tuner`

If any production path still depends on a legacy file here, migrate it to the
new package structure.
