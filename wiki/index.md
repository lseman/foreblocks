# foreBlocks Docs

Welcome to the in-repo documentation system for `foreblocks` and its companion tooling in `foretools`.

This documentation is organized like a versioned wiki:

- tutorials for runnable starting points
- guides for major subsystems
- architecture pages for internal structure
- reference pages for stable entry points and configuration
- contributor docs for maintaining the documentation itself

## Start Here

If you are new to the project, read these in order:

1. [Overview](overview.md)
2. [Getting Started](getting-started.md)
3. [Public API Reference](reference/public-api.md)

## Tutorials

- [Getting Started](getting-started.md)
- [Train a Direct Model](tutorials/train-direct-model.md)
- [Generate Synthetic Series](tutorials/generate-synthetic-series.md)

## Guides

- [Preprocessor Guide](preprocessor.md)
- [Custom Blocks Guide](custom_blocks.md)
- [Transformer Guide](transformer.md)
- [MoE Guide](moe.md)
- [DARTS Guide](darts.md)

## Architecture

- [System Overview](architecture/system-overview.md)
- [Forecasting Pipeline](architecture/forecasting-pipeline.md)

## Reference

- [Public API](reference/public-api.md)
- [Configuration](reference/configuration.md)
- [Repository Map](reference/repository-map.md)

## Contributing

- [Documentation Workflow](contributing/docs-workflow.md)

## Notes

- The documentation is intended to stay versioned with the codebase.
- The top-level `README.md` remains the landing page for GitHub visitors.
- The canonical docs source lives in `wiki/`, not in a separate GitHub Wiki repository.
- The published docs URL space is `/docs/`, while the custom landing page remains at site root.
