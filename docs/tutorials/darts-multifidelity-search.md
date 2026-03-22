# Run A DARTS Search

This tutorial shows the intended end-to-end DARTS workflow in ForeBlocks: configure a search trainer, run a small multi-fidelity search, inspect the promoted candidates, and optionally analyze the final run.

## Install

Core DARTS workflow, including the analyzer:

```bash
pip install "foreblocks[darts]"
```

## Step 1: create the trainer

```python
from foreblocks.darts import DARTSTrainer

trainer = DARTSTrainer(
    input_dim=6,
    hidden_dims=[32, 64, 128],
    forecast_horizon=12,
    seq_length=48,
    device="auto",
)
```

At this point you have a search controller, not just a single model. It knows how to generate candidates, train searched models, derive discrete architectures, and retrain the best one.

## Step 2: run a small multi-fidelity search

```python
results = trainer.multi_fidelity_search(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_candidates=12,
    search_epochs=8,
    final_epochs=40,
    max_samples=32,
    top_k=4,
    use_amp=False,
)
```

Recommended first-run strategy:

- keep `num_candidates` small
- keep `search_epochs` small
- disable AMP until the loop is stable
- only scale up after the result structure looks correct

## Step 3: inspect the result dictionary

```python
print(results.keys())
print(results["final_results"]["final_metrics"])
print(len(results["candidates"]), len(results["top_candidates"]))
```

The most useful keys are:

- `final_model`: retrained fixed model
- `best_candidate`: the promoted/search-trained winner
- `final_results`: metrics and training information
- `trained_candidates`: per-candidate search artifacts

## Step 4: save the winning discrete model

```python
trainer.save_best_model("best_darts_model.pth")
```

This saves the best retrained final model together with the recorded metrics and search configuration.

## Step 5: inspect search behavior directly

If you want to debug the search space before a full run, call the intermediate APIs on a single candidate.

### Zero-cost metrics

```python
metrics = trainer.evaluate_zero_cost_metrics(
    model=candidate_model,
    dataloader=val_loader,
    max_samples=32,
    num_batches=1,
    fast_mode=True,
)
```

### Bilevel search for one candidate

```python
search_run = trainer.train_darts_model(
    model=candidate_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=15,
    arch_learning_rate=3e-3,
    model_learning_rate=1e-3,
)
```

### Convert to a fixed architecture

```python
fixed_model = trainer.derive_final_architecture(search_run["model"])
```

### Retrain that fixed model

```python
final_run = trainer.train_final_model(
    model=fixed_model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=50,
)
```

## Optional: analyze the final search result

With `foreblocks[darts]` installed:

```python
from foreblocks.darts import StreamlinedDARTSAnalyzer

analyzer = StreamlinedDARTSAnalyzer(results)
print(analyzer.analysis_df.head())
```

Use this when you want:

- architectural feature summaries
- simple statistical inspection of promoted candidates
- plots that help explain why some candidates won

## Reading the result like a practitioner

Focus on these questions:

1. Did zero-cost ranking surface plausible candidates?
2. Did the promoted models improve after short DARTS training?
3. Did the final retrained model preserve that advantage?
4. Did the search collapse onto one family too early?

If the answer to any of those is no, tighten the search space before you increase the budget.

## Related pages

- [DARTS Guide](../darts.md)
- [DARTS Search Pipeline](../architecture/darts-pipeline.md)
- [Troubleshooting](../troubleshooting.md)
