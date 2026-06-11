---
title: Web UI
description: Visual pipeline editor for building forecasting workflows.
editLink: true
---


[[toc]]
# Web UI — Visual Pipeline Editor

The `webui/` directory contains a browser-based node editor for building and running foreblocks forecasting pipelines without writing code. Nodes represent pipeline stages; edges connect outputs to inputs. Execution happens on a local FastAPI backend that streams progress back to the browser via WebSocket.

## Starting the server

```bash
cd webui
npm install         # first time only
npm run build       # compile the React frontend into dist/
python server.py    # starts on http://localhost:8000
```text
pip install fastapi uvicorn torch matplotlib
```text
Browser (React + ReactFlow)
  ↓  POST /execute  (generated Python code + workflow graph)
FastAPI server  →  ThreadPoolExecutor worker
  ↓  WebSocket /ws/{task_id}  (live events)
Browser receives: logs, progress, node results, artifacts
```text

- `sync=false` (default): returns immediately with a `task_id`; poll or subscribe via WebSocket for results
- `sync=true`: waits up to `timeout_sec` and returns inline results if finished in time

Response:
```json
{"success": true, "task_id": "uuid4", "results": null}
```json

States: `queued` → `running` → `success` | `error`

### `GET /logs/{task_id}`

Retrieve timestamped execution logs.

```json
{"task_id": "...", "logs": ["[10:42:01] ⏳ Starting execution", ...]}
```toml

`set_result(node_id, result_type, data)` — `node_id` should match the node's ID in the workflow graph. `result_type` is one of the type strings from the table above.

## Auto-detection fallback

If user code does not call `report.set_result()`, the backend attempts automatic result extraction by looking for these variables in the execution namespace:

| Variable | Action |
|---|---|
| `trainer`, `X_val`, `Y_val` | Calls `trainer.metrics(X_val, Y_val)` and `trainer.plot_prediction(...)` |
| `metrics` | Attaches as metrics result on the output node |
| `time_series`, `train_size` | Passed to `plot_prediction` as `full_series` / `offset` |

## Environment injected into user code

| Name | Value |
|---|---|
| `torch` | `import torch` |
| `report` | `ReportBridge` instance for the current task |
| `__name__` | `"__main__"` |

`numpy`, `matplotlib.pyplot`, and `matplotlib` (Agg backend) are imported inside the worker before `exec()` but are not injected into the namespace by default — import them explicitly in your code if needed.
