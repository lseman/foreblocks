# Web UI — Visual Pipeline Editor

The `webui/` directory contains a browser-based node editor for building and running foreblocks forecasting pipelines without writing code. Nodes represent pipeline stages; edges connect outputs to inputs. Execution happens on a local FastAPI backend that streams progress back to the browser via WebSocket.

## Starting the server

```bash
cd webui
npm install         # first time only
npm run build       # compile the React frontend into dist/
python server.py    # starts on http://localhost:8000
```

Open `http://localhost:8000` in a browser. The static frontend is served from `dist/`.

Dependencies:
```
pip install fastapi uvicorn torch matplotlib
```

## Architecture

```
Browser (React + ReactFlow)
  ↓  POST /execute  (generated Python code + workflow graph)
FastAPI server  →  ThreadPoolExecutor worker
  ↓  WebSocket /ws/{task_id}  (live events)
Browser receives: logs, progress, node results, artifacts
```

The frontend generates Python code from the node graph and sends it to the backend. The backend executes the code in a thread-pool worker and pushes structured events back to all connected WebSocket clients for that task.

## REST API

### `POST /execute`

Submit a pipeline for execution.

```json
{
  "code": "# generated Python code\ntrainer = ...",
  "workflow": {"nodes": [...], "connections": [...]},
  "sync": false,
  "timeout_sec": 1.5
}
```

- `sync=false` (default): returns immediately with a `task_id`; poll or subscribe via WebSocket for results
- `sync=true`: waits up to `timeout_sec` and returns inline results if finished in time

Response:
```json
{"success": true, "task_id": "uuid4", "results": null}
```

### `GET /status/{task_id}`

Poll task state.

```json
{
  "task_id": "...",
  "state": "running",
  "progress": 0.55,
  "message": "Generating plot",
  "results": null
}
```

States: `queued` → `running` → `success` | `error`

### `GET /logs/{task_id}`

Retrieve timestamped execution logs.

```json
{"task_id": "...", "logs": ["[10:42:01] ⏳ Starting execution", ...]}
```

### `GET /artifact/{task_id}/{name}`

Download a saved artifact file (e.g., a PNG plot saved by user code).

## WebSocket events

Connect to `ws://localhost:8000/ws/{task_id}` to receive live updates.

On connect, the server sends a `snapshot` event with current state and the last 50 log lines. Subsequent events:

| Event type | Fields | Meaning |
|---|---|---|
| `log` | `line` | New log line from worker |
| `state` | `state`, `progress`, `message` | Progress update |
| `node_result` | `node_id`, `result` | A node has produced output |
| `artifact` | `name` | A file artifact was saved |
| `done` | `success`, `results` | Execution finished |
| `error` | `message` | Task not found or server error |

### Result types

The `result` field in `node_result` events has a `type` discriminator:

| Type | Data format |
|---|---|
| `plot` | Base64-encoded PNG |
| `image` | Base64-encoded binary image |
| `metrics` | `{key: float}` dict |
| `table` | `{columns: [...], rows: [[...]]}` |
| `json` | Any JSON-serializable value |
| `array` | Flat or nested list |
| `text` | Plain string fallback |

## `report` object in user code

The worker injects a `report` object into the execution namespace. Use it to push structured results back to the UI from within your pipeline code:

```python
# These are available without import inside the executed code:
# report, torch, numpy (as np)

report.log("Loading dataset...")
report.progress(0.2, "Data loaded")

metrics = {"mae": 0.042, "rmse": 0.071}
report.set_result("output_1", "metrics", metrics)

fig = trainer.plot_prediction(X_val, Y_val, show=False)
report.set_result("plot_node", "plot", fig)   # matplotlib figure → base64 PNG

report.artifact("forecast.png", fig)           # also saves file to artifacts/
```

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
