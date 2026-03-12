# MLTracker Dashboard V2

Modern Vite + React + TypeScript implementation for the MLTracker dashboard.

## Why this folder exists

- Keeps legacy dashboard untouched at `foreblocks/mltracker/dashboard`
- Provides a cleaner architecture for incremental migration
- Offers a fast dev loop and typed API boundary

## Run

```bash
cd foreblocks/mltracker/dashboard_v2
npm install
npm run dev
```

By default, API calls use `VITE_API_BASE=/api`.

To point to another backend URL:

```bash
VITE_API_BASE=http://127.0.0.1:8000/api npm run dev
```

Important:
- `VITE_API_BASE` must be an API URL, not a filesystem path.
- The tracker data directory is configured on the backend via `MLTRACKER_DIR`.

## Migrated legacy functionality

- Experiment list with search
- Run listing with server-side query and local filters
- Sort by start time or duration
- Group by none, status, or day
- Status chips with counts
- Tag filter and numeric metric range filter
- Run selection (`select visible`, `clear selected`)
- Run comparison panel (`/runs/compare`)
- Compare metric history multi-line chart (`/runs/:id/metrics/history`)
- Sweep scatter plot for numeric params/metrics
- Run detail drawer (info, params, tags, metric keys)
- Artifact listing with download links
- Delete run action
- Dark/light theme with persistence
- View preferences persisted per experiment
- Saved views and command palette (`Ctrl/Cmd+K`)
- Shareable URL state sync for current view
