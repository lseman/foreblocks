# MLTracker Dashboard V2

Modern Vite + React + TypeScript implementation for the MLTracker dashboard.

## Why this folder exists

- Serves MLTracker data through the FastAPI backend
- Builds into a static bundle that the backend can mount at `/dashboard`

## Run

```bash
cd foreblocks/mltracker/dashboard_v2
npm install
npm run build
```

Once built, start the MLTracker API as usual and open:

```bash
http://127.0.0.1:8000/dashboard
```

For frontend-only development:

```bash
npm install
npm run dev
```

By default, API calls use `/api`.
The Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

To point to another backend URL:

```bash
VITE_API_BASE=http://127.0.0.1:8000/api npm run dev
```

Important:
- `VITE_API_BASE` must be an API URL, not a filesystem path.
- The tracker data directory is configured on the backend via `MLTRACKER_DIR`.
- The SQLite database is never read directly by the frontend; the dashboard reads through the backend API and therefore the same tracker instance.

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
