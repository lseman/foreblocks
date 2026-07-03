# MLTracker Docker Compose

Run the MLTracker API and dashboard from a folder that contains `mltracker.db`
and the `artifacts/` directory.

```bash
cd foreblocks/mltracker
cp .env.example .env
docker compose up --build
```

Open:

```text
http://localhost:8000/dashboard
```

To serve a different tracker folder:

```bash
MLTRACKER_HOST_DIR=/absolute/path/to/mltracker_data docker compose up --build
```

The container mounts that folder at `/data/mltracker` and sets
`MLTRACKER_DIR=/data/mltracker`, so the FastAPI app reads and writes SQLite and
artifacts directly from the mounted volume.

Useful endpoints:

- `GET /api/health`
- `GET /api/experiments`
- `GET /api/runs?experiment_name=default&include=full`
- `GET /api/overview?experiment_name=default`
