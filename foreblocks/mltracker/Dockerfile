FROM node:20-slim AS dashboard-build

WORKDIR /build/dashboard

COPY foreblocks/mltracker/dashboard_v2/package.json foreblocks/mltracker/dashboard_v2/package-lock.json ./
RUN npm ci

COPY foreblocks/mltracker/dashboard_v2/index.html ./
COPY foreblocks/mltracker/dashboard_v2/tsconfig*.json ./
COPY foreblocks/mltracker/dashboard_v2/vite.config.ts ./
COPY foreblocks/mltracker/dashboard_v2/src ./src
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MLTRACKER_DIR=/data/mltracker \
    PORT=8000

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi>=0.100" \
    "uvicorn[standard]>=0.22" \
    "pydantic>=2.0" \
    "python-multipart>=0.0.9"

COPY foreblocks/__init__.py ./foreblocks/__init__.py
COPY foreblocks/mltracker ./foreblocks/mltracker
COPY --from=dashboard-build /build/dashboard/dist ./foreblocks/mltracker/dashboard_v2/dist

RUN mkdir -p /data/mltracker

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json, urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3); data=json.load(r); raise SystemExit(0 if data.get('status') in ('healthy', 'degraded') else 1)"

CMD ["uvicorn", "foreblocks.mltracker.api:app", "--host", "0.0.0.0", "--port", "8000"]
