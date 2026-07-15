# MLTracker - Machine Learning Experiment Tracking and Visualization Platform

MLTracker is a comprehensive machine learning experiment tracking, logging, and visualization platform with a web dashboard, Text User Interface (TUI), REST API, and Docker support. It enables teams to track experiments, compare model performance, visualize metrics, and manage ML workflows efficiently.

## Overview

MLTracker provides:

- **Experiment Tracking**: Log parameters, metrics, artifacts, and model checkpoints across training runs
- **Web Dashboard (v2)**: Modern web-based interface for experiment visualization, comparison, and analysis
- **Text User Interface (TUI)**: Terminal-based interface for quick experiment management and monitoring
- **REST API**: Programmatic access to experiment data for integration with other tools
- **Python Client**: Easy-to-use client library for logging experiments from Python code
- **Docker Support**: Containerized deployment with Docker and Docker Compose

## Directory Structure

```
mltracker/
├── api.py                        # REST API implementation (FastAPI/Flask)
│   # Endpoints for:
│   # - Experiment management (create, list, update, delete)
│   # - Metric logging and retrieval
│   # - Parameter tracking
│   # - Artifact management
│   # - User and project management
│
├── client.py                     # Client library for MLTracker API
│   # Client methods for:
│   # - Initializing experiments
│   # - Logging metrics and parameters
│   # - Uploading artifacts
│   # - Querying experiment data
│
├── mltracker.py                  # Core MLTracker functionality
│   # Main tracking engine:
│   # - Experiment lifecycle management
│   # - Metric aggregation and storage
│   # - Parameter tracking
│   # - Artifact handling
│
├── mltracker_client.py           # Client utilities and helpers
│   # Additional client-side utilities
│
├── mltracker_tui.py              # Text User Interface (TUI) implementation
│   # Terminal-based interface for:
│   # - Experiment listing and filtering
│   # - Metric comparison
│   # - Parameter inspection
│   # - Quick experiment management
│
├── dashboard_v2/                 # Web dashboard implementation (v2)
│   # Modern web-based dashboard:
│   # - Experiment visualization
│   # - Metric comparison charts
│   # - Parameter analysis
│   # - Model performance dashboards
│   # - Interactive filtering and sorting
│
├── mltracker_data/               # Data storage for MLTracker
│   # Experiment data, metrics, and artifacts storage
│
├── docker-compose.yml            # Docker Compose configuration for production
├── docker-compose.dev.yml        # Docker Compose configuration for development
├── Dockerfile                    # Docker image build configuration
├── Dockerfile.dev-api            # Development API Docker image
├── DOCKER.md                     # Docker deployment documentation
├── .env                          # Environment configuration (local)
├── .env.example                  # Example environment configuration
├── __init__.py                   # Package initialization
├── README.md                     # This file
└── .ruff_cache/                  # Ruff linter cache

## Core API

### Python Client

```python
import mltracker

# Initialize tracker
tracker = mltracker.Tracker(project="my-project")

# Start experiment
with tracker.experiment(name="baseline_model") as exp:
    # Log parameters
    exp.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    })
    
    # Log metrics during training
    for epoch in range(100):
        # ... training code ...
        exp.log_metrics({
            "loss": train_loss,
            "accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
    
    # Log artifacts
    exp.log_artifact("model.pth", path="/path/to/model.pth")
```

### REST API

The MLTracker REST API provides endpoints for:

- **Experiments**: `GET /api/experiments`, `POST /api/experiments`, `GET /api/experiments/{id}`, `PUT /api/experiments/{id}`, `DELETE /api/experiments/{id}`
- **Metrics**: `POST /api/experiments/{id}/metrics`, `GET /api/experiments/{id}/metrics`
- **Parameters**: `POST /api/experiments/{id}/parameters`, `GET /api/experiments/{id}/parameters`
- **Artifacts**: `POST /api/experiments/{id}/artifacts`, `GET /api/experiments/{id}/artifacts/{artifact_id}`
- **Projects**: `GET /api/projects`, `POST /api/projects`

### TUI Commands

The MLTracker TUI provides interactive commands for:

- Listing experiments with filtering and sorting
- Comparing metrics across multiple experiments
- Inspecting parameters and artifacts
- Viewing training curves and visualizations
- Managing experiments (pause, resume, delete)

## Deployment

### Docker Deployment

MLTracker can be deployed using Docker:

```bash
# Using docker-compose
docker-compose up -d

# Development mode
docker-compose -f docker-compose.dev.yml up -d
```

### Environment Variables

Key environment variables (see `.env.example`):

- `MLTRACKER_DATABASE_URL`: Database connection string
- `MLTRACKER_API_KEY`: API key for authentication
- `MLTRACKER_PROJECT_DIR`: Directory for experiment data storage
- `MLTRACKER_PORT`: Port for the API server
- `MLTRACKER_DASHBOARD_PORT`: Port for the web dashboard

## Key Features

1. **Comprehensive Experiment Tracking**: Parameters, metrics, artifacts, and model checkpoints
2. **Web Dashboard (v2)**: Modern, interactive interface for experiment visualization and comparison
3. **TUI Interface**: Terminal-based tool for quick experiment management without leaving the command line
4. **REST API**: Programmatic access for integration with CI/CD pipelines and other tools
5. **Python Client**: Simple, intuitive API for logging experiments from Python code
6. **Docker Support**: Containerized deployment with Docker and Docker Compose
7. **Data Storage**: Flexible storage for experiment data, metrics, and artifacts
8. **Project Management**: Organize experiments into projects for better management

## Dependencies

- Python 3.8+
- FastAPI or Flask (for REST API)
- SQLite or PostgreSQL (for data storage)
- React or Vue.js (for web dashboard v2)
- Rich or Textual (for TUI implementation)
- Docker and Docker Compose (for containerized deployment)
