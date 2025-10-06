"""
MLTracker Client
Python client for interacting with MLTracker API
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class MLTrackerClient:
    """Client for MLTracker API"""
    
    def __init__(self, tracking_uri: str = "http://localhost:8000"):
        self.base_url = tracking_uri.rstrip("/")
        self._active_run_id = None
    
    def _request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request"""
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, **kwargs)
        
        if response.status_code >= 400:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        return response.json() if response.content else None
    
    # Experiment methods
    def create_experiment(self, name: str) -> Dict:
        """Create a new experiment"""
        return self._request("POST", "/api/experiments", json={"name": name})
    
    def get_experiment(self, name: str) -> Dict:
        """Get experiment by name"""
        return self._request("GET", f"/api/experiments/{name}")
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments"""
        return self._request("GET", "/api/experiments")
    
    # Run methods
    def start_run(self, experiment_name: str = "default", run_name: Optional[str] = None) -> str:
        """Start a new run"""
        response = self._request("POST", "/api/runs", json={
            "experiment_name": experiment_name,
            "run_name": run_name
        })
        self._active_run_id = response["run_id"]
        return self._active_run_id
    
    def end_run(self, run_id: Optional[str] = None, status: str = "FINISHED"):
        """End a run"""
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run to end")
        
        self._request("PUT", f"/api/runs/{rid}/end", json={"status": status})
        
        if rid == self._active_run_id:
            self._active_run_id = None
    
    def get_run(self, run_id: str) -> Dict:
        """Get run details"""
        return self._request("GET", f"/api/runs/{run_id}")
    
    def search_runs(self, experiment_name: Optional[str] = None) -> List[Dict]:
        """Search for runs"""
        params = {"experiment_name": experiment_name} if experiment_name else {}
        result = self._request("GET", "/api/runs", params=params)
        return result["runs"]
    
    # Logging methods
    def log_param(self, key: str, value: Any, run_id: Optional[str] = None):
        """Log a parameter"""
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")
        
        self._request("POST", f"/api/runs/{rid}/params", json={
            "key": key,
            "value": value
        })
    
    def log_params(self, params: Dict[str, Any], run_id: Optional[str] = None):
        """Log multiple parameters"""
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")
        
        self._request("POST", f"/api/runs/{rid}/params/batch", json=params)
    
    def log_metric(self, key: str, value: float, step: int = 0, run_id: Optional[str] = None):
        """Log a metric"""
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")
        
        self._request("POST", f"/api/runs/{rid}/metrics", json={
            "key": key,
            "value": value,
            "step": step
        })
    
    def log_metrics(self, metrics: Dict[str, float], step: int = 0, run_id: Optional[str] = None):
        """Log multiple metrics"""
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")
        
        self._request("POST", f"/api/runs/{rid}/metrics/batch", 
                     json=metrics, params={"step": step})
    
    def set_tag(self, key: str, value: str, run_id: Optional[str] = None):
        """Set a tag"""
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")
        
        self._request("POST", f"/api/runs/{rid}/tags", json={
            "key": key,
            "value": value
        })
    
    # Artifact methods
    def log_artifact(self, local_path: str, artifact_path: str = "", run_id: Optional[str] = None):
        """Upload an artifact"""
        rid = run_id or self._active_run_id
        if not rid:
            raise ValueError("No active run. Call start_run() first.")
        
        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        
        with open(local_file, 'rb') as f:
            files = {'file': (local_file.name, f)}
            params = {'artifact_path': artifact_path} if artifact_path else {}
            
            response = requests.post(
                f"{self.base_url}/api/runs/{rid}/artifacts",
                files=files,
                params=params
            )
            
            if response.status_code >= 400:
                raise Exception(f"Upload failed: {response.text}")
            
            return response.json()
    
    def list_artifacts(self, run_id: str) -> List[Dict]:
        """List artifacts for a run"""
        result = self._request("GET", f"/api/runs/{run_id}/artifacts")
        return result["artifacts"]
    
    def download_artifact(self, run_id: str, artifact_path: str, local_path: str):
        """Download an artifact"""
        url = f"{self.base_url}/api/runs/{run_id}/artifacts/{artifact_path}"
        response = requests.get(url, stream=True)
        
        if response.status_code >= 400:
            raise Exception(f"Download failed: {response.text}")
        
        output_path = Path(local_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Analysis methods
    def get_metric_history(self, run_id: str, metric_key: Optional[str] = None) -> Dict:
        """Get metric history"""
        params = {"metric_key": metric_key} if metric_key else {}
        result = self._request("GET", f"/api/runs/{run_id}/metrics/history", params=params)
        return result["metrics"]
    
    def compare_runs(self, run_ids: List[str]) -> Dict:
        """Compare multiple runs"""
        return self._request("POST", "/api/runs/compare", json=run_ids)
    
    def health_check(self) -> Dict:
        """Check API health"""
        return self._request("GET", "/api/health")


@contextmanager
def start_run(client: MLTrackerClient, experiment_name: str = "default", 
              run_name: Optional[str] = None):
    """Context manager for runs"""
    run_id = client.start_run(experiment_name, run_name)
    try:
        yield run_id
        client.end_run(run_id, "FINISHED")
    except Exception as e:
        client.end_run(run_id, "FAILED")
        raise e


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = MLTrackerClient("http://localhost:8000")
    
    # Check health
    print("Health check:", client.health_check())
    
    # Example 1: Using context manager
    with start_run(client, "sklearn_experiments", "random_forest_v1"):
        # Log parameters
        client.log_params({
            "model": "RandomForest",
            "n_estimators": 100,
            "max_depth": 10
        })
        
        # Simulate training
        for epoch in range(5):
            client.log_metrics({
                "train_acc": 0.7 + epoch * 0.05,
                "val_acc": 0.65 + epoch * 0.045
            }, step=epoch)
        
        # Log tag
        client.set_tag("version", "v1.0")
    
    # Example 2: Manual run management
    run_id = client.start_run("deep_learning", "cnn_baseline")
    
    try:
        client.log_param("architecture", "CNN")
        client.log_param("layers", 5)
        
        client.log_metric("loss", 0.45)
        client.log_metric("accuracy", 0.89)
        
        # Upload artifact (if you have a file)
        # client.log_artifact("model.pkl", "models")
        
        client.end_run(run_id, "FINISHED")
    except Exception as e:
        client.end_run(run_id, "FAILED")
        raise e
    
    # Search and compare runs
    runs = client.search_runs("sklearn_experiments")
    print(f"\nFound {len(runs)} runs in sklearn_experiments")
    
    # Compare runs
    if len(runs) >= 2:
        comparison = client.compare_runs([runs[0]["run_id"], runs[1]["run_id"]])
        print("\nComparison:")
        print(f"Metrics: {comparison['metric_keys']}")
        print(f"Params: {comparison['param_keys']}")