
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import shutil

# Ensure we can import foreblocks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "foreblocks")))

from foreblocks.training.trainer import Trainer
from foreblocks.mltracker import MLTracker

def verify_integration():
    # Setup
    if os.path.exists("test_mltracker_data"):
        shutil.rmtree("test_mltracker_data")
    
    tracker = MLTracker("./test_mltracker_data")
    
    # Dummy Data & Model
    X = torch.randn(10, 5)
    y = torch.randn(10, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2)
    
    model = nn.Linear(5, 1)
    
    # Init Trainer with tracker
    trainer = Trainer(
        model=model,
        config={"num_epochs": 2, "experiment_name": "trainer_test"},
        mltracker=tracker
    )
    
    # Train
    print("Starting training...")
    trainer.train(loader)
    print("Training finished.")
    
    # Verify
    runs = tracker.search_runs("trainer_test")
    print(f"Found {len(runs)} runs.")
    
    if len(runs) == 0:
        print("FAIL: No runs found.")
        return

    run = runs[0]
    print(f"Run ID: {run['run_id']}")
    print(f"Params keys: {list(run['params'].keys())}")
    print(f"Metrics keys: {list(run['metrics'].keys())}")
    
    if "train_loss" in run['metrics']:
        print("SUCCESS: train_loss logged.")
    else:
        print("FAIL: train_loss not logged.")

if __name__ == "__main__":
    verify_integration()
