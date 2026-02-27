
import random
import time
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../foreblocks/mltracker")))
from mltracker import MLTracker

def generate_data():
    tracker = MLTracker("./mltracker_data")
    
    experiments = ["classification_v1", "reg_detection", "nlp_finetune"]
    
    for exp in experiments:
        print(f"Generating runs for {exp}...")
        for i in range(5):
            run_name = f"run_v{i}_{random.randint(1000,9999)}"
            with tracker.run(exp, run_name) as run_id:
                # Log Params
                tracker.log_params({
                    "lr": random.choice([1e-3, 3e-4, 1e-4]),
                    "batch_size": random.choice([32, 64, 128]),
                    "optimizer": "adamw",
                    "dropout": random.uniform(0.1, 0.5)
                })
                
                # Log Metrics
                for step in range(50):
                    loss = 2.0 * math.exp(-0.1 * step) + random.uniform(0, 0.1)
                    acc = 0.5 + 0.4 * (1 - math.exp(-0.05 * step)) + random.uniform(0, 0.05)
                    tracker.log_metrics({"train_loss": loss, "val_acc": acc}, step=step)
                    # time.sleep(0.01) # fast generation
                
                # Log Tag
                tracker.set_tag("user", "seman")
                tracker.set_tag("mode", "test")
                
                # Log Artifact (dummy)
                tracker.log_bytes(b"some model bytes", "model.pkl", "models")
                
    print("Done generating mock data.")

if __name__ == "__main__":
    generate_data()
