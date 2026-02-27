
import time
import numpy as np
from bohb import BOHB, torch_mlp_objective

def benchmark():
    config_space = {
        "lr": ("float", (1e-5, 1e-1, "log")),
        "batch_size": ("choice", [16, 32, 64, 128, 256]),
        "dropout": ("float", (0.0, 0.5)),
        "hidden": ("int", (16, 256)),
    }

    print("Starting benchmark (ATPE enabled with blocking)...")
    start_time = time.time()
    
    # Run a short BOHB run with ATPE enabled + params
    bohb = BOHB(
        config_space=config_space,
        evaluate_fn=torch_mlp_objective,
        min_budget=1,
        max_budget=27, 
        eta=3,
        n_iterations=2,
        verbose=False,
        seed=42,
        tpe_overrides={
            "gamma": 0.15,
            "n_startup_trials": 5,
            "n_ei_candidates": 24,
            "atpe": True,
            "atpe_params": {"filter_type": "zscore", "filter_threshold": 2.0},
            "blocking_threshold": 0.7,
        },  # aggressive blocking for test
    )

    best_cfg, best_loss = bohb.run()
    end_time = time.time()
    
    print(f"Benchmark finished in {end_time - start_time:.2f} seconds")
    print(f"Best Loss: {best_loss}")
    print(f"Best Config: {best_cfg}")
    return best_loss, end_time - start_time

if __name__ == "__main__":
    benchmark()
