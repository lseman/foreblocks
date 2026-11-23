import subprocess
import sys
from textwrap import dedent


def test_cpu_training_without_plotting_dependencies():
    script = dedent("""
        import sys
        sys.modules["matplotlib"] = None

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from foreblocks import Trainer, TrainingConfig

        model = torch.nn.Linear(3, 1)
        trainer = Trainer(model, config=TrainingConfig(use_amp=False),
                          auto_track=False, device="cpu")
        batch = TensorDataset(torch.ones(2, 3), torch.zeros(2, 1))
        trainer.train_epoch(DataLoader(batch, batch_size=2))
        assert "foreblocks.evaluation.visualization" not in sys.modules
    """)
    subprocess.run([sys.executable, "-c", script], check=True)
