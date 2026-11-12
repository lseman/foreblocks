"""Basic outlier processing must not import the neural model stack."""

import subprocess
import sys
import textwrap


def test_classical_outliers_work_with_model_imports_blocked():
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        class BlockModels(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "foreblocks.models" or fullname.startswith("foreblocks.models."):
                    raise AssertionError(f"Unexpected neural model import: {fullname}")

        sys.meta_path.insert(0, BlockModels())
        import numpy as np
        from foreblocks.processing.core import outlier

        x = np.array([0., .1, .2, .3, .4, 15., 16.])
        result = outlier._remove_outliers(x, "hbos", .2)
        assert np.isnan(result[-2:]).all()
        assert "TranAD" not in vars(outlier)
        assert "TranADDetector" not in vars(outlier)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
