import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ensure_package(name: str, path: pathlib.Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: pathlib.Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("foretools", ROOT / "foretools")
_ensure_package("foretools.emd_like", ROOT / "foretools" / "emd_like")
_load_module("foretools.emd_like.common", ROOT / "foretools" / "emd_like" / "common.py")
core_module = _load_module("foretools.emd_like.core", ROOT / "foretools" / "emd_like" / "core.py")
VMDCore = core_module.VMDCore


def _identity_smoother(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _build_demodulated_quadratures(
    if_true: np.ndarray, if_prev: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the exact quadratures of a unit-amplitude chirp under the current
    VNCMD convention:

        x[n] = u[n] cos(phi_prev[n]) + v[n] sin(phi_prev[n])
             = Re[(u[n] - j v[n]) exp(j phi_prev[n])]

    For a true phase phi_true, the envelope phase mismatch is
    delta = phi_true - phi_prev, which corresponds to:

        u = cos(delta), v = -sin(delta)
    """
    phase_true = 2.0 * np.pi * np.cumsum(np.asarray(if_true, dtype=np.float64))
    phase_prev = 2.0 * np.pi * np.cumsum(np.asarray(if_prev, dtype=np.float64))
    phase_true -= phase_true[:1]
    phase_prev -= phase_prev[:1]
    delta = phase_true - phase_prev
    u_quad = np.cos(delta)[None, :]
    v_quad = (-np.sin(delta))[None, :]
    return u_quad, v_quad


class TestVNCMDIFUpdate(unittest.TestCase):
    def setUp(self) -> None:
        self.core = VMDCore.__new__(VMDCore)

    def test_update_decreases_overestimated_linear_chirp_track(self) -> None:
        n = 512
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        if_true = 0.08 + 0.10 * t
        if_prev = np.clip(if_true + 0.03, 0.0, 0.5)
        u_quad, v_quad = _build_demodulated_quadratures(if_true, if_prev)

        if_next = self.core._vncmd_update_if_tracks(
            if_prev=if_prev[None, :],
            u_quad=u_quad,
            v_quad=v_quad,
            if_smoother=_identity_smoother,
            if_center_smooth=0.0,
            if_step=1.0,
            DC=0,
        )[0]

        inner = slice(8, -8)
        prev_err = np.mean(np.abs(if_prev[inner] - if_true[inner]))
        next_err = np.mean(np.abs(if_next[inner] - if_true[inner]))

        self.assertLess(np.mean(if_next[inner] - if_prev[inner]), 0.0)
        self.assertLess(next_err, prev_err)

    def test_update_increases_underestimated_linear_chirp_track(self) -> None:
        n = 512
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        if_true = 0.08 + 0.10 * t
        if_prev = np.clip(if_true - 0.03, 0.0, 0.5)
        u_quad, v_quad = _build_demodulated_quadratures(if_true, if_prev)

        if_next = self.core._vncmd_update_if_tracks(
            if_prev=if_prev[None, :],
            u_quad=u_quad,
            v_quad=v_quad,
            if_smoother=_identity_smoother,
            if_center_smooth=0.0,
            if_step=1.0,
            DC=0,
        )[0]

        inner = slice(8, -8)
        prev_err = np.mean(np.abs(if_prev[inner] - if_true[inner]))
        next_err = np.mean(np.abs(if_next[inner] - if_true[inner]))

        self.assertGreater(np.mean(if_next[inner] - if_prev[inner]), 0.0)
        self.assertLess(next_err, prev_err)


if __name__ == "__main__":
    unittest.main()
