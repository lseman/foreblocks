from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np


class BatchSelector(ABC):
    @abstractmethod
    def select(
        self, candidates: list[dict[str, Any]], scores: np.ndarray, n: int
    ) -> list[int]:
        pass

