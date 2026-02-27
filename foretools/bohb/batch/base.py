from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class BatchSelector(ABC):
    @abstractmethod
    def select(
        self, candidates: List[Dict[str, Any]], scores: np.ndarray, n: int
    ) -> List[int]:
        pass

