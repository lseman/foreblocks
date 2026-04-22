from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AcquisitionStrategy(ABC):
    @abstractmethod
    def score(
        self,
        config: dict[str, Any],
        good_models: dict[str, dict[str, Any]],
        bad_models: dict[str, dict[str, Any]],
    ) -> float:
        pass

