from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class AcquisitionStrategy(ABC):
    @abstractmethod
    def score(
        self,
        config: Dict[str, Any],
        good_models: Dict[str, Dict[str, Any]],
        bad_models: Dict[str, Dict[str, Any]],
    ) -> float:
        pass

