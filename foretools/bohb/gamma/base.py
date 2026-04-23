from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class GammaStrategy(ABC):
    @abstractmethod
    def n_good(self, n_obs: int) -> int:
        pass
