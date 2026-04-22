from typing import Any

class TrialPruned(Exception):
    """Exception raised when a trial should be pruned."""
    pass

class Trial:
    """
    A trial object passed to the objective function, allowing it to report
    intermediate results and be pruned early if it performs poorly.
    """
    def __init__(self, config: dict[str, Any], budget: float, bohb_instance):
        self.config = config
        self.budget = float(budget)
        self.bohb = bohb_instance
        self.reports: dict[int, float] = {}
        self._is_pruned = False

    def report(self, step: int, loss: float) -> None:
        """
        Reports an intermediate loss at a given step (e.g., epoch).
        Raises TrialPruned if the trial should be stopped.
        """
        step = int(step)
        loss = float(loss)
        should_prune = self.bohb._should_prune_step(self, step, loss)
        self.reports[step] = loss
        self.bohb._record_trial_report(self, step, loss)

        if should_prune:
            self._is_pruned = True
            raise TrialPruned()

    def should_prune(self) -> bool:
        """
        Returns True if the trial was pruned. Can be checked manually if not relying
        on TrialPruned exceptions.
        """
        return self._is_pruned
