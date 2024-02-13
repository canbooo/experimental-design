from typing import Callable
import numpy as np

from experiment_design.scorers import Scorer


def get_best_try(creator: Callable[[], np.ndarray], scorer: Scorer, steps: int) -> np.ndarray:
    steps = max(1, steps)
    best_score, best_doe = -np.inf, None
    for _ in range(steps):
        doe = creator()
        score = scorer(doe)
        if score > best_score:
            best_doe = doe
            best_score = score
    return best_doe


def simulated_annealing():
    raise NotImplementedError