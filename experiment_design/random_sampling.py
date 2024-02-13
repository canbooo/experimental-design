from typing import Optional, Union, Callable

import numpy as np
from scipy.stats import uniform

from experiment_design.scorers import Scorer, make_default_scorer
from experiment_design.variable import Variable


def _create(variables: list[Variable], sample_size: int) -> np.ndarray:
    """
    Create a design of experiments (DoE) by randomly sampling from passed variables

    :param variables: Determines the dimensions of the resulting sample
    :param sample_size: the number of points to be created
    :return: DoE matrix with shape (len(variables), samples_size)
    """
    samples = uniform(0, 1).rvs((sample_size, len(variables)))
    for i_dim, variable in enumerate(variables):
        samples[:, i_dim] = variable.value_of(samples[:, i_dim])
    return samples


def create(variables: list[Variable], sample_size: int,
           steps: Optional[int] = None,
           scorer: Optional[Scorer] = None
           ) -> np.ndarray:
    if steps < 2:
        return _create(variables, sample_size)

    if scorer is None:
        scorer = make_default_scorer(variables, target_correlation=0.)

    best_score, best_doe = -np.inf, None
    for _ in range(steps):
        doe = _create(variables, sample_size)
        score = scorer(doe)
        if score > best_score:
            best_doe = doe
            best_score = score
    return best_doe


def extend(old_sample: np.ndarray, variables: list[Variable],
           sample_size: int, steps: Optional[int] = None,
           scorer: Optional[Scorer] = None):
    if steps < 2:
        return _create(variables, sample_size)

    if scorer is None:
        scorer = make_default_scorer(variables, target_correlation=0.)
    def append_and_score(new_doe: np.ndarray) -> float:
        doe = np.append(old_sample, new_doe)
        return scorer(doe)

    return create(variables, sample_size, steps=steps, scorer=append_and_score)
