from functools import partial
from typing import Optional

import numpy as np
from scipy.stats import uniform

from experiment_design.optimize import get_best_try
from experiment_design.scorers import Scorer, make_default_scorer
from experiment_design.variable import DesignSpace, Variable, VariableCollection


def _create(variables: VariableCollection, sample_size: int) -> np.ndarray:
    """
    Create a design of experiments (DoE) by randomly sampling from passed variables

    :param variables: Determines the dimensions of the resulting sample
    :param sample_size: the number of points to be created
    :return: DoE matrix with shape (len(variables), samples_size)
    """
    doe = uniform(0, 1).rvs((sample_size, len(variables)))
    if not isinstance(variables, DesignSpace):
        variables = DesignSpace(variables)
    return variables.value_of(doe)


def create(
    variables: VariableCollection,
    sample_size: int,
    steps: Optional[int] = None,
    scorer: Optional[Scorer] = None,
) -> np.ndarray:
    """
    Create a design of experiments (DoE) by randomly sampling from the
    variable distributions.

    :param variables: Determines the dimensions of the resulting sample
    :param sample_size: the number of points to be created
    :param scorer: Used to rank the generated DoEs. Specifically, steps
    number of DoEs will be created and the one with the highest score
    will be returned.
    :param steps: Number of DoEs to be created to choose the best from
    :return: DoE matrix with shape (len(variables), samples_size)
    """
    if not isinstance(variables, DesignSpace):
        variables = DesignSpace(variables)

    if steps < 2:
        return _create(variables, sample_size)

    if scorer is None:
        scorer = make_default_scorer(variables, target_correlation=0.0)
    return get_best_try(partial(_create, variables, sample_size), scorer, steps)


def extend(
    old_sample: np.ndarray,
    variables: list[Variable],
    sample_size: int,
    steps: Optional[int] = None,
    scorer: Optional[Scorer] = None,
):
    """
    Extend a design of experiment (DoE) by randomly sampling from the
    variable distributions.

    :param old_sample: Old DoE matrix with shape (len(variables), old_sample_size)
    :param variables: Determines the dimensions of the resulting sample
    :param sample_size: The number of points to be added to the old_sample
    :param scorer: Used to rank the generated DoEs. Specifically, steps
    number of DoEs will be created and the one with the highest score
    will be returned.
    :param steps: Number of DoEs to be created to choose the best from
    :return: Matrix of new points to be added with shape (len(variables), samples_size)
    """
    if steps < 2:
        return _create(variables, sample_size)

    if scorer is None:
        scorer = make_default_scorer(variables, target_correlation=0.0)

    def append_and_score(new_doe: np.ndarray) -> float:
        doe = np.append(old_sample, new_doe)
        return scorer(doe)

    return create(variables, sample_size, steps=steps, scorer=append_and_score)
