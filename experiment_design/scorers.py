from typing import Protocol, Union

import numpy as np
from scipy.spatial.distance import pdist

from experiment_design.variable import Variable


class Scorer(Protocol):
    def __call__(self, doe: np.ndarray) -> float:
        ...


def make_corr_error_scorer(target_correlation: np.ndarray, eps: float = 1e-8) -> Scorer:
    """
    Create a scorer, that computes the maximum absolute correlation error between the samples
    and the target_correlation.

    :param target_correlation: A symmetric matrix with shape (len(variables), len(variables)),
    representing the linear dependency between the dimensions. If a float, all non-diagonal entries
    of the unit matrix will be set to this value.
    :param eps: a small positive value to improve the stability of the log operation

    :return: a scorer that returns the log negative maximum absolute correlation error
    """

    def _scorer(doe: np.ndarray) -> float:
        error = np.max(np.abs(np.corrcoef(doe, rowvar=False) - target_correlation))
        return np.log(error + eps)

    return _scorer


def make_min_pairwise_distance_scorer(max_distance: float = 1.) -> Scorer:
    """
    Create a scorer, that computes the minimum pairwise distance between the samples.
    :param max_distance: Used for scaling the log

    :return: a scorer that returns the log minimum pairwise distance divided by the log max distance
    """
    max_log_distance = np.log(max_distance)

    def _scorer(doe: np.ndarray) -> float:
        min_pdist = np.log(np.min(pdist(doe)))
        return min_pdist - max_log_distance

    return _scorer


def make_default_scorer(variables: list[Variable],
                        target_correlation: Union[np.ndarray, float] = 0.):
    mins = np.array([var.value_of(0.001) for var in variables])
    maxs = np.array([var.value_of(0.999) for var in variables])
    dmax = np.linalg.norm(maxs - mins)
    dist_scorer = make_min_pairwise_distance_scorer(dmax)
    num_var = len(variables)
    if np.isscalar(target_correlation):
        target_correlation = np.eye(num_var) * (1 - target_correlation) + np.ones((num_var, num_var)) * target_correlation
    corr_scorer = make_corr_error_scorer(target_correlation)
    return lambda doe: dist_scorer(doe) + corr_scorer(doe)
