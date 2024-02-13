from typing import Protocol, Union, Optional

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
    representing the linear dependency between the dimensions.
    :param eps: a small positive value to improve the stability of the log operation
    :return: a scorer that returns the log negative maximum absolute correlation error
    """
    if np.max(np.abs(target_correlation)) > 1:
        raise ValueError('Correlations should be in the interval [-1,1].')

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
    """
    Create a default scorer, which creates a scorer from the sum of minimum
    pairwise distance and maximum correlation error scorers. See those scorers
    for more details.

    :param variables: variables of the doe. Used to determine the dimension number and
    bounds of the space
    :param target_correlation: A symmetric matrix with shape (len(variables), len(variables)),
    representing the linear dependency between the dimensions. If a float, all non-diagonal entries
    of the unit matrix will be set to this value.
    :return: scorer as a sum of the minimum pairwise distance and maximum correlation error scorers.
    """
    dmax = get_max_distance(variables)
    dist_scorer = make_min_pairwise_distance_scorer(dmax)
    target_correlation = get_correlation_matrix(target_correlation=target_correlation,
                                                num_variables=len(variables))
    corr_scorer = make_corr_error_scorer(target_correlation)
    return lambda doe: dist_scorer(doe) + corr_scorer(doe)


def get_max_distance(variables: list[Variable]) -> float:
    lower, upper = [], []
    for var in variables:
        if hasattr(var, "lower_bound") and var.lower_bound:
            lower.append(var.lower_bound)
        else:
            lower.append(var.value_of(0.001))

        if hasattr(var, "upper_bound") and var.upper_bound:
            upper.append(var.upper_bound)
        else:
            upper.append(var.value_of(0.999))

    return np.linalg.norm(np.array(upper) - np.array(lower))


def get_correlation_matrix(target_correlation: Union[np.ndarray, float] = 0., num_variables: Optional[int] = None
                           ) -> np.ndarray:
    if not np.isscalar(target_correlation):
        return target_correlation
    if not num_variables:
        raise ValueError("num_variables have to be passed if the target_correlation is a scalar.")
    return np.eye(num_variables) * (1 - target_correlation) + np.ones(
        (num_variables, num_variables)) * target_correlation
