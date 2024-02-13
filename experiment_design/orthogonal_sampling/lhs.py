from typing import Union, Optional
from functools import partial

import numpy as np
from scipy.stats import uniform
from scipy.linalg import solve_triangular

from experiment_design.optimize import get_best_try
from experiment_design.scorers import get_correlation_matrix, Scorer, make_default_scorer, make_corr_error_scorer
from experiment_design.variable import Variable, map_probabilities_to_values


def _create_probabilities(num_variables: int, sample_size: int):
    doe = uniform.rvs(size=(sample_size, num_variables))
    return (np.argsort(doe, axis=0) - 0.5) / sample_size


def _second_moment_transformation(doe: np.ndarray, target_correlation: np.ndarray) -> np.ndarray:
    # Assumption: doe is uniform in [0,1] for all dimensions
    target_cov_upper = np.linalg.cholesky(target_correlation / 12).T  # convert to covariance before Cholesky
    cur_cov_upper = np.linalg.cholesky(np.cov(doe, rowvar=False)).T
    inv_cov_upper = solve_triangular(cur_cov_upper, target_cov_upper)
    return (doe - doe.mean(0)).dot(inv_cov_upper) + 0.5


def _iman_connover(doe: np.ndarray, target_correlation: np.ndarray) -> np.ndarray:
    # See Chapter 4.3.2 of Local Latin Hypercube Refinement for Uncertainty Quantification and Optimization (2022)
    new = _second_moment_transformation(doe, target_correlation=target_correlation)
    return np.argsort(np.argsort(new, axis=0), axis=0) / new.shape[0]


def _generate_lhd_probabilities(num_variables: int, sample_size: int, target_correlation: np.ndarray) -> np.ndarray:
    probabilities = _create_probabilities(num_variables, sample_size)
    target_correlation = get_correlation_matrix(target_correlation=target_correlation,
                                                num_variables=num_variables)
    return _iman_connover(probabilities, target_correlation)


def generate_lhd_probabilities(num_variables: int, sample_size: int, scorer: Scorer,
                               target_correlation: np.ndarray, steps: int) -> np.ndarray:
    return get_best_try(
        partial(_generate_lhd_probabilities, num_variables, sample_size, target_correlation),
        scorer,
        steps
    )


class LatinHypercubeDesignCreator:

    def __init__(self, target_correlation: Union[np.ndarray, float] = 0., central_design=False) -> None:
        self.target_correlation = target_correlation
        self.central_design = central_design

    def __call__(self, variables: list[Variable], sample_size: int,
                 steps: Optional[int] = None,
                 scorer: Optional[Scorer] = None,
                 ) -> np.ndarray:
        """
        Create a design of experiments (DoE)

        :param variables: Determines the dimensions of the resulting sample
        :param sample_size: the number of points to be created
        :param scorer: Used to rank the generated DoEs. Specifically, steps
        number of DoEs will be created and the one with the highest score
        will be returned.
        :param steps: Number of DoEs to be created to choose the best from
        :return: DoE matrix with shape (len(variables), samples_size)
        """
        # TODO: Assert uniform variables
        init_steps = max(1, round(0.1 * steps))
        opt_steps = max(1, steps - init_steps)
        target_correlation = get_correlation_matrix(self.target_correlation, num_variables=len(variables))
        init_scorer = make_corr_error_scorer(target_correlation)
        doe = generate_lhd_probabilities(len(variables), sample_size, init_scorer, target_correlation,
                                         init_steps)
        if scorer is None:
            scorer = make_default_scorer(variables, target_correlation)
        doe = map_probabilities_to_values(doe, variables)
        return simulated_annealing(doe, scorer, opt_steps)

