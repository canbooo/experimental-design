from functools import partial
from typing import Callable, Optional, Union

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve_triangular
from scipy.stats import uniform

from experiment_design.experiment_designer import ExperimentDesigner
from experiment_design.optimize import (
    random_search,
    simulated_annealing_by_perturbation,
)
from experiment_design.scorers import (
    Scorer,
    ScorerFactory,
    create_correlation_matrix,
    create_default_scorer_factory,
    select_local,
)
from experiment_design.variable import DesignSpace, VariableCollection


class OrthogonalSamplingDesigner(ExperimentDesigner):

    def __init__(
        self,
        target_correlation: Union[np.ndarray, float] = 0.0,
        central_design: bool = False,
        dense_filling: bool = True,
        scorer_factory: Optional[ScorerFactory] = None,
    ) -> None:
        self.target_correlation = target_correlation
        self.central_design = central_design
        if dense_filling:
            self.empty_size_check = np.max
        else:
            self.empty_size_check = np.min
        if scorer_factory is None:
            scorer_factory = create_default_scorer_factory(
                target_correlation=target_correlation
            )
        super(OrthogonalSamplingDesigner, self).__init__(scorer_factory=scorer_factory)

    def _create(
        self,
        variables: DesignSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
    ) -> np.ndarray:
        target_correlation = create_correlation_matrix(
            self.target_correlation, num_variables=variables.dimensions
        )
        if (initial_steps + final_steps) <= 2:
            # Enable faster use cases:
            return create_orthogonal_design(
                variables=variables,
                sample_size=sample_size,
                target_correlation=target_correlation,
                central_design=self.central_design,
            )

        doe = random_search(
            partial(
                create_orthogonal_design,
                variables=variables,
                sample_size=sample_size,
                target_correlation=target_correlation,
                central_design=self.central_design,
            ),
            scorer,
            initial_steps,
        )
        return simulated_annealing_by_perturbation(doe, scorer, steps=final_steps)

    def _extend(
        self,
        old_sample: np.ndarray,
        variables: DesignSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
    ) -> np.ndarray:
        probabilities = variables.cdf_of(select_local(old_sample, variables))
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(
                "Non-finite probability encountered. Please check the distributions."
            )

        bins_per_dimension = sample_size + old_sample.shape[0]

        empty = _find_sufficient_empty_bins(
            probabilities, bins_per_dimension, sample_size
        )

        new_sample = random_search(
            partial(
                _create_candidates_from,
                empty_bins=empty,
                variables=variables,
                central_design=self.central_design,
            ),
            scorer,
            initial_steps,
        )

        return simulated_annealing_by_perturbation(
            new_sample, scorer, steps=final_steps
        )


def create_orthogonal_design(
    variables: VariableCollection,
    sample_size: int,
    target_correlation: np.ndarray,
    central_design: bool = True,
) -> np.ndarray:
    if not isinstance(variables, DesignSpace):
        variables = DesignSpace(variables)
    # Sometimes, we may randomly generate probabilities with
    # singular correlation matrices. Try 3 times to avoid issue until we give up
    for k in range(3):
        probabilities = create_lhd_probabilities(
            len(variables), sample_size, central_design=central_design
        )
        doe = variables.value_of(probabilities)
        try:
            return iman_connover_transformation(doe, target_correlation)
        except np.linalg.LinAlgError as exc:
            pass
    raise


def create_lhd_probabilities(
    num_variables: int, sample_size: int, central_design: bool = True
) -> np.ndarray:
    doe = uniform.rvs(size=(sample_size, num_variables))
    doe = (np.argsort(doe, axis=0) + 0.5) / sample_size
    if central_design:
        return doe
    delta = 1 / sample_size
    return doe + uniform(-delta / 2, delta).rvs(size=(sample_size, num_variables))


def iman_connover_transformation(
    doe: np.ndarray,
    target_correlation: np.ndarray,
    means: Optional[np.ndarray] = None,
    standard_deviations: Optional[np.ndarray] = None,
) -> np.ndarray:
    # See Chapter 4.3.2 of
    # Local Latin Hypercube Refinement for Uncertainty Quantification and Optimization, Can Bogoclu, (2022)
    if means is None:
        means = np.mean(doe, axis=0)
    if standard_deviations is None:
        standard_deviations = np.std(doe, axis=0, keepdims=True)
        standard_deviations = standard_deviations.reshape((1, -1))
    target_covariance = (
        standard_deviations.T.dot(standard_deviations) * target_correlation
    )

    transformed = second_moment_transformation(doe, means, target_covariance)
    order = np.argsort(np.argsort(transformed, axis=0), axis=0)
    return np.take_along_axis(np.sort(doe, axis=0), order, axis=0)


def second_moment_transformation(
    doe: np.ndarray,
    means: Union[float, np.ndarray],
    target_covariance: np.ndarray,
) -> np.ndarray:
    target_cov_upper = np.linalg.cholesky(
        target_covariance
    ).T  # convert to covariance before Cholesky
    cur_cov_upper = np.linalg.cholesky(np.cov(doe, rowvar=False)).T
    inv_cov_upper = solve_triangular(cur_cov_upper, target_cov_upper)
    return (doe - means).dot(inv_cov_upper) + means


def _find_sufficient_empty_bins(
    probabilities: np.ndarray,
    bins_per_dimension: int,
    required_sample_size: int,
    empty_size_check: Callable[[np.ndarray], float] = np.max,
) -> np.ndarray:
    empty = _find_empty_bins(probabilities, bins_per_dimension)
    cols = np.where(empty)[1]
    while (
        empty_size_check(np.unique(cols, return_counts=True)[1]) < required_sample_size
    ):
        bins_per_dimension += 1
        empty = _find_empty_bins(probabilities, bins_per_dimension)
        cols = np.where(empty)[1]
    return empty


def _find_empty_bins(probabilities: np.ndarray, bins_per_dimension: int) -> np.ndarray:
    """
    Find empty bins on an orthogonal sampling grid given the probabilities.

    :param probabilities: Array of cdf values of the observed points.
    :param bins_per_dimension: Determines the size of the grid to be tested.
    :return: Boolean array of empty bins with shape=(n_bins, n_dims).
    """
    empty_bins = np.ones((bins_per_dimension, probabilities.shape[1]), dtype=bool)
    edges = np.arange(bins_per_dimension + 1) / bins_per_dimension
    edges = edges.reshape((-1, 1))
    for i_dim in range(probabilities.shape[1]):
        condition = np.logical_and(
            probabilities[:, i_dim] >= edges[:-1], probabilities[:, i_dim] < edges[1:]
        )
        empty_bins[:, i_dim] = np.logical_not(condition.any(1))
    return empty_bins


def _create_candidates_from(
    empty_bins: np.ndarray, variables: DesignSpace, central_design: bool = False
) -> np.ndarray:
    empty_rows, empty_cols = np.where(empty_bins)
    bins_per_dimension, dimensions = empty_bins.shape
    delta = 1 / bins_per_dimension
    probabilities = np.empty((empty_rows, dimensions))
    for i_dim in range(dimensions):
        values = empty_rows[empty_cols == i_dim]
        np.random.shuffle(values)
        diff = empty_rows - values.size
        if diff < 0:
            values = values[:empty_rows]
        elif diff > 0:
            available = [idx for idx in range(bins_per_dimension) if idx not in values]
            extra = np.random.choice(available, diff, replace=False)
            values = np.append(extra, values)
        probabilities[:, i_dim] = values * delta + delta / 2
    if not central_design:
        probabilities += uniform(-delta / 2, delta).rvs(size=(empty_rows, dimensions))
    return variables.value_of(probabilities)
