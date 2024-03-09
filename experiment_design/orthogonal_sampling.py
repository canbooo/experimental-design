from functools import partial
from typing import Callable, Optional, Union

import numpy as np
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
    """
    Create or extend an orthogonal sampling design. Orthogonal sampling design partitions the design space into
    bins of equal marginal probability and places samples such that each bin is only filled once for each dimension.
    If all variables are uniform, orthogonal sampling becomes a Latin hypercube sampling.

    :param target_correlation: A symmetric matrix with shape (len(variables), len(variables)), representing the linear
        dependency between the dimensions. If a float, all non-diagonal entries of the unit matrix will be set to this
        value.
    :param inter_bin_randomness: Controls the randomness of placed points between the bin bounds. Specifically, 0. means that
        the points are placed at the center of each bin, whereas 1. leads to a random point placement within the bounds.
        Any other fractions leads to a random placement within that fraction of the bin bounds in each dimension.

    :param non_occupied_bins: Only relevant for extending the design, i.e. if old points are provided, and if the constraint
        regarding the number of occupation of each bin has to be violated. True means that each bin is occupied at least
        once for each dimension, although some bins might be occupied more often. Otherwise, each bin is occupied once
        or less often, leading to empty bins in some cases.
    :param scorer_factory: A factory that creates scorers for the given variables, sample_size and in the cast of an
        extension, old sampling points. If not passed, a default one will be created, that evaluates the maximum
        correlation error and minimum pairwise distance. See `experiment_design.scorers.create_default_scorer_factory`
        for more details.
    """

    def __init__(
        self,
        target_correlation: Union[np.ndarray, float] = 0.0,
        inter_bin_randomness: float = 1.0,
        non_occupied_bins: bool = False,
        scorer_factory: Optional[ScorerFactory] = None,
    ) -> None:
        self.target_correlation = target_correlation
        self.inter_bin_randomness = inter_bin_randomness
        if non_occupied_bins:
            self.empty_size_check = np.min
        else:
            self.empty_size_check = np.max
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
        verbose: int,
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
                inter_bin_randomness=self.inter_bin_randomness,
            )

        if verbose:
            print("Creating an initial design")
        doe = random_search(
            creator=partial(
                create_orthogonal_design,
                variables=variables,
                sample_size=sample_size,
                target_correlation=target_correlation,
                inter_bin_randomness=self.inter_bin_randomness,
            ),
            scorer=scorer,
            steps=initial_steps,
            verbose=verbose,
        )
        if verbose:
            print("Optimizing the initial design")
        return simulated_annealing_by_perturbation(
            doe, scorer, steps=final_steps, verbose=verbose
        )

    def _extend(
        self,
        old_sample: np.ndarray,
        variables: DesignSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
        verbose: int,
    ) -> np.ndarray:
        local_doe = select_local(old_sample, variables)
        probabilities = variables.cdf_of(local_doe)
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(
                "Non-finite probability encountered. Please check the distributions."
            )

        bins_per_dimension = sample_size + local_doe.shape[0]

        empty = _find_sufficient_empty_bins(
            probabilities,
            bins_per_dimension,
            sample_size,
            empty_size_check=self.empty_size_check,
        )
        if verbose:
            print("Creating candidate points to extend the design")
        new_sample = random_search(
            creator=partial(
                _create_candidates_from,
                empty_bins=empty,
                variables=variables,
                sample_size=sample_size,
                inter_bin_randomness=self.inter_bin_randomness,
            ),
            scorer=scorer,
            steps=initial_steps,
            verbose=verbose,
        )
        if verbose:
            print("Optimizing candidate points to extend the design")
        return simulated_annealing_by_perturbation(
            new_sample, scorer, steps=final_steps
        )


def create_orthogonal_design(
    variables: VariableCollection,
    sample_size: int,
    target_correlation: np.ndarray,
    inter_bin_randomness: float = 1.0,
) -> np.ndarray:
    """Create an orthogonal design without any optimization."""
    if not isinstance(variables, DesignSpace):
        variables = DesignSpace(variables)
    # Sometimes, we may randomly generate probabilities with
    # singular correlation matrices. Try 3 times to avoid issue until we give up
    for k in range(3):
        probabilities = create_lhd_probabilities(
            len(variables), sample_size, inter_bin_randomness=inter_bin_randomness
        )
        doe = variables.value_of(probabilities)
        try:
            return iman_connover_transformation(doe, target_correlation)
        except np.linalg.LinAlgError:
            pass

    return doe


def create_lhd_probabilities(
    num_variables: int,
    sample_size: int,
    inter_bin_randomness: float = 1.0,
) -> np.ndarray:
    """Create probabilities for a Latin hypercube design."""
    if not 0.0 <= inter_bin_randomness <= 1.0:
        raise ValueError(
            f"inter_bin_randomness has to be between 0 and 1, got {inter_bin_randomness}"
        )
    doe = uniform.rvs(size=(sample_size, num_variables))
    doe = (np.argsort(doe, axis=0) + 0.5) / sample_size
    if inter_bin_randomness == 0.0:
        return doe
    delta = inter_bin_randomness / sample_size
    return doe + uniform(-delta / 2, delta).rvs(size=(sample_size, num_variables))


def iman_connover_transformation(
    doe: np.ndarray,
    target_correlation: np.ndarray,
    means: Optional[np.ndarray] = None,
    standard_deviations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rearrange the values of doe to reduce correlation error while keeping the Latin hypercube constraint"""
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
    """Second-moment transformation for achieving the target covariance"""
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
    empty_bins: np.ndarray,
    variables: DesignSpace,
    sample_size: int,
    inter_bin_randomness: float = 1.0,
) -> np.ndarray:
    if not 0.0 <= inter_bin_randomness <= 1.0:
        raise ValueError(
            f"inter_bin_randomness has to be between 0 and 1, got {inter_bin_randomness}"
        )
    empty_rows, empty_cols = np.where(empty_bins)
    bins_per_dimension, dimensions = empty_bins.shape
    delta = 1 / bins_per_dimension
    probabilities = np.empty((sample_size, dimensions))
    for i_dim in range(dimensions):
        values = empty_rows[empty_cols == i_dim]
        np.random.shuffle(values)
        diff = sample_size - values.size
        if diff < 0:
            values = values[:sample_size]
        elif diff > 0:
            available = [idx for idx in range(bins_per_dimension) if idx not in values]
            extra = np.random.choice(available, diff, replace=False)
            values = np.append(extra, values)
        probabilities[:, i_dim] = values * delta + delta / 2
    if inter_bin_randomness > 0.0:
        delta *= inter_bin_randomness
        probabilities += uniform(-delta / 2, delta).rvs(size=(sample_size, dimensions))
    return variables.value_of(probabilities)
