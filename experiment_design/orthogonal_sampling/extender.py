from functools import partial
from typing import Optional, Union

import numpy as np
from scipy.stats import uniform

from experiment_design.optimize import get_best_try, simulated_annealing_by_perturbation
from experiment_design.scorers import (
    Scorer,
    get_correlation_matrix,
    make_corr_error_scorer,
    make_min_pairwise_distance_scorer,
)
from experiment_design.variable import DesignSpace, VariableCollection

DEFAULT_CORRELATION_SCORE_WEIGHT = 0.2


def find_empty_bins(probabilities: np.ndarray, bins_per_dimension: int) -> np.ndarray:
    """
    Find empty bins on an orthogonal sampling grid given the probabilities

    :param probabilities: Array of cdf values of the observed points
    :param bins_per_dimension: Determines the size of the grid to be tested
    :return: Boolean array of empty bins with shape=(n_bins, n_dims)
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


class OrthogonalDesignExtender:
    def __init__(
        self,
        target_correlation: Union[np.ndarray, float] = 0.0,
        central_design: bool = False,
        dense_filling: bool = True,
        verbose: int = 0,
    ) -> None:
        self.target_correlation = target_correlation
        self.central_design = central_design
        if dense_filling:
            self.empty_size_check = np.max
        else:
            self.empty_size_check = np.min
        self.verbose = verbose

    def find_sufficient_empty_bins(
        self,
        probabilities: np.ndarray,
        bins_per_dimension: int,
        required_sample_size: int,
    ) -> np.ndarray:

        empty = find_empty_bins(probabilities, bins_per_dimension)
        cols = np.where(empty)[1]
        while (
            self.empty_size_check(np.unique(cols, return_counts=True)[1])
            < required_sample_size
        ):
            bins_per_dimension += 1
            empty = find_empty_bins(probabilities, bins_per_dimension)
            cols = np.where(empty)[1]
        return empty

    def _get_new_candidates(
        self, variables: DesignSpace, empty: np.ndarray
    ) -> np.ndarray:
        empty_rows, empty_cols = np.where(empty)
        bins_per_dimension, dimensions = empty.shape
        delta = 1 / bins_per_dimension
        probabilities = np.empty((sample_size, dimensions))
        for i_dim in range(dimensions):
            values = empty_rows[empty_cols == i_dim]
            np.random.shuffle(values)
            diff = sample_size - values.size
            if diff < 0:
                values = values[:sample_size]
            elif diff > 0:
                available = [
                    idx for idx in range(bins_per_dimension) if idx not in values
                ]
                extra = np.random.choice(available, diff, replace=False)
                values = np.append(extra, values)
            probabilities[:, i_dim] = values * delta + delta / 2
        if not self.central_design:
            probabilities += uniform(-delta / 2, delta).rvs(
                size=(sample_size, dimensions)
            )
        return variables.value_of(probabilities)

    def __call__(
        self,
        old_sample: np.ndarray,
        variables: VariableCollection,
        sample_size: int,
        steps: Optional[int] = None,
        scorer: Optional[Scorer] = None,
    ) -> np.ndarray:
        """
        Extend a design of experiment (DoE)

        :param old_sample: Old DoE matrix with shape (len(variables), old_sample_size)
        :param variables: Determines the dimensions of the resulting sample
        :param sample_size: The number of points to be added to the old_sample
        :param scorer: Used to rank the generated DoEs. Specifically, steps
        number of DoEs will be created and the one with the highest score
        will be returned.
        :param steps: Number of DoEs to be created to choose the best from
        :return: Matrix of new points to be added with shape (len(variables), samples_size)
        """
        if not isinstance(variables, DesignSpace):
            variables = DesignSpace(variables)
        probabilities = variables.cdf_of(old_sample)
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(
                "Non-finite probabilitiy encountered. Please check the distributions."
            )

        bins_per_dimension = sample_size + old_sample.shape[0]
        init_steps, opt_steps = get_init_opt_steps(bins_per_dimension, steps=steps)
        empty = self.find_sufficient_empty_bins(
            probabilities, bins_per_dimension, sample_size
        )

        if scorer is None:
            target_correlation = get_correlation_matrix(
                self.target_correlation, num_variables=len(variables)
            )
            scorer = make_default_local_scorer(
                variables, old_sample, target_correlation
            )

        new_sample = get_best_try(
            partial(self._get_new_candidates, variables, empty),
            scorer,
            init_steps,
        )

        return simulated_annealing_by_perturbation(
            new_sample, scorer, steps=opt_steps, verbose=self.verbose
        )


def make_default_local_scorer(
    variables: DesignSpace,
    old_sample: np.ndarray,
    target_correlation: np.ndarray,
    correlation_score_weight: float = 0.2,
) -> Scorer:
    lower, upper = variables.lower_bound[None, :], variables.upper_bound[None, :]
    local_mask = np.logical_and(
        (old_sample >= lower).all(1), (old_sample <= upper).all(1)
    )

    corr_scorer = make_corr_error_scorer(target_correlation)
    lower = np.minimum(lower, old_sample.min(0))
    upper = np.minimum(upper, old_sample.max(0))
    max_distance = np.linalg.norm(upper - lower)
    dist_scorer = make_min_pairwise_distance_scorer(max_distance)

    def _scorer(doe: np.ndarray) -> float:
        dist_score = dist_scorer(np.append(old_sample, doe, axis=0))
        corr_score = corr_scorer(np.append(old_sample[local_mask], doe, axis=0))
        return dist_score + correlation_score_weight * corr_score

    return _scorer


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from experiment_design.designer import get_init_opt_steps
    from experiment_design.orthogonal_sampling.creator import (
        OrthogonalDesignCreator,
        create_fast_orthogonal_design,
        create_probabilities,
    )
    from experiment_design.variable import (
        DesignSpace,
        create_continuous_uniform_variables,
    )

    # np.random.seed(666)
    sample_size = 8
    lb, ub = -2, 2
    vars = create_continuous_uniform_variables([lb, lb], [ub, ub])
    cr = OrthogonalDesignCreator(central_design=False)
    doe = cr(vars, sample_size, steps=250)
    # doe = create_fast_orthogonal_design(vars, sample_size)

    new_doe = OrthogonalDesignExtender()(doe, vars, sample_size)
    grid = np.linspace(lb, ub, sample_size * 2 + 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(doe[:, 0], doe[:, 1])
    ax.scatter(new_doe[:, 0], new_doe[:, 1])

    full_fact = grid
    ax.vlines(grid, lb, ub)
    ax.hlines(grid, lb, ub)
    plt.show()
