from typing import Optional, Union

import numpy as np
from scipy.stats import uniform

from experiment_design.optimize import simulated_annealing_by_perturbation
from experiment_design.scorers import Scorer, make_corr_error_scorer, make_min_pairwise_distance_scorer
from experiment_design.types import VariableCollection

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
        empty = find_empty_bins(probabilities, bins_per_dimension)
        rows, cols = np.where(empty)
        while (
            self.empty_size_check(np.unique(cols, return_counts=True)[1]) < sample_size
        ):
            bins_per_dimension += 1
            empty = find_empty_bins(probabilities, bins_per_dimension)
            rows, cols = np.where(empty)

        delta = 1 / bins_per_dimension
        new_sample = np.empty((sample_size, len(variables)))
        for i_dim in range(len(variables)):
            values = rows[cols == i_dim]
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
            new_sample[:, i_dim] = values * delta + delta / 2
        if not self.central_design:
            new_sample += uniform(-delta / 2, delta).rvs(size=(sample_size, len(variables)))
        new_sample = variables.value_of(new_sample)

        if scorer is None:
            lower, upper = variables.lower_bound[None, :], variables.upper_bound[None, :]
            local_mask = np.logical_and((old_sample >= lower).all(1),
                                  (old_sample <= upper).all(1))

            corr_scorer = make_corr_error_scorer(self.target_correlation)
            lower = np.minimum(lower, old_sample.min(0))
            upper = np.minimum(upper, old_sample.max(0))
            max_distance = np.linalg.norm(upper - lower)
            dist_scorer = make_min_pairwise_distance_scorer(max_distance)

            def scorer(doe: np.ndarray) -> float:
                dist_score = dist_scorer(np.append(old_sample, doe, axis=0))
                corr_score = corr_scorer(np.append(old_sample[local_mask], doe, axis=0))
                return dist_score + DEFAULT_CORRELATION_SCORE_WEIGHT * corr_score

        return simulated_annealing_by_perturbation(
            new_sample, scorer, steps=opt_steps, verbose=self.verbose
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from experiment_design.orthogonal_sampling.creator import (
        OrthogonalDesignCreator,
        create_fast_orthogonal_design,
        create_probabilities, get_init_opt_steps,
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
