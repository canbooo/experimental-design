from typing import Union, Optional

import numpy as np
from scipy.stats import uniform

from experiment_design.scorers import Scorer
from experiment_design.types import VariableCollection


def find_empty_bins(probabilities: np.ndarray, bins_per_dimension: int) -> np.ndarray:
    """
    Find empty bins on an orthogonal sampling grid given the probabilities

    :param probabilities: Array of cdf values of the observed points
    :param bins_per_dimension: Determines the size of the grid to be tested
    :return: Boolean array of empty bins with shape=(n_bins, n_dims)
    """
    empty_bins = np.ones((bins_per_dimension, probabilities.shape[1]), dtype=bool)
    edges = np.arange(bins_per_dimension + 1) / bins_per_dimension
    for i_bin in range(bins_per_dimension):
        condition = np.logical_and(
            probabilities > edges[i_bin], probabilities <= edges[i_bin + 1]
        )
        empty_bins[i_bin, :] = np.logical_not(condition.any(0))
    return empty_bins


class OrthogonalDesignExtender:
    def __init__(
        self,
        target_correlation: Union[np.ndarray, float] = 0.0,
        central_design: bool = False,
        dense_filling: bool = True,
    ) -> None:
        self.target_correlation = target_correlation
        self.central_design = central_design
        if dense_filling:
            self.empty_size_check = np.max
        else:
            self.empty_size_check = np.min

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

        bins_per_dimension = sample_size
        empty = find_empty_bins(probabilities, bins_per_dimension)
        rows, cols = np.where(empty)
        while np.max(np.unique(cols, return_counts=True)[1]) < sample_size:
            bins_per_dimension += 1
            empty = find_empty_bins(probabilities, bins_per_dimension)
            rows, cols = np.where(empty)
        print(np.unique(cols, return_counts=True))

        delta = 1 / bins_per_dimension
        doe = np.empty((sample_size, len(variables)))
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
            doe[:, i_dim] = values * delta + delta / 2
        if not self.central_design:
            doe += uniform(-delta / 2, delta).rvs(size=(sample_size, len(variables)))
        # TODO: Optimization
        return variables.value_of(doe)


if __name__ == "__main__":
    from experiment_design.variable import (
        create_continuous_uniform_variables,
        DesignSpace,
    )
    from experiment_design.orthogonal_sampling.creator import (
        OrthogonalDesignCreator,
        create_fast_orthogonal_design,
        create_probabilities,
    )
    import matplotlib.pyplot as plt

    np.random.seed(666)
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
