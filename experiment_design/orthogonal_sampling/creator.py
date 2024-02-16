from functools import partial
from typing import Optional, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import uniform

from experiment_design.optimize import get_best_try, simulated_annealing_by_perturbation
from experiment_design.scorers import (
    Scorer,
    get_correlation_matrix,
    make_default_scorer,
)
from experiment_design.variable import DesignSpace, VariableCollection


def create_probabilities(
    num_variables: int, sample_size: int, central_design: bool = True
):
    doe = uniform.rvs(size=(sample_size, num_variables))
    doe = (np.argsort(doe, axis=0) + 0.5) / sample_size
    if central_design:
        return doe
    delta = 1 / sample_size
    return doe + uniform(-delta / 2, delta).rvs(size=(sample_size, num_variables))


def second_moment_transformation(
    doe: np.ndarray,
    target_correlation: np.ndarray,
    mean: Union[np.ndarray, float] = 0.5,
    variance: Union[np.ndarray, float] = 1 / 12,
) -> np.ndarray:
    target_cov_upper = np.linalg.cholesky(
        target_correlation * variance
    ).T  # convert to covariance before Cholesky
    cur_cov_upper = np.linalg.cholesky(np.cov(doe, rowvar=False)).T
    inv_cov_upper = solve_triangular(cur_cov_upper, target_cov_upper)
    return (doe - mean).dot(inv_cov_upper) + mean


def iman_connover_transformation(
    doe: np.ndarray,
    target_correlation: np.ndarray,
    mean: Union[np.ndarray, float] = 0.5,
    variance: Union[np.ndarray, float] = 1 / 12,
) -> np.ndarray:
    # See Chapter 4.3.2 of
    # Local Latin Hypercube Refinement for Uncertainty Quantification and Optimization, Can Bogoclu, (2022)
    new = second_moment_transformation(
        doe, target_correlation, mean=mean, variance=variance
    )
    order = np.argsort(np.argsort(new, axis=0), axis=0)
    return np.take_along_axis(np.sort(doe, axis=0), order, axis=0)


def generate_lhd_probabilities(
    num_variables: int,
    sample_size: int,
    target_correlation: np.ndarray,
    central_design: bool = True,
) -> np.ndarray:
    target_correlation = get_correlation_matrix(
        target_correlation=target_correlation, num_variables=num_variables
    )
    # Sometimes, we may randomly generate probabilities with
    # singular correlation matrices. Try 3 times to avoid issue until we give up
    for k in range(3):
        probabilities = create_probabilities(
            num_variables, sample_size, central_design=central_design
        )
        try:
            return iman_connover_transformation(probabilities, target_correlation)
        except np.linalg.LinAlgError:
            pass


def create_fast_orthogonal_design(
    variables: VariableCollection,
    sample_size: int,
    steps: Optional[int] = None,
    scorer: Optional[Scorer] = None,
) -> np.ndarray:
    """
    Create an orthogonal design without any correlation transformation. Useful for
    creating very large designs (n >> 10_000). For smaller designs, please use
    OrthogonalDesignCreator

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
    if steps is not None or scorer is not None:
        raise ValueError(
            "This function does not use any optimization. Please use OrthogonalDesignCreator"
        )
    doe = create_probabilities(len(variables), sample_size, central_design=True)
    return variables.value_of(doe)


def get_init_opt_steps(
    samples_size: int, steps: Optional[int] = None, proportion: float = 0.1
) -> tuple[int, int]:
    if steps is None:
        if samples_size <= 100:
            steps = 20000
        else:
            steps = 2000
    init_steps = max(1, round(proportion * steps))
    opt_steps = max(1, steps - init_steps)
    return init_steps, opt_steps


class OrthogonalDesignCreator:
    def __init__(
        self,
        target_correlation: Union[np.ndarray, float] = 0.0,
        central_design: bool = False,
        verbose: int = 0,
    ) -> None:
        self.target_correlation = target_correlation
        self.central_design = central_design
        self.verbose = verbose

    def __call__(
        self,
        variables: VariableCollection,
        sample_size: int,
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
        if not isinstance(variables, DesignSpace):
            variables = DesignSpace(variables)
        num_variables = variables.dimensions
        target_correlation = get_correlation_matrix(
            self.target_correlation, num_variables=num_variables
        )
        init_steps, opt_steps = get_init_opt_steps(sample_size, steps=steps)
        if (init_steps + opt_steps) <= 2:
            # Enable faster use cases:
            doe = generate_lhd_probabilities(
                num_variables=num_variables,
                sample_size=sample_size,
                target_correlation=target_correlation,
                central_design=self.central_design,
            )
            return variables.value_of(doe)
        target_correlation = get_correlation_matrix(
            self.target_correlation, num_variables=num_variables
        )
        if scorer is None:
            scorer = make_default_scorer(variables, target_correlation)

        doe = get_best_try(
            partial(
                generate_lhd_probabilities,
                num_variables,
                sample_size,
                target_correlation,
                central_design=self.central_design,
            ),
            scorer,
            init_steps,
        )

        doe = variables.value_of(doe)
        return simulated_annealing_by_perturbation(
            doe, scorer, steps=opt_steps, verbose=self.verbose
        )
