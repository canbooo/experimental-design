from functools import partial

import numpy as np
from scipy.stats import uniform

from experiment_design.experiment_designer import ExperimentDesigner
from experiment_design.optimize import random_search
from experiment_design.scorers import Scorer
from experiment_design.variable import DesignSpace, VariableCollection


class RandomSamplingDesigner(ExperimentDesigner):
    """Create or extend a design of experiments (DoE) by randomly sampling from the variable distributions."""

    def _create(
        self,
        variables: DesignSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
        verbose: int,
    ) -> np.ndarray:
        steps = initial_steps + final_steps
        return random_search(
            creator=partial(sample_from, variables, sample_size),
            scorer=scorer,
            steps=steps,
            verbose=verbose,
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
        steps = initial_steps + final_steps
        return random_search(
            creator=partial(sample_from, variables, sample_size),
            scorer=scorer,
            steps=steps,
            verbose=verbose,
        )


def sample_from(variables: VariableCollection, sample_size: int) -> np.ndarray:
    """
    Sample from the distributions of the variables.

    :param variables: Determines the dimensions of the resulting sample.
    :param sample_size: the number of points to be created.
    :return: Sample matrix with shape (len(variables), samples_size).
    """
    doe = uniform(0, 1).rvs((sample_size, len(variables)))
    if not isinstance(variables, DesignSpace):
        variables = DesignSpace(variables)
    return variables.value_of(doe)
