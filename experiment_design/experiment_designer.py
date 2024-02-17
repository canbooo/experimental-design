import abc
from typing import Optional

import numpy as np

from experiment_design.scorers import (
    Scorer,
    ScorerFactory,
    create_default_scorer_factory,
)
from experiment_design.variable import DesignSpace, VariableCollection

DEFAULT_INITIAL_OPTIMIZATION_PROPORTION = 0.1


class ExperimentDesigner(abc.ABC):
    """
    Abstract class of experiment designers that create and extend designs of experiments, i.e. experiment plans

    :param scorer_factory: A factory that creates scorers for the given variables, sample_size and in the cast of an
        extension, old sampling points. If not passed, a default one will be created, that evaluates the maximum
        correlation error and minimum pairwise distance. See `experiment_design.scorers.create_default_scorer_factory`
        for more details.
    """

    def __init__(self, scorer_factory: Optional[ScorerFactory] = None) -> None:
        if scorer_factory is None:
            scorer_factory = create_default_scorer_factory()
        self.scorer_factory = scorer_factory

    def design(
        self,
        variables: VariableCollection,
        sample_size: int,
        old_sample: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
        initial_optimization_proportion: float = DEFAULT_INITIAL_OPTIMIZATION_PROPORTION,
        verbose: int = 0,
    ) -> np.ndarray:
        """
        Create or extend a design of experiments (DoE).

        :param variables: Determines the dimensions of the resulting sample.
        :param sample_size: the number of points to be created.
        :param old_sample: Old DoE matrix with shape (len(variables), old_sample_size). If provided,
            it will be extended with sample_size new points, otherwise a new DoE will be created.
            In both cases, only the new points will be returned.
        :param steps: Number of DoEs to be created to choose the best from.
        :param initial_optimization_proportion:  Proportion of steps that will be used to create an
            initial DoE with a good score. Rest of the steps will be used to optimize the candidate points.
        :param verbose: Controls print messages. 2 leads to maximum verbosity,
        :return: DoE matrix with shape (len(variables), samples_size)
        """
        if not isinstance(variables, DesignSpace):
            variables = DesignSpace(variables)
        scorer = self.scorer_factory(variables, sample_size, old_sample=old_sample)
        initial_steps, final_steps = calculate_optimization_step_numbers(
            sample_size, steps, proportion=initial_optimization_proportion
        )
        if old_sample is None:
            return self._create(
                variables, sample_size, scorer, initial_steps, final_steps, verbose
            )
        return self._extend(
            old_sample,
            variables,
            sample_size,
            scorer,
            initial_steps,
            final_steps,
            verbose,
        )

    @abc.abstractmethod
    def _create(
        self,
        variables: DesignSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
        verbose: int,
    ) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError


def calculate_optimization_step_numbers(
    sample_size: int,
    steps: Optional[int] = None,
    proportion: float = DEFAULT_INITIAL_OPTIMIZATION_PROPORTION,
) -> tuple[int, int]:
    """
    Calculate the proportion of steps to be used for the initial and final optimizations.
    Also provide a sensible default total number of steps if they are not provided

    :param sample_size: The number of samples to be created. Only used to guess steps if it is
        not provided.
    :param steps: The number of total steps. If not passed, it will be guessed depending on the
        sample_size.
    :param proportion: Proportion of initial to total steps.
    :return: initial and final optimization steps (minimum 1 for each stage).
    """
    if steps is None:
        if sample_size <= 128:
            steps = 20000
        else:
            steps = 2000
    init_steps = max(1, round(proportion * steps))
    opt_steps = max(1, steps - init_steps)
    return init_steps, opt_steps
