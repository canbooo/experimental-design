import abc
from typing import Optional

import numpy as np

from experiment_design.scorers import (
    CreationScoreFactory,
    ExtensionScoreFactory,
    Scorer,
)
from experiment_design.variable import DesignSpace, VariableCollection

INITIAL_OPTIMIZATION_PROPORTION = 0.1


class Designer(abc.ABC):

    def __init__(
        self,
        creation_score_factory: CreationScoreFactory,
        extension_score_factory: ExtensionScoreFactory,
        initial_optimization_proportion: float = INITIAL_OPTIMIZATION_PROPORTION,
    ):
        self.creation_score_factory = creation_score_factory
        self.extension_score_factory = extension_score_factory
        self.initial_optimization_proportion = initial_optimization_proportion

    def create(
        self,
        variables: VariableCollection,
        sample_size: int,
        steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create a design of experiments (DoE)

        :param variables: Determines the dimensions of the resulting sample
        :param sample_size: the number of points to be created
        :param steps: Number of DoEs to be created to choose the best from
        :return: DoE matrix with shape (len(variables), samples_size)
        """
        if not isinstance(variables, DesignSpace):
            variables = DesignSpace(variables)
        scorer = self.creation_score_factory(variables, sample_size)
        initial_steps, final_steps = get_init_opt_steps(
            sample_size, steps, proportion=self.initial_optimization_proportion
        )
        return self._create(variables, sample_size, scorer, initial_steps, final_steps)

    def extend(
        self,
        old_sample: np.ndarray,
        variables: VariableCollection,
        sample_size: int,
        steps: Optional[int] = None,
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
        scorer = self.extension_score_factory(old_sample, variables, sample_size)
        initial_steps, final_steps = get_init_opt_steps(
            sample_size, steps, proportion=self.initial_optimization_proportion
        )
        return self._extend(
            old_sample, variables, sample_size, scorer, initial_steps, final_steps
        )

    @abc.abstractmethod
    def _create(
        self,
        variables: DesignSpace,
        sample_size: int,
        scorer: Scorer,
        initial_steps: int,
        final_steps: int,
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
    ) -> np.ndarray:
        raise NotImplementedError


def get_init_opt_steps(
    samples_size: int,
    steps: Optional[int] = None,
    proportion: float = INITIAL_OPTIMIZATION_PROPORTION,
) -> tuple[int, int]:
    if steps is None:
        if samples_size <= 100:
            steps = 20000
        else:
            steps = 2000
    init_steps = max(1, round(proportion * steps))
    opt_steps = max(1, steps - init_steps)
    return init_steps, opt_steps
