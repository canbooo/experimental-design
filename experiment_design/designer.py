from typing import Protocol, Optional, Callable

import numpy as np


from experiment_design.scorers import Scorer
from experiment_design.variable import Variable


class DesignCreator(Protocol):
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
        ...


class DesignExtender:
    def __call__(self, old_sample: np.ndarray, variables: list[Variable], sample_size: int,
                 steps: Optional[int] = None, scorer: Optional[Scorer] = None) -> np.ndarray:
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
        ...


