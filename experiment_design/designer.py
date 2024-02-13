from typing import Protocol, Optional

import numpy as np

from experiment_design.variable import Variable


class ExperimentDesigner(Protocol):
    def create(self, variables: list[Variable], sample_size: int) -> np.ndarray:
        """
        Create a design of experiments (DoE)

        :param variables: Determines the dimensions of the resulting sample
        :param sample_size: the number of points to be created
        :return: DoE matrix with shape (len(variables), samples_size)
        """
        ...

    def extend_doe(self, old_sample: np.ndarray, variables: list[Variable], sample_size: int,
                   ) -> np.ndarray:
        """
        Extend a design of experiment (DoE)

        :param old_sample: Old DoE matrix with shape (len(variables), old_sample_size)
        :param variables: Determines the dimensions of the resulting sample
        :param sample_size: The number of points to be added to the old_sample
        :return: Matrix of new points to be added with shape (len(variables), samples_size)
        """
        ...