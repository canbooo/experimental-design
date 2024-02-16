from typing import Protocol, Union

import numpy as np

from experiment_design.variable.variable import DesignSpace


class Variable(Protocol):
    def value_of(
        self, probability: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Given a probability or an array of probabilities,
        return the value using the inverse cdf of the distribution
        """

    def cdf_of(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given a value or an array of values,
        return the probability using the cdf of the distribution
        """

    def get_finite_lower_bound(self) -> float:
        """
        Return a lower bound to be used for the experiment design.
        If the user passed a lower bound or the variable distribution has a finite one,
        it will be returned. Otherwise, return an appropriate finite value.
        """

    def get_finite_upper_bound(self) -> float:
        """
        Return an upper bound to be used for the experiment design.
        If the user passed an upper bound or the variable distribution has a finite one,
        it will be returned. Otherwise, return an appropriate finite value.
        """


VariableCollection = Union[list[Variable], DesignSpace]
