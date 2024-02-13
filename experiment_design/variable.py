from dataclasses import dataclass
from typing import Optional, Union, Callable, Protocol

import numpy as np
from scipy.stats import rv_continuous, uniform, rv_discrete, randint

class Variable(Protocol):

    def value_of(self, probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        ...

@dataclass
class ContinuousVariable:
    distribution: Optional[rv_continuous] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def __post_init__(self) -> None:
        if self.distribution is None and None in [self.lower_bound, self.upper_bound]:
            raise ValueError("Either the distribution or both "
                             "lower_bound and upper_bound have to be set.")
        if self.distribution is None:
            self.distribution = uniform(self.lower_bound,
                                        self.upper_bound - self.lower_bound)
        if not isinstance(self.distribution, rv_continuous):
            raise ValueError("Only continuous distributions are supported.")

    def value_of(self, probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.distribution.ppf(probability)


@dataclass
class DiscreteVariable:
    distribution: rv_discrete
    value_mapper: Callable[[Union[float, int]], Union[float, int, str]] = lambda x: x

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, rv_discrete):
            raise ValueError("Only discrete distributions are supported.")

    def value_of(self, probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        values = self.distribution.ppf(probability)
        if not isinstance(values, np.ndarray):
            return self.value_mapper(values)

        shape = values.shape
        values = np.apply_along_axis(self.value_mapper, values.ravel(), axis=0)
        return values.reshape(shape)

def create_discrete_variables(discrete_sets: list[list[Union[int, float, str]]]
                              ) -> list[DiscreteVariable]
    variables = []
    for discrete_set in discrete_sets:
        n_values = len(discrete_set)
        if n_values < 2:
            raise ValueError("At least two values are required for discrete variables")
        variables.append(
            DiscreteVariable(
                distribution=randint(0, n_values),
                value_mapper=lambda x: discrete_set[x]
            )
        )
    return variables

def create_uniform_variables(continuous_lower_bounds: list[float],
                             continuous_upper_bounds: list[float]
                             ) -> list[Union[DiscreteVariable, ContinuousVariable]]:
    if len(continuous_lower_bounds) != len(continuous_upper_bounds):
        raise ValueError("Number of lower bounds has to be equal to the number of upper bounds")
    variables = []
    for lower, upper in zip(continuous_lower_bounds, continuous_upper_bounds):
        variables.append(
            ContinuousVariable(lower_bound=lower,
                               upper_bound=upper
            )
        )
    return variables
