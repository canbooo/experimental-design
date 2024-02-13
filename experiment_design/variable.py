from dataclasses import dataclass
from typing import Optional, Union, Callable, Protocol, Any

import numpy as np
from scipy.stats import randint, rv_continuous, rv_discrete, uniform


class Variable(Protocol):

    def value_of(self, probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        ...


def is_frozen_discrete(dist: Any) -> bool:
    if not hasattr(dist, 'dist'):
        return False
    return isinstance(dist.dist, rv_discrete)


def is_frozen_continuous(dist: Any) -> bool:
    if not hasattr(dist, 'dist'):
        return False
    return isinstance(dist.dist, rv_continuous)


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
        if None not in [self.lower_bound, self.upper_bound] and self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound has to be smaller than upper_bound")
        if not is_frozen_continuous(self.distribution):
            raise ValueError("Only frozen continuous distributions are supported.")

    def value_of(self, probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        values = self.distribution.ppf(probability)
        if self.upper_bound is not None or self.lower_bound is not None:
            return np.clip(values, self.lower_bound, self.upper_bound)
        return values


@dataclass
class DiscreteVariable:
    distribution: rv_discrete
    value_mapper: Callable[[float], Union[float, int]] = lambda x: x

    def __post_init__(self) -> None:
        if not is_frozen_discrete(self.distribution):
            raise ValueError("Only frozen discrete distributions are supported.")
        self.value_mapper = np.frompyfunc(self.value_mapper, nin=1, nout=1)

    def value_of(self, probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        values = self.distribution.ppf(probability)
        return self.value_mapper(values)


def create_discrete_variables(discrete_sets: list[list[Union[int, float, str]]]
                              ) -> list[DiscreteVariable]:
    variables = []
    for discrete_set in discrete_sets:
        n_values = len(discrete_set)
        if n_values < 2:
            raise ValueError("At least two values are required for discrete variables")
        variables.append(
            DiscreteVariable(
                distribution=randint(0, n_values),
                # Don't forget to bind the discrete_set below either by
                # defining a kwarg as done here, or by generating in in another
                # scope, e.g. function. Otherwise, the last value of discrete_set
                # i.e. the last entry of discrete_sets will be used for all converters
                # Check https://stackoverflow.com/questions/19837486/lambda-in-a-loop
                # for a description as this is expected python behaviour.
                value_mapper=lambda x, values=discrete_set: values[int(x)]
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


def map_probabilities_to_values(probabilities: np.ndarray, variables: list[Variable]) -> np.ndarray:
    """
    Given an array of probabilities, return the corresponding values of the variables

    :param probabilities: matrix of probabilities with shape (sample_size, len(variables))
    :param variables: Variables to compute the corresponding values using the value_of method
    :return: matrix of values with the same shape as the probabilities
    """
    samples = np.zeros(probabilities.shape)
    for i_dim, variable in enumerate(variables):
        samples[:, i_dim] = variable.value_of(samples[:, i_dim])
    return samples
