from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Union

import numpy as np
from scipy.stats import randint, rv_continuous, rv_discrete, uniform

# Following is ugly, but it is scipy's fault for not exposing rv_frozen
# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen


def is_frozen_discrete(dist: Any) -> bool:
    return isinstance(dist, rv_frozen) and isinstance(dist.dist, rv_discrete)


def is_frozen_continuous(dist: Any) -> bool:
    return isinstance(dist, rv_frozen) and isinstance(dist.dist, rv_continuous)


def change_field_representation(
    dataclass_instance: dataclass, representations_to_change: dict[str, Any]
) -> str:
    """Just like the default __repr__ but supports reformatting some values."""
    final = []
    for current_field in dataclass_instance.__dataclass_fields__.values():
        if not current_field.repr:
            continue
        name = current_field.name
        value = representations_to_change.get(
            name, dataclass_instance.__getattribute__(name)
        )
        final.append(f"{name}={value}")
    return f"{dataclass_instance.__class__.__name__}({', '.join(final)})"


def create_distribution_representation(distribution: rv_frozen) -> str:
    args = ", ".join([str(a) for a in distribution.args])
    kwargs = ", ".join([f"{k}={v}" for k, v in distribution.kwds])
    params = [a for a in [args, kwargs] if a]
    return f"{distribution.dist.name}({', '.join(params)})"


@dataclass
class ContinuousVariable:
    distribution: Optional[rv_frozen] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def __post_init__(self) -> None:
        if self.distribution is None and None in [self.lower_bound, self.upper_bound]:
            raise ValueError(
                "Either the distribution or both "
                "lower_bound and upper_bound have to be set."
            )
        if self.distribution is None:
            self.distribution = uniform(
                self.lower_bound, self.upper_bound - self.lower_bound
            )
        if (
            None not in [self.lower_bound, self.upper_bound]
            and self.lower_bound >= self.upper_bound
        ):
            raise ValueError("lower_bound has to be smaller than upper_bound")
        if not is_frozen_continuous(self.distribution):
            raise ValueError("Only frozen continuous distributions are supported.")

    def value_of(
        self, probability: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Given a probability or an array of probabilities,
        return the value using the inverse cdf of the distribution
        """
        values = self.distribution.ppf(probability)
        if self.upper_bound is not None or self.lower_bound is not None:
            return np.clip(values, self.lower_bound, self.upper_bound)
        return values

    def cdf_of(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given a value or an array of values,
        return the probability using the cdf of the distribution
        """
        return self.distribution.cdf(value)

    def get_finite_lower_bound(
        self, infinite_support_probability_tolerance: float = 1e-6
    ) -> float:
        if self.lower_bound is not None:
            return self.lower_bound
        value = self.value_of(0.0)
        if np.isfinite(value):
            return value
        return self.value_of(infinite_support_probability_tolerance)

    def get_finite_upper_bound(
        self, infinite_support_probability_tolerance: float = 1e-6
    ):
        if self.upper_bound is not None:
            return self.upper_bound
        value = self.value_of(1.0)
        if np.isfinite(value):
            return value
        return self.value_of(1 - infinite_support_probability_tolerance)

    def __repr__(self) -> str:
        distribution_representation = create_distribution_representation(
            self.distribution
        )
        return change_field_representation(
            self, {"distribution": distribution_representation}
        )


@dataclass
class DiscreteVariable:
    distribution: rv_frozen
    value_mapper: Callable[[float], Union[float, int]] = lambda x: x
    inverse_value_mapper: Callable[[float, int], Union[float]] = lambda x: x

    def __post_init__(self) -> None:
        if not is_frozen_discrete(self.distribution):
            raise ValueError("Only frozen discrete distributions are supported.")
        self.value_mapper = np.vectorize(self.value_mapper)
        self.inverse_value_mapper = np.vectorize(self.inverse_value_mapper)

    def value_of(
        self, probability: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Given a probability or an array of probabilities,
        return the value using the inverse cdf of the distribution
        """
        values = self.distribution.ppf(probability)
        return self.value_mapper(values)

    def cdf_of(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Given a value or an array of values,
        return the probability using the cdf of the distribution
        """
        return self.distribution.cdf(self.inverse_value_mapper(values))

    def get_finite_lower_bound(
        self, infinite_support_probability_tolerance: float = 1e-6
    ) -> float:
        support = self.distribution.support()
        if np.isfinite(support[0]):
            return self.value_mapper(support[0])
        return self.value_of(infinite_support_probability_tolerance)

    def get_finite_upper_bound(
        self, infinite_support_probability_tolerance: float = 1e-6
    ) -> float:
        support = self.distribution.support()
        if np.isfinite(support[1]):
            return self.value_mapper(support[1])
        return self.value_of(1 - infinite_support_probability_tolerance)

    def __repr__(self) -> str:
        distribution_representation = create_distribution_representation(
            self.distribution
        )
        return change_field_representation(
            self, {"distribution": distribution_representation}
        )


def create_discrete_uniform_variables(
    discrete_sets: list[list[Union[int, float, str]]]
) -> list[DiscreteVariable]:
    """
    Given sets of possible values, create corresponding discrete variables
    with equal probability of each value
    """
    variables = []
    for discrete_set in discrete_sets:
        n_values = len(discrete_set)
        if n_values < 2:
            raise ValueError("At least two values are required for discrete variables")
        # In the following, it is OK and even advantageous to have a mutable
        # default argument as a very rare occasion. Therefore, we disable inspection.
        # noinspection PyDefaultArgument
        variables.append(
            DiscreteVariable(
                distribution=randint(0, n_values),
                # Don't forget to bind the discrete_set below either by
                # defining a kwarg as done here, or by generating in another
                # scope, e.g. function. Otherwise, the last value of discrete_set
                # i.e. the last entry of discrete_sets will be used for all converters
                # Check https://stackoverflow.com/questions/19837486/lambda-in-a-loop
                # for a description as this is expected python behaviour.
                value_mapper=lambda x, values=sorted(discrete_set): values[int(x)],
                inverse_value_mapper=lambda x, values=sorted(
                    discrete_set
                ): values.index(x),
            )
        )
    return variables


def create_continuous_uniform_variables(
    continuous_lower_bounds: Iterable[float], continuous_upper_bounds: Iterable[float]
) -> list[ContinuousVariable]:
    """
    Given lower and upper bounds, create uniform variables
    """
    if len(continuous_lower_bounds) != len(continuous_upper_bounds):
        raise ValueError(
            "Number of lower bounds has to be equal to the number of upper bounds"
        )
    variables = []
    for lower, upper in zip(continuous_lower_bounds, continuous_upper_bounds):
        variables.append(ContinuousVariable(lower_bound=lower, upper_bound=upper))
    return variables


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


def create_variables_from_distributions(
    distributions: list[rv_frozen],
) -> list[Variable]:
    """
    Given a list of distributions, create the corresponding continuous or discrete variables

    :param distributions: Frozen scipy distributions of the variables to be created
    :return: List of created variables
    """
    variables = []
    for dist in distributions:
        if is_frozen_discrete(dist):
            variables.append(DiscreteVariable(distribution=dist))
        elif is_frozen_continuous(dist):
            variables.append(ContinuousVariable(distribution=dist))
        else:
            raise ValueError(
                f"Each distribution must be a frozen discrete or continuous type, got {type(dist)}"
            )
    return variables