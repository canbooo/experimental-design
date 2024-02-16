import numpy as np
import pytest
from scipy import stats

import experiment_design.variable as module_under_test
import experiment_design.variable.space


@pytest.fixture
def discrete_bernoulli() -> module_under_test.DiscreteVariable:
    values = [42, 666]
    return module_under_test.DiscreteVariable(
        distribution=stats.bernoulli(0.5), value_mapper=lambda x: values[int(x)]
    )


@pytest.fixture
def standard_normal() -> module_under_test.ContinuousVariable:
    return module_under_test.ContinuousVariable(distribution=stats.norm(0, 1))


@pytest.fixture(
    params=(
        [stats.uniform(-2, 4) for _ in range(2)],
        [stats.uniform(-2, 4), stats.bernoulli(0.8)],
        [stats.bernoulli(0.8) for _ in range(2)],
    ),
    ids=("Continuous", "Mixed", "Discrete"),
)
def design_space(request) -> experiment_design.variable.space.DesignSpace:
    variables = module_under_test.create_variables_from_distributions(request.param)
    return experiment_design.variable.space.DesignSpace(variables)


def test_is_frozen_discrete():
    assert module_under_test.is_frozen_discrete(stats.uniform()) is False
    assert module_under_test.is_frozen_discrete(stats.bernoulli) is False
    assert module_under_test.is_frozen_discrete(stats.bernoulli(0.5)) is True


def test_is_frozen_continuous():
    assert module_under_test.is_frozen_continuous(stats.bernoulli(0.5)) is False
    assert module_under_test.is_frozen_continuous(stats.uniform) is False
    assert module_under_test.is_frozen_continuous(stats.uniform()) is True


def test_create_continuous_discrete_uniform_variables():
    variables = module_under_test.create_discrete_uniform_variables(
        [[1, 2], [3, 4, 5], [9, 8]]
    )
    probabilities = np.array([1e-6, 0.6, 1])
    expected = np.array([[1, 2, 2], [3, 4, 5], [8, 9, 9]])

    result = np.array([var.value_of(probabilities) for var in variables])
    assert np.all(expected == result)


def test_create_continuous_uniform_variables():
    variables = module_under_test.create_continuous_uniform_variables(
        [1, 42, 665], [3, 52, 667]
    )
    probabilities = np.array([0, 0.5, 1])
    expected = np.array([[1, 2, 3], [42, 47, 52], [665, 666, 667]])

    result = np.array([var.value_of(probabilities) for var in variables])
    assert np.all(expected == result)


def test_create_variables_from_distributions():
    distributions = [stats.uniform(0, 1), stats.bernoulli(0.5)]
    variables = module_under_test.create_variables_from_distributions(distributions)
    assert isinstance(variables[0], module_under_test.ContinuousVariable)
    assert isinstance(variables[1], module_under_test.DiscreteVariable)


class TestContinuousVariable:
    def test_fail_distribution(self):
        with pytest.raises(ValueError):
            module_under_test.ContinuousVariable(distribution=stats.bernoulli(0.5))

    def test_fail_ambiguous_definition(self):
        with pytest.raises(ValueError):
            module_under_test.ContinuousVariable()

        with pytest.raises(ValueError):
            module_under_test.ContinuousVariable(lower_bound=0)

    def test_fail_invalid_bounds(self):
        with pytest.raises(ValueError):
            module_under_test.ContinuousVariable(lower_bound=0, upper_bound=-1)

    def test_value_of_from_bounds(self):
        var = module_under_test.ContinuousVariable(lower_bound=-1, upper_bound=1)
        assert var.value_of(0) == -1
        assert var.value_of(1) == 1
        assert var.distribution.dist.name == "uniform"

    def test_value_of_from_dist(
        self, standard_normal: module_under_test.ContinuousVariable
    ):
        assert not np.isfinite(standard_normal.value_of(np.arange(2))).any()
        assert standard_normal.value_of(0.5) == 0
        assert standard_normal.distribution.dist.name == "norm"

    def test_value_of_from_dist_and_bound(
        self, standard_normal: module_under_test.ContinuousVariable
    ):
        standard_normal.lower_bound = -5
        assert not np.isfinite(standard_normal.value_of(1.0))
        assert standard_normal.value_of(0) == -5
        assert standard_normal.value_of(0.5) == 0

    def test_finite_lower_bound_given(
        self, standard_normal: module_under_test.ContinuousVariable
    ):
        standard_normal.lower_bound = -5
        assert standard_normal.get_finite_lower_bound() == -5

    def test_finite_lower_bound_finite(self):
        var = module_under_test.ContinuousVariable(distribution=stats.uniform(0, 1))
        assert var.get_finite_lower_bound() == 0

    def test_finite_lower_bound_infinite(
        self, standard_normal: module_under_test.ContinuousVariable
    ):
        tol = 2.5e-2
        assert np.isclose(
            standard_normal.get_finite_lower_bound(
                infinite_support_probability_tolerance=tol
            ),
            -1.95996,
        )

    def test_finite_upper_bound_given(
        self, standard_normal: module_under_test.ContinuousVariable
    ):
        standard_normal.upper_bound = 5
        assert standard_normal.get_finite_upper_bound() == 5

    def test_finite_upper_bound_finite(self):
        var = module_under_test.ContinuousVariable(distribution=stats.uniform(0, 1))
        assert var.get_finite_upper_bound() == 1

    def test_finite_upper_bound_infinite(
        self, standard_normal: module_under_test.ContinuousVariable
    ):
        tol = 2.5e-2
        assert np.isclose(
            standard_normal.get_finite_upper_bound(
                infinite_support_probability_tolerance=tol
            ),
            1.95996,
        )


class TestDiscreteVariable:
    def test_fail_distribution(self):
        with pytest.raises(ValueError):
            module_under_test.DiscreteVariable(distribution=stats.uniform(0, 1))

    def test_value_of(self):
        var = module_under_test.DiscreteVariable(distribution=stats.bernoulli(0.5))
        assert var.value_of(1e-6) == 0  # 0 return -1 for both bernoulli and randint
        assert var.value_of(1) == 1
        assert var.distribution.dist.name == "bernoulli"

    def test_value_of_with_mapper(
        self, discrete_bernoulli: module_under_test.DiscreteVariable
    ):
        assert discrete_bernoulli.value_of(1e-6) == 42
        assert discrete_bernoulli.value_of(1) == 666
        assert np.all(
            discrete_bernoulli.value_of(np.array([1e-6, 1])) == np.array([42, 666])
        )

    def test_get_finite_lower_bound(
        self, discrete_bernoulli: module_under_test.DiscreteVariable
    ):
        assert discrete_bernoulli.get_finite_lower_bound() == 42

    def test_get_finite_upper_bound(
        self, discrete_bernoulli: module_under_test.DiscreteVariable
    ):
        assert discrete_bernoulli.get_finite_upper_bound() == 666


class TestDesignSpace:

    def test_value_of(self, design_space: experiment_design.variable.space.DesignSpace):
        probabilities = np.array([[0.1, 0.1], [0.5, 0.6], [0.9, 0.8]])
        if isinstance(design_space.variables[0], module_under_test.DiscreteVariable):
            # Both are discrete
            expected = np.array([[0, 0], [1, 1], [1, 1]])
        elif isinstance(design_space.variables[1], module_under_test.DiscreteVariable):
            # Mixed case
            expected = np.array([[-1.6, 0], [0, 1], [1.6, 1]])
        else:
            # Both are continuous
            expected = np.array([[-1.6, -1.6], [0, 0.4], [1.6, 1.2]])
        assert np.all(np.isclose(design_space.value_of(probabilities), expected))

    def test_design_space_size_getters(
        self, design_space: experiment_design.variable.space.DesignSpace
    ):
        assert design_space.dimensions == 2
        assert len(design_space) == 2

    def test_design_space_lower_bound(
        self, design_space: experiment_design.variable.space.DesignSpace
    ):
        if isinstance(design_space.variables[0], module_under_test.DiscreteVariable):
            # Both are discrete
            expected = np.array([0, 0])
        elif isinstance(design_space.variables[1], module_under_test.DiscreteVariable):
            # Mixed case
            expected = np.array([-2, 0])
        else:
            # Both are continuous
            expected = np.array([-2, -2])
        assert all(design_space.lower_bound == expected)

    def test_design_space_upper_bound(
        self, design_space: experiment_design.variable.space.DesignSpace
    ):
        if isinstance(design_space.variables[0], module_under_test.DiscreteVariable):
            # Both are discrete
            expected = np.array([1, 1])
        elif isinstance(design_space.variables[1], module_under_test.DiscreteVariable):
            # Mixed case
            expected = np.array([2, 1])
        else:
            # Both are continuous
            expected = np.array([2, 2])
        assert all(design_space.upper_bound == expected)
