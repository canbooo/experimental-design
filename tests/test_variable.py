import numpy as np
import pytest
from scipy import stats

import experiment_design.variable as module_under_test


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

    def test_value_of_from_dist(self):
        var = module_under_test.ContinuousVariable(distribution=stats.norm(0, 1))
        assert not np.isfinite(var.value_of(np.arange(2))).any()
        assert var.value_of(0.5) == 0
        assert var.distribution.dist.name == "norm"

    def test_value_of_from_dist_and_bound(self):
        var = module_under_test.ContinuousVariable(
            distribution=stats.norm(0, 1), lower_bound=-5
        )
        assert not np.isfinite(var.value_of(1.0))
        assert var.value_of(0) == -5
        assert var.value_of(0.5) == 0


class TestDiscreteVariable:

    def test_fail_distribution(self):
        with pytest.raises(ValueError):
            module_under_test.DiscreteVariable(distribution=stats.uniform(0, 1))

    def test_value_of(self):
        var = module_under_test.DiscreteVariable(distribution=stats.bernoulli(0.5))
        assert var.value_of(1e-6) == 0  # 0 return -1 for both bernoulli and randint
        assert var.value_of(1) == 1
        assert var.distribution.dist.name == "bernoulli"

    def test_value_of_with_mapper(self):
        values = [42, 666
                  ] # In general, these should be sorted in ascending order
        var = module_under_test.DiscreteVariable(
            distribution=stats.bernoulli(0.5), value_mapper=lambda x: values[int(x)]
        )
        assert var.value_of(1e-6) == 42
        assert var.value_of(1) == 666
        assert np.all(var.value_of(np.array([1e-6, 1])) == np.array(values))


def test_create_continuous_discrete_uniform_variables():
    variables = module_under_test.create_discrete_uniform_variables([[1, 2], [3, 4, 5], [9, 8]])
    probabilities = np.array([1e-6, 0.6, 1])
    expected = np.array([[1, 2, 2], [3, 4, 5], [8, 9, 9]])

    result = np.array([var.value_of(probabilities) for var in variables])
    assert np.all(expected == result)


def test_create_continuous_uniform_variables():
    variables = module_under_test.create_continuous_uniform_variables([1, 42, 665], [3, 52, 667])
    probabilities = np.array([0, 0.5, 1])
    expected = np.array([[1, 2, 3], [42, 47, 52], [665, 666, 667]])

    result = np.array([var.value_of(probabilities) for var in variables])
    assert np.all(expected == result)
