import pytest
import numpy as np
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
        var = module_under_test.ContinuousVariable(distribution=stats.norm(0, 1),
                                 lower_bound=-5)
        assert not np.isfinite(var.value_of(1.))
        assert var.value_of(0) == -5
        assert var.value_of(0.5) == 0


class TestDiscreteVariable:

    def test_fail_distribution(self):
        with pytest.raises(ValueError):
            module_under_test.DiscreteVariable(distribution=stats.uniform(0, 1))

    def test_value_of(self):
        var = module_under_test.DiscreteVariable(distribution=stats.bernoulli(0.5))
        assert var.value_of(1e-6) == 0 # 0 return -1 for both bernoulli and randint
        assert var.value_of(1) == 1
        assert var.distribution.dist.name == "bernoulli"

    def test_value_of_with_mapper(self):
        values = ["Off", "On"]
        var = module_under_test.DiscreteVariable(distribution=stats.bernoulli(0.5),
                                                 value_mapper=lambda x: values[int(x)])
        assert var.value_of(1e-6) == "Off"
        assert var.value_of(1) == "On"
        assert np.all(var.value_of(np.array([1e-6, 1])) == np.array(values))
