[![tests](https://github.com/canbooo/experimental-design/actions/workflows/tests.yml/badge.svg)](https://github.com/canbooo/experimental-design/actions/workflows/tests.yml)

# `experiment-design`: Tools to create and extend experiment plans

Documentation is under construction. You can install using

`pip install experiment-design`

See  [demos](./demos) for example usage.

## Create and extend Latin hypercube designs

![LHS extension by doubling](./media/lhs_extension_by_doubling.gif)

![LHS extension one by one](./media/lhs_extension_by_constant.gif)

![Local LHS extension](./media/lhs_extension_local.gif)

## Orthogonal designs with any[^1] distribution

![OS extension by doubling](./media/os_extension_by_doubling.gif)

[^1]: As long as it is supported by `scipy.stats`

## Create and extend correlated designs

![Correlated LHS](./media/lhs_correlation.gif)
