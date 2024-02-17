import matplotlib.pyplot as plt
import numpy as np

from experiment_design.orthogonal_sampling import OrthogonalSamplingDesigner
from experiment_design.variable import create_continuous_uniform_variables

# np.random.seed(666)
sample_size = 8
lb, ub = -2, 2
vars = create_continuous_uniform_variables([lb, lb], [ub, ub])
cr = OrthogonalSamplingDesigner(central_design=False)
doe = cr.design(vars, sample_size, steps=250)

new_doe = cr.design(vars, sample_size, steps=250, old_sample=doe)
grid = np.linspace(lb, ub, sample_size * 2 + 1)

fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(doe[:, 0], doe[:, 1])
ax.scatter(new_doe[:, 0], new_doe[:, 1])

full_fact = grid
ax.vlines(grid, lb, ub)
ax.hlines(grid, lb, ub)
plt.show()
