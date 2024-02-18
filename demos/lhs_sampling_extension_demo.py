import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from experiment_design.orthogonal_sampling import OrthogonalSamplingDesigner
from experiment_design.variable import create_continuous_uniform_variables


def create_iterative_plot(
    step_does: list[np.ndarray],
    step_grids: list[np.ndarray],
    lower_bound: float,
    upper_bound: float,
):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = []
    for step, doe in enumerate(step_does):
        scatter_plot = ax.scatter(doe[:, 0], doe[:, 1], label=f"Step {step}")
        colors.append(scatter_plot.get_facecolors()[0])

    for color, grid in zip(colors[::-1], grids[::-1]):
        ax.vlines(grid, lower_bound, upper_bound, color=color)
        ax.hlines(grid, lower_bound, upper_bound, color=color)
    plt.legend()
    return fig, ax


if __name__ == "__main__":
    np.random.seed(666)  # set seed to make run reproducible
    # initial doe
    sample_size = 4
    lb, ub = -2, 2
    variables = create_continuous_uniform_variables([lb, lb], [ub, ub])
    cr = OrthogonalSamplingDesigner(inter_bin_randomness=0.8)
    does = [cr.design(variables, sample_size, steps=1000, verbose=2)]
    grids = [np.linspace(lb, ub, sample_size + 1)]

    create_iterative_plot(does, grids, lb, ub)

    for i_step in range(1, 4):
        old_sample = np.concatenate(does, axis=0)
        new_doe = cr.design(
            variables, sample_size, steps=1000, old_sample=old_sample, verbose=2
        )
        does.append(new_doe)
        # double the sample size each step so that LHS constraint can always be fulfilled
        sample_size *= 2
        new_grid = np.linspace(lb, ub, sample_size + 1)
        grids.append(new_grid)
        new_sample = np.append(old_sample, new_doe, axis=0)
        corr_error = np.max(np.abs(np.corrcoef(new_sample, rowvar=False) - np.eye(2)))
        min_dist = np.min(pdist(new_doe))
        title = f"Max. absolute correlation error: {corr_error:.3f} Min. pairwise distance: {min_dist:.3f}"
        fig, ax = create_iterative_plot(does, grids, lb, ub)
        ax.set_title(title)

    plt.show()
