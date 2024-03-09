from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from scipy import stats
from scipy.spatial.distance import pdist

from experiment_design.orthogonal_sampling import OrthogonalSamplingDesigner
from experiment_design.variable import create_variables_from_distributions


def create_iterative_plot(
    step_does: list[np.ndarray],
    step_grids: Optional[list[np.ndarray]] = None,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = []
    for step, doe in enumerate(step_does):
        scatter_plot = ax.scatter(doe[:, 0], doe[:, 1], label=f"Step {step}")
        colors.append(scatter_plot.get_facecolors()[0])
    plt.legend()
    if step_grids is None:
        return fig, ax

    for color, grid in zip(colors[::-1], step_grids[::-1]):
        ax.vlines(grid, grid.min(), grid.max(), color=color)
        ax.hlines(grid, grid.min(), grid.max(), color=color)

    return fig, ax


def create_title(doe):
    corr_error = np.max(np.abs(np.corrcoef(doe, rowvar=False) - np.eye(2)))
    min_dist = np.min(pdist(doe))
    return f"Max. correlation error: {corr_error:.3f} Min. pairwise distance: {min_dist:.3f}"


if __name__ == "__main__":
    np.random.seed(666)  # set seed to make run reproducible
    mpl.rcParams["axes.prop_cycle"] = cycler(
        color=[
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
            "#999999",
            "#e41a1c",
            "#dede00",
        ]
    )

    # double the sample size each step so that LHS constraint can always be fulfilled
    start_sample_size = 4

    variables = create_variables_from_distributions([stats.norm() for _ in range(2)])
    designer = OrthogonalSamplingDesigner(inter_bin_randomness=0.8)

    does, grids = [], []
    old_sample = None
    for i_step in range(4):
        sample_size = max(start_sample_size * 2 ** (i_step - 1), start_sample_size)
        new_sample = designer.design(
            variables, sample_size, steps=1000, old_sample=old_sample, verbose=2
        )
        does.append(new_sample)

        if old_sample is None:
            new_doe = new_sample
        else:
            new_doe = np.append(old_sample, new_sample, axis=0)
        delta = 5e-3
        new_grid = stats.norm().ppf(np.linspace(delta, 1 - delta, new_doe.shape[0] + 1))
        grids.append(new_grid)
        old_sample = np.concatenate(does, axis=0)
        title = create_title(new_doe)
        fig, ax = create_iterative_plot(does, step_grids=grids)
        ax.set_title(title)
        plt.axis("off")
        plt.savefig(
            f"../media/os_extension_by_doubling_{i_step}.png", bbox_inches="tight"
        )

    plt.show()
