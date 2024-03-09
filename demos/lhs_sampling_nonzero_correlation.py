from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from scipy.spatial.distance import pdist

from experiment_design.orthogonal_sampling import OrthogonalSamplingDesigner
from experiment_design.scorers import (
    create_correlation_matrix,
    create_default_scorer_factory,
)
from experiment_design.variable import create_continuous_uniform_variables


def create_iterative_plot(
    step_does: list[np.ndarray],
    step_grids: Optional[list[np.ndarray]] = None,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = []
    for step, doe in enumerate(step_does):
        alpha = 1.0 if step == len(step_does) - 1 else 0.2
        scatter_plot = ax.scatter(doe[:, 0], doe[:, 1], alpha=alpha)
        colors.append(scatter_plot.get_facecolors()[0])

    if step_grids is None:
        return fig, ax

    for color, grid in zip(colors[::-1], step_grids[::-1]):
        ax.vlines(grid, grid.min(), grid.max(), color=color)
        ax.hlines(grid, grid.min(), grid.max(), color=color)

    return fig, ax


def create_title(doe, target_corr):
    target_corr_mat = create_correlation_matrix(
        target_correlation=target_corr, num_variables=doe.shape[1]
    )
    corr_error = np.max(np.abs(np.corrcoef(doe, rowvar=False) - target_corr_mat))
    min_dist = np.min(pdist(doe))
    return f"Target correlation: {target_corr} Max. correlation error: {corr_error:.3f}"


if __name__ == "__main__":
    np.random.seed(42)  # set seed to make run reproducible
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
    sample_size = 32
    lb, ub = -2, 2
    variables = create_continuous_uniform_variables([lb, lb], [ub, ub])
    correlations = [-0.9, -0.5, 0.5, 0.9]

    does, grids = [], []

    for i_step, corr in enumerate(correlations):
        designer = OrthogonalSamplingDesigner(
            scorer_factory=create_default_scorer_factory(
                correlation_score_weight=0.9,
                distance_score_weight=0.1,
                target_correlation=corr,
            )
        )
        new_doe = designer.design(variables, sample_size, steps=1000, verbose=2)
        does.append(new_doe)
        new_grid = np.linspace(lb, ub, new_doe.shape[0] + 1)
        grids.append(new_grid)
        old_sample = np.concatenate(does, axis=0)
        title = create_title(new_doe, corr)
        fig, ax = create_iterative_plot(does, step_grids=grids)
        ax.set_title(title)
        plt.axis("off")
        plt.savefig(f"../media/lhs_correlation_{i_step}.png", bbox_inches="tight")

    plt.show()
