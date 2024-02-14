from copy import deepcopy
from typing import Callable, Optional

import numpy as np
from scipy.special import comb as combine

from experiment_design.scorers import Scorer


def get_best_try(creator: Callable[[], np.ndarray], scorer: Scorer, steps: int) -> np.ndarray:
    steps = max(1, steps)
    best_score, best_doe = -np.inf, None
    for _ in range(steps):
        doe = creator()
        score = scorer(doe)
        if score > best_score:
            best_doe = doe
            best_score = score
    return best_doe


def get_available_column(num_variables: int, sample_size: int, switched_pairs: np.ndarray) -> int:
    max_combinations = combine(sample_size, 2)
    uniques, col_counts = np.unique(switched_pairs[:, 0], return_counts=True)
    uniques = set(uniques[col_counts >= max_combinations])
    possibles = [i_col for i_col in np.arange(num_variables) if i_col not in uniques]
    if possibles:
        return np.random.choice(possibles)
    return np.random.choice(list(range(num_variables)))


def get_available_row(sample_size: int, switched_pairs_in_columns: np.ndarray,
                       blocked_rows: Optional[list[int]] = None) -> int:
    blocked_rows = set(blocked_rows) if blocked_rows else {}
    possible_rows = [row for row in range(sample_size) if row not in blocked_rows]
    max_switches = sample_size - 1
    depleted, counts = np.unique(switched_pairs_in_columns.ravel(), return_counts=True)
    depleted = set(depleted[counts >= max_switches])
    possible_rows = [i_row for i_row in possible_rows if i_row not in depleted]
    if possible_rows:
        return np.random.choice(possible_rows)
    return np.random.choice(possible_rows)


def perturb_rows_along_column(matrix: np.ndarray, switched_pairs: list[tuple[int, int, int]],
                              column: Optional[int] = None) -> np.ndarray:
    """
    Randomly switches rows of a matrix along one column

    :param matrix:
    :param column: The index of column, along which the switching is done. If
        not given, it will be chosen randomly.
    :param switched_pairs:
    :return:
    """
    rows, columns = matrix.shape
    if switched_pairs:
        pairs = np.array(switched_pairs, dtype=int)
    else:
        pairs = np.empty((0, 3), dtype=int)

    column = get_available_column(columns, rows, pairs) if column is None else column
    pairs = pairs[pairs[:, 0] == column, 1:]
    row_1 = get_available_row(sample_size=rows, switched_pairs_in_columns=pairs, blocked_rows=None)
    row_2 = get_available_row(sample_size=rows, switched_pairs_in_columns=pairs, blocked_rows=[row_1])

    matrix[row_1, column], matrix[row_2, column] = matrix[row_2, column], matrix[row_1, column]
    switched_pairs.append((column, row_1, row_2))
    return matrix


"""
should be Part of scoring but was in optimize
 if doe_old is None:
        appender = appender_loc = lambda x: x
    else:
        locs = [doe_start.min(0, keepdims=True), doe_start.max(0, keepdims=True)]
        locs = np.logical_and((doe_old >= locs[0]).all(1),
                              (doe_old <= locs[1]).all(1))
        appender_loc = lambda x: np.append(doe_old[locs].reshape((-1, x.shape[1])), x, axis=0)
        appender = lambda x: np.append(doe_old, x, axis=0)  # will be used for calculating score
    
    dist_max = np.max(appender(doe_start), axis=0) - np.min(appender(doe_start), axis=0)
    dist_max = np.log(np.sqrt(np.sum(dist_max ** 2)))
"""


def simulated_annealing_by_perturbation(doe: np.ndarray, scorer: Scorer, steps: int = 1000, decay: float = .95,
                                        simulation_time: float = 25., max_steps_without_improvement: int = 25,
                                        verbose: int = 0) -> np.ndarray:
    if simulation_time <= 1e-16:
        raise ValueError("simulation_time must be strictly positive.")
    if not 0 <= decay <= 1:
        raise ValueError('decay has to be between 0 and 1.')
    if max_steps_without_improvement < 1:
        max_steps_without_improvement = 1

    best_doe = deepcopy(doe)

    best_score = -np.inf
    start_score = best_score
    max_possible_switch = doe.shape[1] * combine(doe.shape[0], 2)
    steps_without_improvement = 0
    switched_pairs = []
    old_switched_pairs = []
    for i_try in range(1, steps + 1):
        doe_try, pair = perturb_rows_along_column(doe, switched_pairs=switched_pairs)
        switched_pairs.append(pair)
        curr_score = scorer(doe_try)
        anneal_prob = np.exp(-(curr_score - start_score) / simulation_time)

        if curr_score >= start_score or np.random.random() <= anneal_prob:
            doe_start = deepcopy(doe_try)
            old_switched_pairs = deepcopy(switched_pairs)
            start_score = curr_score
            switched_pairs = []
            simulation_time *= decay
            if curr_score > best_score:
                best_doe = deepcopy(doe_start)
                best_score = start_score
                steps_without_improvement = 0
                if verbose:
                    print(f"{i_try} - score: {best_score:.4f}")
        steps_without_improvement += 1

        if steps_without_improvement >= max_steps_without_improvement:
            simulation_time *= decay
            # Bound Randomness by setting back to best result
            # This may help convergence
            doe = deepcopy(best_doe)
            switched_pairs = deepcopy(old_switched_pairs)
            start_score = best_score
            steps_without_improvement = 0

        if len(switched_pairs) >= max_possible_switch:
            break

    if verbose > 0:
        print(f"Final score: {best_score:.4f}")
    return best_doe
