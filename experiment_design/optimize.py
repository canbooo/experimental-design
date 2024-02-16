import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.special import comb as combine

from experiment_design.scorers import Scorer


def get_best_try(
    creator: Callable[[], np.ndarray], scorer: Scorer, steps: int
) -> np.ndarray:
    steps = max(1, steps)
    best_score, best_doe = -np.inf, None
    for _ in range(steps):
        doe = creator()
        score = scorer(doe)
        if score > best_score:
            best_doe = doe
            best_score = score
    return best_doe


@dataclass
class MatrixRowSwitchCache:
    row_size: int
    column_size: int
    _max_switches_per_column: int = field(init=False, repr=False, default=None)
    _cache: list[np.ndarray] = field(init=False, repr=False, default=None)
    _cache_sizes: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # We switch two rows, thus the magic number
        self._max_switches_per_column = combine(self.row_size, 2)
        self.reset()

    @property
    def is_full(self) -> bool:
        return np.min(self._cache_sizes) >= self._max_switches_per_column

    def reset(self) -> None:
        self._cache_sizes = np.zeros(self.column_size, dtype=int)
        self._cache = [np.zeros((0, 2), dtype=int) for _ in range(self.column_size)]

    def choose_column(self) -> int:
        return np.random.choice(
            np.where(self._cache_sizes < self._max_switches_per_column)[0]
        )

    def choose_rows(self, column: int) -> tuple[int, int]:
        def choose_non_occupied(occupied: set[int]) -> int:
            return np.random.choice(
                [idx for idx in range(self.row_size) if idx not in occupied]
            )

        row_cache = self._cache[column]
        max_switches_per_row = self.row_size - 1
        fulls, counts = np.unique(row_cache, return_counts=True)
        row_1 = choose_non_occupied(set(fulls[counts >= max_switches_per_row]))
        row_2 = choose_non_occupied(
            set(row_cache[np.any(row_cache == row_1, axis=1)].ravel())
        )
        return row_1, row_2

    def cache(self, column: int, row_1: int, row_2: int) -> None:
        if np.any(np.all(self._cache[column] == np.array([[row_1, row_2]]), axis=1)):
            # TODO: REMOVE THIS
            raise RuntimeError("Unexpected cache invalidation")
        self._cache[column] = np.append(
            self._cache[column], np.array([[row_1, row_2]]), axis=0
        )
        self._cache_sizes[column] += 1

    def switch_rows_and_cache(self, matrix: np.ndarray) -> np.ndarray:
        column = self.choose_column()
        row_1, row_2 = self.choose_rows(column)
        self.cache(column, row_1, row_2)
        matrix[row_1, column], matrix[row_2, column] = (
            matrix[row_2, column],
            matrix[row_1, column],
        )
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


def simulated_annealing_by_perturbation(
    doe: np.ndarray,
    scorer: Scorer,
    steps: int = 1000,
    cooling_rate: float = 0.95,
    temperature: float = 25.0,
    max_steps_without_improvement: int = 25,
    verbose: int = 2,
) -> np.ndarray:
    if temperature <= 1e-16:
        raise ValueError("temperature must be strictly positive.")
    if not 0 <= cooling_rate <= 1:
        raise ValueError("decay has to be between 0 and 1.")

    def cool_down_temperature(temp: float, min_temperature: float = 1e-6) -> float:
        return max(temp * cooling_rate, min_temperature)

    if max_steps_without_improvement < 1:
        max_steps_without_improvement = 1

    doe_start = doe.copy()
    best_doe = doe.copy()
    start_score = anneal_step_score = best_score = scorer(doe)

    steps_without_improvement = 0
    switch_cache = MatrixRowSwitchCache(row_size=doe.shape[0], column_size=doe.shape[1])
    old_switch_cache = MatrixRowSwitchCache(
        row_size=doe.shape[0], column_size=doe.shape[1]
    )
    for i_try in range(1, steps + 1):
        doe_try = switch_cache.switch_rows_and_cache(doe_start.copy())
        curr_score = scorer(doe_try)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow")
            transition_probability = np.exp(
                -(anneal_step_score - curr_score) / temperature
            )

        if (
            curr_score >= anneal_step_score
            or np.random.random() <= transition_probability
        ):
            doe_start = doe_try.copy()
            old_switch_cache = deepcopy(switch_cache)
            anneal_step_score = curr_score
            switch_cache.reset()
            temperature = cool_down_temperature(temperature)
            if curr_score > best_score:
                best_doe = doe_try.copy()
                best_score = curr_score
                steps_without_improvement = 0
                if verbose > 1:
                    print(
                        f"{i_try} - start score improved by {100 * (best_score - start_score) / abs(start_score):.1f} %"
                    )
        steps_without_improvement += 1

        if steps_without_improvement >= max_steps_without_improvement:
            temperature = cool_down_temperature(temperature)
            # Bound Randomness by setting back to best result
            # This often accelerates convergence speed
            doe_start = best_doe.copy()
            switch_cache = deepcopy(old_switch_cache)
            anneal_step_score = best_score
            steps_without_improvement = 0

        if switch_cache.is_full:
            if verbose > 0:
                print("No more perturbation left to improve the score")
            break

    if verbose > 0:
        print(
            f"Final score improved start score by {100 * (best_score - start_score) / abs(start_score):.1f} %"
        )
    return best_doe
