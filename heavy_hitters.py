from typing import Union

import numpy as np
from numpy.random import RandomState

from samplers import AbstractSampler


# 1-dimensional
def r_heavy_hitters(
    sampler: AbstractSampler,
    thres: float,
    eps: float,
    rho: float,
    delta: float,
    random_state: Union[RandomState, int, None] = None,
):
    # print("r_heavy_hitters...")
    assert 0 < thres < 1
    assert 0 < eps < thres
    assert 0 < rho < 1
    assert 0 < delta <= rho / 3

    if not isinstance(random_state, RandomState):
        random_state = RandomState(random_state)

    n1 = int(np.ceil(np.log(2 / (delta * (thres - eps))) / (thres - eps)))
    # print(n1)
    candidates = sampler(size=n1)
    candidates = np.unique(candidates, axis=0)

    n2 = int(
        np.ceil(
            (np.log(2 / delta) + (np.sqrt(n1) + 1) * np.log(2))  # * 648
            / (rho**2 * eps**2)
        )
    )
    # print(n2)
    samples = sampler(size=n2)

    unique_samples, count = np.unique(samples, axis=0, return_counts=True)
    count = count.astype(float) / n2

    rand_thres = random_state.uniform(thres - 2 * eps / 3, thres - eps / 3)
    # print(count, rand_thres, n1 + n2)

    _, idx_intersect, _ = np.intersect1d(
        unique_samples, candidates, return_indices=True
    )

    unique_samples_intersect = unique_samples[idx_intersect]
    count_intersect = count[idx_intersect]
    return unique_samples_intersect[count_intersect >= rand_thres]