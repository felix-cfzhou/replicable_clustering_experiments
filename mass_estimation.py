from typing import Union

import numpy as np
from numpy.random import RandomState

from samplers import AbstractSampler


# eps should be a power of 10 e.g. 1e-3
def r_prob_mass(
    sampler: AbstractSampler, 
    N: int, 
    rho: float, 
    eps: float, 
    delta: float, 
    random_state: Union[RandomState, int, None] = None,
):
    assert 0 < rho < 1
    assert 0 < eps < 1
    assert 0 < delta < rho / 3
    
    if not isinstance(random_state, RandomState):
        random_state = RandomState(random_state)

    alpha = 2 * eps / (rho - 2 * delta + 1)
    eps_prime = eps * (rho - 2 * delta) / (rho + 1 - 2 * delta)
    n = (
        int(
            np.ceil(
                (np.log(1 / delta) + N * np.log(2))
                / (eps**2 * (rho - 2 * delta) ** 2)
            )
        )
        * 2
    )
    decimals = int(np.log10(1 / eps))
    # print(alpha, eps_prime, n, decimals)

    samples = sampler(size=n)
    unique_samples, count = np.unique(samples, axis=0, return_counts=True)
    # print(unique_samples, count)
    len(unique_samples) <= N
    count = count.astype(float) / n

    offset = random_state.uniform(low=0.0, high=alpha, size=len(unique_samples))
    rounded_count = np.around(count - offset, decimals=decimals) + offset
    normalized_count = rounded_count - ((rounded_count.sum() - 1.0) / len(unique_samples))
    np.clip(normalized_count, 0.0, 1.0, out=normalized_count) # clip values

    return unique_samples, normalized_count