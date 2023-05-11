from typing import Union

import numpy as np
from numpy.random import RandomState

from mass_estimation import r_prob_mass
from quad_tree import QuadTreeSampler, r_quad_tree


def r_coreset(
    sampler,
    k: int,
    eps: float,
    rho: float,
    delta: float,
    Gamma: float,
    beta: float,
    Delta: float = np.sqrt(2),
    skip_layers: int = 1,
    random_state: Union[RandomState, int, None] = None,
):
    assert 0 < eps < 1
    assert 0 < rho < 1
    assert 0 < delta < rho / 3

    if not isinstance(random_state, RandomState):
        random_state = RandomState(random_state)

    root = r_quad_tree(
        sampler,
        k=k,
        eps=eps,
        rho=rho,
        delta=delta,
        Gamma=Gamma,
        beta=beta,
        Delta=Delta,
        skip_layers=skip_layers,
        random_state=random_state,
    )

    N = len(root.get_leaves())

    quad_tree_sampler = QuadTreeSampler(sampler, root)
    coreset, mass = r_prob_mass(
        quad_tree_sampler,
        N=N,
        rho=rho,
        eps=eps / 10,
        delta=delta,
        random_state=random_state,
    )

    return coreset, mass
