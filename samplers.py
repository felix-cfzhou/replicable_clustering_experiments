from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.random import RandomState
from scipy.stats import truncnorm
from sklearn import datasets


class AbstractSampler(ABC):
    def __init__(self, random_state: Union[RandomState, int, None] = None) -> None:
        if not isinstance(random_state, RandomState):
            random_state = RandomState(random_state)
        self.random_state = random_state

    @abstractmethod
    def __call__(self, size: int) -> np.array:
        pass


class MixtureTruncNormSampler(AbstractSampler):
    def __call__(self, size: int) -> np.array:
        scale = 0.1

        loc = -0.6
        a, b = (-1.0 - loc) / scale, (1.0 - loc) / scale
        pos_samples = truncnorm.rvs(
            a, b, loc=loc, scale=scale, size=2 * size, random_state=self.random_state
        )

        loc = 0.4
        a, b = (-1.0 - loc) / scale, (1.0 - loc) / scale
        neg_samples = truncnorm.rvs(
            a, b, loc=loc, scale=scale, size=2 * size, random_state=self.random_state
        )

        all_samples = np.concatenate([pos_samples, neg_samples], axis=None)
        self.random_state.shuffle(all_samples)

        return all_samples.reshape((-1, 2))[:size]


class MoonsSampler(AbstractSampler):
    def __call__(self, size: int) -> np.array:
        samples, _ = datasets.make_moons(
            n_samples=size, shuffle=True, noise=0.02, random_state=self.random_state
        )
        samples[:, 0] -= 0.5
        samples[:, 1] -= 0.2
        samples[:, 0] /= 2.0

        return samples
