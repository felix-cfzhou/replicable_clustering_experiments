from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

from samplers import AbstractSampler
from coreset import r_coreset


def plot2D(
    data: np.array,
    ax: Optional[Axes] = None,
    *args,
    **kwargs,
):
    assert len(data.shape) == 2
    assert data.shape[1] == 2

    if ax is None:
        ax = plt.gca()

    ax.scatter(data[:, 0], data[:, 1], *args, **kwargs)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    
    
class MemorizedSampler(AbstractSampler):
    def __init__(self, sampler, random_state: Union[RandomState, int, None] = None):
        super().__init__(random_state)
        self.sampler = sampler
        self.samples = []

    def __call__(self, size: int):
        samples = self.sampler(size)
        self.samples.append(samples.copy())
        return samples

    def get_samples(self):
        return np.concatenate(self.samples, axis=0)
    
    

def compare_plot_kmeans(
    sampler: AbstractSampler, random_seed: int, fig: Figure, axes: Axes
):
    for idx in range(2):
        memorized_sampler = MemorizedSampler(sampler)
        random_state = RandomState(random_seed)
        coreset, mass = r_coreset(
            memorized_sampler,
            k=3,
            eps=0.99,
            rho=0.3,
            delta=0.01,
            Gamma=0.5,
            beta=1.0,
            Delta=np.sqrt(2),
            random_state=random_state,  # shared internal random seed
        )
        # print(coreset, mass)
        # return
        data = memorized_sampler.get_samples()
        indices = random_state.choice(data.shape[0], 10000, replace=False)

        kmeans = KMeans(n_clusters=3).fit(data)
        rKmeans = KMeans(n_clusters=3).fit(coreset, sample_weight=mass)

        plot2D(data[indices], axes[idx], s=16.0, c="tab:blue", label="Samples")
        plot2D(
            kmeans.cluster_centers_,
            axes[idx],
            s=64.0,
            c="tab:orange",
            label="Non-Replicable Centers",
        )
        plot2D(
            rKmeans.cluster_centers_,
            axes[idx],
            s=64.0,
            c="tab:red",
            label="Replicable Centers",
        )
        plot2D(
            coreset,
            axes[idx],
            s=64.0,
            # linewidth=5,
            c="tab:olive",
            marker="1",
            label="Coreset",
        )

        axes[idx].set_xlim(-1, 1)
        axes[idx].set_ylim(-1, 1)
        axes[idx].set_title(f"Execution {idx+1}")
        if idx == 1:
            axes[idx].legend(loc="center left", bbox_to_anchor=(1, 0.5))