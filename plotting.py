from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.random import RandomState
from sklearn.cluster import KMeans

from coreset import r_coreset
from samplers import AbstractSampler


def plot2D(
    data: np.array,
    ax: Optional[Axes] = None,
    transpose: bool = False,
    *args,
    **kwargs,
):
    assert len(data.shape) == 2
    assert data.shape[1] == 2

    if ax is None:
        ax = plt.gca()

    if transpose:
        ax.scatter(data[:, 1], data[:, 0], *args, **kwargs)
    else:
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

        kmeans = KMeans(n_clusters=3, random_state=random_state).fit(data)
        rKmeans = KMeans(n_clusters=3, random_state=random_state).fit(
            coreset, sample_weight=mass
        )

        plot2D(data[indices], axes[idx], transpose=True, s=16.0, c="tab:cyan", label="Samples")
        plot2D(
            kmeans.cluster_centers_,
            axes[idx],
            transpose=True,
            s=80.0,
            c="tab:orange",
            edgecolors='black',
            marker="^",
            label="Non-Replicable Centers",
        )
        plot2D(
            rKmeans.cluster_centers_,
            axes[idx],
            transpose=True,
            s=80.0,
            c="tab:red",
            edgecolors='black',
            marker="s",
            label="Replicable Centers",
        )
        plot2D(
            coreset,
            axes[idx],
            transpose=True,
            s=40.0,
            # linewidth=5,
            c="tab:green",
            edgecolors='black',
            marker="P",
            label="Coreset",
        )

        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].set_xlim(-1, 1)
        axes[idx].set_ylim(-1, 1)
        axes[idx].set_title(
            f"  Execution {idx+1}",
            loc="left",
            y=1.0, 
            pad=-14,
        )
        if idx == 1:
            axes[idx].legend(
                loc="lower right",
                # bbox_to_anchor=(1, 0.5)
            )
