from itertools import product
from typing import List, Union

import numpy as np
from numpy.random import RandomState

from heavy_hitters import r_heavy_hitters
from samplers import AbstractSampler


class QuadTreeNode:
    offsets = [np.array([dx, dy]) for dx, dy in product([-1.0, 1.0], repeat=2)]

    def __init__(
        self, point: np.array, radius: float, is_heavy: bool = False, parent=None
    ):
        self.point = point
        self.radius = radius
        self.is_heavy = is_heavy
        self.children = [None] * 4
        self.parent = parent

    def get_heavy_nodes(self):
        heavy_nodes = []

        def _explore(node):
            if not node.is_heavy:
                return

            heavy_nodes.append(node.point.reshape((1, -1)))
            for child in node.children:
                _explore(child)

        _explore(self)
        return np.concatenate(heavy_nodes, axis=0)

    def get_leaves(self):
        leaves = []

        # return true if found node
        def _explore(node) -> bool:
            has_heavy_child = False
            for child in node.children:
                if child is not None and child.is_heavy:
                    _explore(child)
                    has_heavy_child = True

            if not has_heavy_child:
                leaves.append(node.point.reshape((1, -1)))

            return has_heavy_child

        _explore(self)
        return np.concatenate(leaves, axis=0)

    def get_child_idx(self, point):
        if point[0] < self.point[0]:
            if point[1] < self.point[1]:
                return 0
            else:
                return 1
        else:
            if point[1] < self.point[1]:
                return 2
            else:
                return 3

    def quad_tree_round(self, point):
        output = np.array([0.0, 0.0])

        node = self
        while node is not None:
            child_idx = node.get_child_idx(point)

            if (
                node.children[child_idx] is not None
                and node.children[child_idx].is_heavy
            ):
                node = node.children[child_idx]
                output = node.point
                continue

            new_node = None
            for idx in range(len(QuadTreeNode.offsets)):
                if node.children[idx] is not None and node.children[idx].is_heavy:
                    new_node = node.children[idx]
                    output = new_node.point
                    break

            node = new_node

        return output

    @staticmethod
    def make_children(nodes):
        child_nodes = []
        for node in nodes:
            radius = node.radius
            for idx, d in enumerate(QuadTreeNode.offsets):
                next_point = node.point + d * radius / 2
                child_node = QuadTreeNode(next_point, radius / 2, parent=node)

                node.children[idx] = child_node
                child_nodes.append(child_node)

        return child_nodes


class IndexSampler(AbstractSampler):
    def __init__(
        self,
        sampler: AbstractSampler,
        nodes: List[QuadTreeNode],
        random_state: Union[RandomState, int, None] = None,
    ):
        super().__init__(random_state)
        self.sampler = sampler
        self.nodes = nodes

    def __call__(self, size: int) -> np.array:
        samples = self.sampler(size)
        idx_samples = [len(self.nodes)] * size
        for i in range(size):
            for j in range(len(self.nodes)):
                if (
                    np.linalg.norm(samples[i] - self.nodes[j].point, ord=np.inf)
                    <= self.nodes[j].radius
                ):
                    idx_samples[i] = j
        return idx_samples


def r_quad_tree(
    sampler: AbstractSampler,
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

    if not isinstance(random_state, RandomState):
        random_state = RandomState(random_state)

    t = 3  # int(np.ceil(1 / 2 * np.log(5 * Delta**2 / (eps * Gamma)) + 1))
    M = (Delta / eps) ** 2  # * 2**10
    gamma = eps / (t * k * M * Delta**2)  # / 20
    # print(t, M, gamma)

    # build quad-tree
    root = QuadTreeNode(point=np.array([0.0, 0.0]), radius=1.0, is_heavy=True)
    H = [root]
    i = 1
    while H:
        # print(i)
        if (2 ** (-i + 1) * Delta) ** 2 <= eps * Gamma / 5:
            break

        child_nodes = QuadTreeNode.make_children(H)

        if i <= skip_layers:  # skip first few layers
            heavy_hitters = range(len(child_nodes))
        else:
            idx_sampler = IndexSampler(sampler, child_nodes)
            thres = gamma * Gamma * 2 ** (2 * i)
            heavy_hitters = r_heavy_hitters(
                idx_sampler,
                thres=thres,
                eps=thres / 2,
                rho=rho / t,
                delta=delta / t,
                random_state=random_state,
            )

        H = []
        for idx in heavy_hitters:
            if idx < len(child_nodes):
                child_nodes[idx].is_heavy = True
                H.append(child_nodes[idx])

        i += 1

    return root


class QuadTreeSampler(AbstractSampler):
    def __init__(
        self,
        sampler: AbstractSampler,
        root: QuadTreeNode,
        random_state: Union[RandomState, int, None] = None,
    ):
        super().__init__(random_state)
        self.sampler = sampler
        self.root = root

    def __call__(self, size: int) -> np.array:
        samples = self.sampler(size)
        for i in range(len(samples)):
            samples[i] = self.root.quad_tree_round(samples[i])

        return samples
