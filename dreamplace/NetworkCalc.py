"""Laplacian Matrix calculation using numpy
"""

from itertools import combinations

import networkx as nx
import numpy as np
import numpy.typing as npt


class NodePinnEt:
    dummy_pin_count: int

    def __init__(self):
        # cal{P}^m: set of pins in movable nodes selected
        self.mv_vp_id_set: set[int] = set()
        # cal{P}^f: set of pins in fixed nodes selected
        self.fx_vp_id_set: set[int] = set()
        # E^1: set of nets selected
        self.net_id_set: set[int] = set()

    def select_nodes_with_vpins(
        self, vp_id_dict: dict[int, list[int]], vpin2node_map: npt.NDArray[np.int32]
    ):
        # \cal{N}^m: set of movable nodes selected
        self.mv_n_id_set: set[int] = set(vpin2node_map[list(self.mv_vp_id_set)])

        # \cal{N}^f: set of fixed nodes selected
        self.fx_n_id_set: set[int] = set(vpin2node_map[list(self.fx_vp_id_set)])

        # \cal{P}(n): node -> set of vpins
        self.vp_id_dict: dict[int, set[int]] = {}
        n_id_set = self.mv_n_id_set.union(self.fx_n_id_set)
        vp_id_set = self.mv_vp_id_set.union(self.fx_vp_id_set)
        for n_id in n_id_set:
            self.vp_id_dict[n_id] = vp_id_set.intersection(vp_id_dict[n_id])


def create_simple_graph(
    mv_vp_count: int,
    fx_vp_count: int,
    net2pin_map: npt.NDArray[npt.NDArray[np.int32]],
    pin2vpin_map: npt.NDArray[np.int32],
) -> tuple[nx.Graph, NodePinnEt]:
    simple_g = nx.Graph()
    npe = NodePinnEt()

    dummy_pin_count = 0
    for net, pins_in_a_net in enumerate(net2pin_map):
        vpin_set = set(pin2vpin_map[pins_in_a_net])
        # if less than 2 vpins, skip
        if len(vpin_set) < 2:
            continue
        # print(pins_in_a_net)
        # print(pin2vpin_map)
        # print(vpin_set)
        # raise UserWarning
        npe.net_id_set.add(net)
        # make a star graph
        if False:  # len(vpin_set) >= 4:
            dummy_vpin = dummy_pin_count + mv_vp_count + fx_vp_count
            dummy_pin_count += 1
            simple_g.add_edges_from([(u, dummy_vpin) for u in vpin_set], label=net)
            for u in vpin_set:
                if u < mv_vp_count:
                    npe.mv_vp_id_set.add(u)
                else:
                    npe.fx_vp_id_set.add(u)
        # make a clique
        else:  # 2 <= len(vpin_set) < 4
            simple_g.add_edges_from(combinations(vpin_set, 2), label=net)
            for u in vpin_set:
                if u < mv_vp_count:
                    npe.mv_vp_id_set.add(u)
                else:
                    npe.fx_vp_id_set.add(u)

    npe.dummy_pin_count = dummy_pin_count
    return simple_g, npe


def calc_vpin_laplacian(
    simple_graph: nx.Graph, node_id_array: npt.NDArray[np.int32]
) -> npt.NDArray[npt.NDArray[np.int32]]:
    return nx.laplacian_matrix(simple_graph, nodelist=node_id_array).toarray()
