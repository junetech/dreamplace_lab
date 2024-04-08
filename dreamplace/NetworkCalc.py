"""Laplacian Matrix calculation using numpy
"""

from itertools import combinations

import networkx as nx
import numpy as np
import numpy.typing as npt


class NodePinnEt:
    dummy_pin_count: int

    def __init__(self):
        # cal{P}_m^1: set of pins in movable nodes selected
        self.mv_vp_id_set: set[int] = set()
        # cal{P}_f^1: set of pins in fixed nodes selected
        self.fx_vp_id_set: set[int] = set()
        # E^1: set of nets selected
        self.net_id_set: set[int] = set()

    def create_selected_node_set(self, vpin2node_map: npt.NDArray[np.int32]):
        # N_m^1: set of movable nodes selected
        self.mv_n_id_set: set[int] = set(vpin2node_map[list(self.mv_vp_id_set)])
        # N_f^1: set of fixed nodes selected
        self.fx_n_id_set: set[int] = set(vpin2node_map[list(self.fx_vp_id_set)])


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
    vp_count: int,
    net2pin_map: npt.NDArray[npt.NDArray[np.int32]],
    pin2vpin_map: npt.NDArray[np.int32],
) -> npt.NDArray[npt.NDArray[np.int32]]:
    nx_graph, npe = create_simple_graph(vp_count, net2pin_map, pin2vpin_map)
    return nx.laplacian_matrix(
        nx_graph, nodelist=np.arange(vp_count + npe.dummy_pin_count)
    ).toarray()
