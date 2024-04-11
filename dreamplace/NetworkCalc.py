"""Laplacian Matrix calculation using numpy
"""

import logging
from itertools import combinations

import networkx as nx
import numpy as np
import numpy.typing as npt
from dreamplace.VirtualElem import VPinDB, StarVDB


def create_simple_graph(
    vpin_db: VPinDB,
    net2pin_map: npt.NDArray[npt.NDArray[np.int32]],
    mv_node_count: int,
    all_node_count: int,
) -> tuple[nx.Graph, StarVDB]:
    simple_g = nx.Graph()

    _fx_star_v_id_list: list[int] = []
    _mv_star_v_id_list: list[int] = []
    _is_selected = [False] * all_node_count
    star_v_db = StarVDB()

    net_count_1, net_count_2, net_count_3 = 0, 0, 0

    for net, pins_in_a_net in enumerate(net2pin_map):
        vpin_set = set(vpin_db.pin2vpin_map[pins_in_a_net])
        # if less than 2 vpins, skip
        if len(vpin_set) < 2:
            net_count_1 += 1
            continue

        # make a star graph
        if len(vpin_set) > 4:
            net_count_2 += 1
            star_v = vpin_db.vp_count + star_v_db.star_v_count
            star_v_db.star_v_count += 1
            simple_g.add_edges_from([(u, star_v) for u in vpin_set], label=net)

            is_net_with_fixed_only = True
            for u in vpin_set:
                node = vpin_db.vpin2node_map[u]
                if node < mv_node_count:
                    # classify star vertices for fixed pins only
                    is_net_with_fixed_only = False
                    break
            if is_net_with_fixed_only:
                _fx_star_v_id_list.append(star_v)
            else:
                _mv_star_v_id_list.append(star_v)

        # make a clique
        else:  # 2 <= len(vpin_set) <=6 4
            net_count_3 += 1
            simple_g.add_edges_from(combinations(vpin_set, 2), label=net)
        for u in vpin_set:
            # Node reduction
            _is_selected[vpin_db.vpin2node_map[u]] = True

    logging.info(f"  Total {len(net2pin_map)} nets")
    logging.info(f"  {net_count_1} nets have less than 2 pins")
    logging.info(f"  {net_count_2} nets have more than 4 pins")
    logging.info(f"  {net_count_3} nets have 2~4 pins")

    star_v_db.fx_star_v_id_list = np.array(_fx_star_v_id_list, dtype=bool)
    star_v_db.mv_star_v_id_list = np.array(_mv_star_v_id_list, dtype=bool)
    star_v_db.is_selected = np.array(_is_selected, dtype=bool)

    return simple_g, star_v_db


def create_virtual_graph(
    simple_graph: nx.Graph, ms_mt_partition: dict[int, int]
) -> tuple[nx.Graph, dict[int, int]]:
    virtual_graph = nx.Graph()

    max_ms_mt_partition = max(ms_mt_partition.values())
    additional_partition_id = max_ms_mt_partition
    # remaining elements -> additional partition ID
    other_partition_id_dict: dict[int, int] = {}

    for v1, v2 in simple_graph.edges():
        vnode1: int
        vnode2: int
        if v1 in ms_mt_partition:
            vnode1 = ms_mt_partition[v1]
        # is other type of element
        elif v1 in other_partition_id_dict:
            vnode1 = other_partition_id_dict[v1]
        else:
            additional_partition_id += 1
            vnode1 = additional_partition_id
            other_partition_id_dict[v1] = vnode1
        if v2 in ms_mt_partition:
            vnode2 = ms_mt_partition[v2]
        # is other type of element
        elif v2 in other_partition_id_dict:
            vnode1 = other_partition_id_dict[v2]
        else:
            additional_partition_id += 1
            vnode2 = additional_partition_id
            other_partition_id_dict[v2] = vnode2
        if vnode1 != vnode2:
            if virtual_graph.has_edge(vnode1, vnode2):
                virtual_graph[vnode1][vnode2]["weight"] += 1
            else:
                virtual_graph.add_edge(vnode1, vnode2, weight=1)

    return virtual_graph, other_partition_id_dict


def calc_laplacian(
    simple_graph: nx.Graph, node_id_array: npt.NDArray[np.int32]
) -> npt.NDArray[npt.NDArray[np.int32]]:
    return nx.laplacian_matrix(simple_graph, nodelist=node_id_array)
