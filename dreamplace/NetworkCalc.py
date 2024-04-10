"""Laplacian Matrix calculation using numpy
"""

from itertools import combinations

import networkx as nx
import numpy as np
import numpy.typing as npt
from dreamplace.VirtualElem import VPinDB, StarVDB


def create_simple_graph(
    vpin_db: VPinDB, net2pin_map: npt.NDArray[npt.NDArray[np.int32]], mv_node_count: int
) -> tuple[nx.Graph, StarVDB]:
    simple_g = nx.Graph()
    star_v_db = StarVDB()

    star_v_db.star_v_count = 0
    for net, pins_in_a_net in enumerate(net2pin_map):
        vpin_set = set(vpin_db.pin2vpin_map[pins_in_a_net])
        # if less than 2 vpins, skip
        if len(vpin_set) < 2:
            continue

        # make a star graph
        if len(vpin_set) >= 4:
            star_v = vpin_db.vp_count + star_v_db.star_v_count
            star_v_db.star_v_count += 1
            simple_g.add_edges_from([(u, star_v) for u in vpin_set], label=net)
            # classify selected virtual pins into movable & fixed
            for u in vpin_set:
                if u < vpin_db.mv_vp_count:
                    star_v_db.mv_vp_id_set.add(u)
                else:
                    star_v_db.fx_vp_id_set.add(u)
            # classify star vertices for fixed pins only
            is_net_with_fixed_only = True
            for u in vpin_set:
                node = vpin_db.vpin2node_map[u]
                if node > mv_node_count:
                    is_net_with_fixed_only = False
                    break
            if is_net_with_fixed_only:
                star_v_db.fx_star_v_id_list.extend(sorted(vpin_set))
            else:
                star_v_db.mv_star_v_id_list.extend(sorted(vpin_set))

        # make a clique
        else:  # 2 <= len(vpin_set) < 4
            simple_g.add_edges_from(combinations(vpin_set, 2), label=net)
            for u in vpin_set:
                if u < vpin_db.mv_vp_count:
                    star_v_db.mv_vp_id_set.add(u)
                else:
                    star_v_db.fx_vp_id_set.add(u)

    return simple_g, star_v_db


def create_virtual_graph(
    simple_graph: nx.Graph, partial_partition: dict[int, int]
) -> tuple[nx.Graph, list[int]]:
    virtual_graph = nx.Graph()

    max_partial_partition = max(partial_partition.values())
    additional_partition_id = max_partial_partition + 1
    additional_partition_id_list: list[int] = []

    for v1, v2 in simple_graph.edges():
        vnode1: int
        vnode2: int
        if v1 in partial_partition:
            vnode1 = partial_partition[v1]
        else:
            vnode1 = additional_partition_id
            additional_partition_id_list.append(vnode1)
            additional_partition_id += 1
        if v2 in partial_partition:
            vnode2 = partial_partition[v2]
        else:
            vnode2 = additional_partition_id
            additional_partition_id_list.append(vnode2)
            additional_partition_id += 1
        if vnode1 != vnode2:
            if virtual_graph.has_edge(vnode1, vnode2):
                virtual_graph[vnode1][vnode2]["weight"] += 1
            else:
                virtual_graph.add_edge(vnode1, vnode2, weight=1)
    return virtual_graph, additional_partition_id_list


def calc_vpin_laplacian(
    simple_graph: nx.Graph, node_id_array: npt.NDArray[np.int32]
) -> npt.NDArray[npt.NDArray[np.int32]]:
    return nx.laplacian_matrix(simple_graph, nodelist=node_id_array).toarray()
