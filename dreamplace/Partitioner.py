import logging

import networkx as nx
import numpy as np
from dreamplace.VirtualElem import StarVDB, VPinDB


def partition(
    simple_graph: nx.Graph, star_v_db: StarVDB, vpin_db: VPinDB
) -> dict[int, int]:
    subgraph_nodes = np.concatenate(
        (vpin_db.small_movable_vp_id_list, star_v_db.mv_star_v_id_list)
    )
    num_subgraph_nodes = len(subgraph_nodes)
    logging.info(f"Subgraph with {num_subgraph_nodes} nodes are partitioned")
    logging.info(
        "  %d movable small nodes + %d star vertices"
        % (len(vpin_db.small_movable_vp_id_list), len(star_v_db.mv_star_v_id_list))
    )
    movable_only_g: nx.Graph = simple_graph.subgraph(subgraph_nodes).copy()

    v2part_dict: dict[int, int] = {}
    part_count = 0

    # for component in [
    #     movable_only_g.subgraph(c).copy()
    #     for c in nx.connected_components(movable_only_g)
    # ]:
    #     if len(component.nodes) == 0:
    #         continue
    #     parts, _ = cugraph.louvain(component)
    #     this_part_count = len(set(parts.values()))
    #     logging.info(f"  {len(parts)} nodes -> {this_part_count} partitions")
    #     v2part_dict.update({key: value + part_count for key, value in parts.items()})
    #     part_count += this_part_count
    num_connected_comp = nx.number_connected_components(movable_only_g)
    logging.info(f"{num_connected_comp} connected components in the subgraph")
    raise UserWarning
    if num_connected_comp > num_subgraph_nodes / 100:
        # 아무리 쪼개도 node 갯수의 1/100 이하로 쪼개기 불가.
        # 각 connected component를 개별 partition으로 사용
        pass
    else:
        # node 갯수의 1/100 이하 갯수의 partition이 나올때까지 쪼개기
        part_gen = nx.community.louvain_partitions(movable_only_g)
        parts: list[list[int]]
        for _parts in part_gen:
            print(type(_parts), len(_parts))
            if len(_parts) < 10000:
                parts = _parts
                break
        logging.info(f"{len(v2part_dict)} nodes -> {part_count} partitions")

    # movable star vertices \union movable virtual pins -> partition
    return v2part_dict
