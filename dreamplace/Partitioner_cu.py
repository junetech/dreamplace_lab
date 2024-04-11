import logging

import cugraph
import networkx as nx
import numpy as np
from dreamplace.VirtualElem import StarVDB, VPinDB


def partition(
    simple_graph: nx.Graph, star_v_db: StarVDB, vpin_db: VPinDB
) -> dict[int, int]:
    subgraph_nodes = np.concatenate(
        (vpin_db.small_movable_vp_id_list, star_v_db.mv_star_v_id_list)
    )
    logging.info(f"Subgraph with {len(subgraph_nodes)} nodes are partitioned")
    movable_only_g: nx.Graph = simple_graph.subgraph(subgraph_nodes).copy()

    v2part_dict: dict[int, int] = {}
    part_count = 0

    for component in [
        movable_only_g.subgraph(c).copy()
        for c in nx.connected_components(movable_only_g)
    ]:
        if len(component.nodes) == 0:
            continue
        parts, _ = cugraph.louvain(component)
        this_part_count = len(set(parts.values()))
        logging.info(f"  {len(parts)} nodes -> {this_part_count} partitions")
        v2part_dict.update({key: value + part_count for key, value in parts.items()})
        part_count += this_part_count
    logging.info(f"{len(v2part_dict)} nodes -> {part_count} partitions")

    # movable star vertices \union movable virtual pins -> partition
    return v2part_dict
