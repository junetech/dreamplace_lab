import cugraph
import networkx as nx
from dreamplace.VirtualElem import StarVDB


def partition(simple_graph: nx.Graph, star_v_db: StarVDB) -> dict[int, int]:
    subgraph_nodes = sorted(star_v_db.mv_vp_id_set.union(star_v_db.mv_star_v_id_list))
    movable_only_g = simple_graph.subgraph(subgraph_nodes)
    parts, _ = cugraph.louvain(movable_only_g)

    return parts
