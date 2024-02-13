"""from numpy to jax.numpy
"""

import jax.numpy as jnp


def calc_laplacian(node_list, edge_matrix):
    node_count = node_list.size
    # edge_count = len(edge_matrix)
    l_matrix = jnp.zeros((node_count, node_count))
    for pins_in_a_net in edge_matrix:
        # for idx, pins_in_a_net in enumerate(edge_matrix):
        # print(idx, "/", edge_count)
        bool_list = jnp.isin(node_list, pins_in_a_net, assume_unique=True)
        if bool_list.sum() <= 1:
            continue
        l_matrix += jnp.diag(bool_list) * bool_list.sum() - jnp.outer(
            bool_list, bool_list
        )
    return l_matrix
