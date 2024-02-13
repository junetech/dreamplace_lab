import jax.numpy as jnp


def calc_laplacian(node_list, edge_matrix):
    l_matrix = jnp.zeros((len(node_list), len(node_list)))
    for pins_in_a_net in edge_matrix:
        bool_list = jnp.isin(node_list, pins_in_a_net, assume_unique=True)
        l_matrix += jnp.diag(bool_list) * bool_list.sum() - jnp.outer(
            bool_list, bool_list
        )
    return l_matrix
