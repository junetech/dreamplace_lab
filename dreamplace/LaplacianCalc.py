import numpy as onp
import jax.numpy as jnp


def calc_laplacian(node_list, edge_matrix):
    jax_nodes = jnp.array(node_list)
    jax_edges = jnp.array(edge_matrix)
    l_matrix = jnp.zeros((len(jax_nodes), len(jax_nodes)))
    for nodes_in_an_edge in jax_edges:
        bool_list = jnp.isin(jax_nodes, nodes_in_an_edge, assume_unique=True)
        l_matrix += jnp.diag(bool_list) * bool_list.sum() - jnp.outer(
            bool_list, bool_list
        )
    return onp.array(l_matrix)
