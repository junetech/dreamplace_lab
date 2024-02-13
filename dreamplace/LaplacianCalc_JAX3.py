"""vmapping jit-compiled additive calculation
does not work
"""

import jax.numpy as jnp
from jax import jit, vmap


@jit
def calc_additive(bool_list):
    """vmap trick from https://github.com/google/jax/issues/13054"""
    if bool_list.sum() <= 1:  # this line is unacceptable
        return jnp.zeros(bool_list.size**2)
    return bool_list.sum() * jnp.diag(bool_list) - vmap(
        vmap(jnp.multiply, (None, 0)), (0, None)
    )(bool_list, bool_list).reshape(bool_list.size**2)


def calc_laplacian(node_list, edge_matrix):
    node_count = node_list.size
    edge_count = len(edge_matrix)
    return_matrix = jnp.zeros(shape=node_count**2)
    for idx, pins_in_a_net in enumerate(edge_matrix):
        print(idx, "/", edge_count)
        return_matrix += calc_additive(
            vmap(jnp.isin)(node_list, jnp.resize(pins_in_a_net, node_count))
        )
    return return_matrix.reshape((node_count, node_count))
