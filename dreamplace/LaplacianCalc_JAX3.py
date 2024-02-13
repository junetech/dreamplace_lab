import jax.numpy as jnp
from jax import jit, vmap


def calc_additive(bool_list):
    """vmap trick from https://github.com/google/jax/issues/13054"""
    if bool_list.sum() <= 1:
        return jnp.zeros(bool_list.size)
    return bool_list.sum() * jnp.diag(bool_list) - vmap(
        vmap(jnp.multiply, (None, 0)), (0, None)
    )(bool_list, bool_list)


def calc_laplacian(node_list, edge_matrix):
    calc_additive_jit = jit(calc_additive)
    node_count = node_list.size
    edge_count = len(edge_matrix)
    return_matrix = jnp.zeros(shape=(node_count, node_count))
    for idx, pins_in_a_net in enumerate(edge_matrix):
        print(idx, "/", edge_count)
        return_matrix += calc_additive_jit(
            vmap(jnp.isin)(node_list, jnp.resize(pins_in_a_net, node_count))
        )
    return return_matrix
