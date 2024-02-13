import jax.numpy as jnp
from jax import jit, vmap


@jit
def calc_additives(bool_list):
    """vmap trick from https://github.com/google/jax/issues/13054"""
    return bool_list.sum() * jnp.diag(bool_list) - vmap(
        vmap(jnp.multiply, (None, 0)), (0, None)
    )(bool_list, bool_list)


def calc_laplacian(node_list, edge_matrix):
    node_count = node_list.size
    edge_count = len(edge_matrix)
    l_matrix = jnp.zeros((node_count, node_count))
    for idx, pins_in_a_net in enumerate(edge_matrix):
        print(idx, "/", edge_count)
        bool_list = jnp.isin(node_list, pins_in_a_net, assume_unique=True)
        if bool_list.sum() <= 1:
            continue
        l_matrix += calc_additives(bool_list)
    return l_matrix
