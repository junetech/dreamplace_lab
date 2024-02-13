# import numpy as np
import jax.numpy as jnp
from jax import jit, vmap


def calc_additives(bool_list):
    """vmap trick from https://github.com/google/jax/issues/13054"""
    return (
        bool_list.sum() * jnp.diag(bool_list)
        - vmap(vmap(jnp.multiply, (None, 0)), (0, None))(bool_list, bool_list)
    ).reshape(len(bool_list) ** 2)


# vv = lambda a, b: jnp.isin(b, a)
def vv(a, b):
    return jnp.isin(b, a)


def calc_laplacian_vmap(node_list, edge_matrix):
    jit_additive = jit(calc_additives, backend="cpu")
    # for pins_in_a_net in edge_matrix:
    #     print(jnp.isin(node_list, pins_in_a_net))
    vm = vmap(vv, (0, None), 0)
    # for result in vm(edge_matrix, node_list):
    #     print(result)
    # for result2 in vmap(calc_additives)(vm(edge_matrix, node_list)):
    #     print(result2)
    return vmap(jnp.sum, in_axes=1)(
        vmap(jit_additive)(vm(edge_matrix, node_list))
    ).reshape(len(node_list), len(node_list))


def calc_laplacian(node_list, edge_matrix):
    node_count = node_list.size
    return calc_laplacian_vmap(
        node_list, jnp.array([jnp.resize(edge, node_count) for edge in edge_matrix])
    )
