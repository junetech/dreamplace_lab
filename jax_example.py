import time

import jax.numpy as jnp
import jax.random as jrnd


def calc_laplacian(node_list, edge_matrix):
    l_matrix = jnp.zeros((len(node_list), len(node_list)))
    for pins_in_a_net in edge_matrix:
        bool_list = jnp.isin(node_list, pins_in_a_net, assume_unique=True)
        l_matrix += jnp.diag(bool_list) * bool_list.sum() - jnp.outer(
            bool_list, bool_list
        )
    return l_matrix


key = jrnd.PRNGKey(1000)

node_count = 100
b = jnp.arange(1, 1 + node_count, dtype=jnp.int32)
# A = jnp.array(
#     [
#         jrnd.choice(
#             key,
#             b,
#             shape=(1, jrnd.randint(key, shape=[1], minval=2, maxval=node_count)[0]),
#         )
#     ]
#     * node_count
# )
A = jnp.array([jrnd.choice(key, b, shape=(1, jnp.int32(node_count / 5)))] * node_count)
print(b.shape, A.shape)
tt = time.time()
l_matrix = calc_laplacian(b, A)
print(l_matrix.shape)
print(f"Laplacian calc takes {time.time()-tt} seconds")
