"""from numpy to jax.numpy
"""

from itertools import combinations
import jax.numpy as jnp


def calc_laplacian(pin_list, edge_matrix):
    # pin_id2idx: dict[int, int] = {pin_id: idx for idx, pin_id in enumerate(pin_list)}
    pin_count = pin_list.size
    l_matrix = jnp.zeros((pin_count, pin_count), dtype=jnp.int16)
    for pins_in_a_net in edge_matrix:
        pins_of_interest = jnp.intersect1d(pins_in_a_net, pin_list)
        degree_plus_one = len(pins_of_interest)
        if degree_plus_one <= 1:
            continue
        pin_indices = [jnp.where(pin_list == pin) for pin in pins_of_interest]
        for u, v in combinations(pin_indices, 2):
            l_matrix.at[u, v].set(l_matrix.at[u, v] - 1)
            l_matrix.at[v, u].set(l_matrix.at[v, u] - 1)
        for pin in pin_indices:
            l_matrix.at[pin, pin].set(l_matrix.at[pin, pin] - 1 + degree_plus_one)
    return l_matrix
