"""Laplacian Matrix calculation using numpy
"""

from itertools import combinations
import numpy as np
import jax.numpy as jnp


def calc_laplacian(pin_list, edge_matrix):
    pin_id2idx: dict[int, int] = {pin_id: idx for idx, pin_id in enumerate(pin_list)}
    pin_count = pin_list.size
    _pin_list = jnp.array(pin_list)
    l_matrix = np.zeros((pin_count, pin_count), dtype=np.int16)
    for pins_in_a_net in edge_matrix:
        pins_of_interest = jnp.intersect1d(pins_in_a_net, _pin_list)
        degree_plus_one = len(pins_of_interest)
        if degree_plus_one <= 1:
            continue
        pin_indices = [pin_id2idx[pin] for pin in pins_of_interest.astype(np.int16)]
        for p, q in combinations(pin_indices, 2):
            l_matrix[p][q] -= 1
            l_matrix[q][p] -= 1
        for p in pin_indices:
            l_matrix[p][p] += degree_plus_one - 1
    return l_matrix
