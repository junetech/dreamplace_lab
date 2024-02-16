"""Laplacian Matrix calculation using numpy
"""

from itertools import combinations
import numpy as np


def calc_laplacian(pin_list, edge_matrix):
    pin_id2idx: dict[int, int] = {pin_id: idx for idx, pin_id in enumerate(pin_list)}
    pin_count = pin_list.size
    l_matrix = np.zeros((pin_count, pin_count), dtype=np.int16)
    for pins_in_a_net in edge_matrix:
        pins_of_interest = np.intersect1d(pins_in_a_net, pin_list)
        degree_plus_one = len(pins_of_interest)
        if degree_plus_one <= 1:
            continue
        for u, v in combinations(pins_of_interest, 2):
            l_matrix[pin_id2idx[u]][pin_id2idx[v]] -= 1
            l_matrix[pin_id2idx[v]][pin_id2idx[u]] -= 1
        for pin in pins_of_interest:
            l_matrix[pin_id2idx[pin]][pin_id2idx[pin]] += degree_plus_one - 1
    return l_matrix
