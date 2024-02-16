"""Laplacian Matrix calculation using numpy
"""

from itertools import combinations
import numpy as np


def calc_laplacian(node_list, edge_matrix):
    node_count = node_list.size
    # edge_count = len(edge_matrix)
    l_matrix = np.zeros((node_count, node_count))
    # for idx, pins_in_a_net in enumerate(edge_matrix):
    for pins_in_a_net in edge_matrix:
        # print(idx, "/", edge_count)
        # bool_list = np.isin(node_list, pins_in_a_net, assume_unique=True)
        # if bool_list.sum() <= 1:
        #     continue
        # l_matrix += np.diag(bool_list) * bool_list.sum() - np.outer(
        #     bool_list, bool_list
        # )
        pins_of_interest = np.intersect1d(pins_in_a_net, node_list)
        if len(pins_of_interest) >= 2:
            for u, v in combinations(pins_of_interest, 2):
                l_matrix[u][v] -= 1
                l_matrix[v][u] -= 1
            for pin in pins_of_interest:
                l_matrix[pin][pin] += 1
    return l_matrix
