"""Laplacian Matrix calculation using numpy
"""

import numpy as np


def calc_laplacian(node_list, edge_matrix):
    node_count = node_list.size
    # edge_count = len(edge_matrix)
    l_matrix = np.zeros((node_count, node_count))
    # for idx, pins_in_a_net in enumerate(edge_matrix):
    for pins_in_a_net in edge_matrix:
        # print(idx, "/", edge_count)
        bool_list = np.isin(node_list, pins_in_a_net, assume_unique=True)
        if bool_list.sum() <= 1:
            continue
        l_matrix += np.diag(bool_list) * bool_list.sum() - np.outer(
            bool_list, bool_list
        )
    return l_matrix
