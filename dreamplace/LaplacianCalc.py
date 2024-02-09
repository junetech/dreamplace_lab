import numpy as np


def calc_laplacian(node_list, edge_matrix):
    l_matrix = np.zeros((len(node_list), len(node_list)))
    for pins_in_a_net in edge_matrix:
        bool_list = np.isin(node_list, pins_in_a_net, assume_unique=True)
        l_matrix += np.diag(bool_list) * bool_list.sum() - np.outer(
            bool_list, bool_list
        )
    return l_matrix
