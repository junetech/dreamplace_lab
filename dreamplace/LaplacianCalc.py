import numba as nb
import numpy as np


@nb.jit(parallel=True)
def isin(a, b):
    """
    https://stackoverflow.com/questions/70865732/faster-numpy-isin-alternative-for-strings-using-numba
    """
    out = np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i] = True
        else:
            out[i] = False
    return out


@nb.jit(parallel=True)
def calc_laplacian(node_list, edge_matrix):
    l_matrix = np.zeros((len(node_list), len(node_list)))
    for pins_in_a_net in edge_matrix:
        bool_list = isin(node_list, pins_in_a_net)
        l_matrix += np.diag(bool_list) * bool_list.sum() - np.outer(
            bool_list, bool_list
        )
    return l_matrix
