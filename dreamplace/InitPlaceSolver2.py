import datetime
import logging
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

from dreamplace.PlaceDB import PlaceDB


class MyProb(cp.Problem):
    mv_n_id: np.array
    x: cp.Variable
    y: cp.Variable


def make_model(placedb: PlaceDB) -> MyProb:
    large_movable_node_area_criterion = 10000
    large_fixed_node_area_criterion = 10000

    # Parameters start
    # movable node id list
    mv_n_id = np.array([n_id for n_id in range(placedb.num_movable_nodes)])
    # fixed node id list
    fx_n_id = np.array(
        [
            n_id
            for n_id in range(
                placedb.num_movable_nodes,
                placedb.num_movable_nodes + placedb.num_terminals,
            )
        ]
    )

    # list of pin offset
    xpo, ypo = placedb.pin_offset_x, placedb.pin_offset_y

    # node width & height
    n_wth, n_hgt = placedb.node_size_x, placedb.node_size_y
    # selected subset of movable nodes
    sel_mv_n_id = np.array(
        [
            n_id
            for n_id in mv_n_id
            if n_wth[n_id] * n_hgt[n_id] >= large_movable_node_area_criterion
        ]
    )
    m_node_count = sel_mv_n_id.size
    print(m_node_count, "large movable nodes selected")
    # selected subset of fixed nodes
    sel_fx_n_id = np.array(
        [
            n_id
            for n_id in fx_n_id
            if n_wth[n_id] * n_hgt[n_id] >= large_fixed_node_area_criterion
        ]
    )
    print(len(sel_fx_n_id), "large fixed nodes selected")

    # selected subset of movable and fixed pins
    sel_mv_p_id = np.concatenate(
        [placedb.node2pin_map[n_id] for n_id in sel_mv_n_id], axis=None
    )
    m_pin_count = sel_mv_p_id.size
    sel_fx_p_id = np.concatenate(
        [placedb.node2pin_map[n_id] for n_id in sel_fx_n_id], axis=None
    )

    # Position of fixed pins
    _xp_f, _yp_f = [], []
    for n_id in sel_fx_n_id:
        _xp_f.extend(
            [placedb.node_x[n_id] + xpo[p_id] for p_id in placedb.node2pin_map[n_id]]
        )
        _yp_f.extend(
            [placedb.node_y[n_id] + ypo[p_id] for p_id in placedb.node2pin_map[n_id]]
        )
    xp_f = np.array(_xp_f)
    yp_f = np.array(_yp_f)

    p_prime_set = np.concatenate((sel_mv_p_id, sel_fx_p_id), axis=None)
    pin_id2idx_map = {p_id: idx for idx, p_id in enumerate(p_prime_set)}

    # Create Laplacian matrix with pins
    pin_count = p_prime_set.size
    L_matrix = np.zeros((pin_count, pin_count))
    for pins_in_a_net in placedb.net2pin_map:
        bool_list = np.isin(p_prime_set, pins_in_a_net, assume_unique=True)
        if bool_list.sum() <= 1:
            continue
        L_matrix += np.diag(bool_list) * bool_list.sum() - np.outer(
            bool_list, bool_list
        )
    L_mm = L_matrix[0:m_pin_count, 0:m_pin_count]
    L_fm = L_matrix[m_pin_count:, 0:m_pin_count]
    L_ff = L_matrix[m_pin_count:, m_pin_count:]

    # block area
    block_wth, block_hgt = placedb.xh - placedb.xl, placedb.yh - placedb.yl

    print("Parameter definition end")
    # Parameters end

    # Variables
    xn_m = cp.Variable(m_node_count)
    xp_m = cp.Variable(m_pin_count)
    yn_m = cp.Variable(m_node_count)
    yp_m = cp.Variable(m_pin_count)
    print("Variables definition end")

    # Constraints
    constrs: List[cp.Constraint] = []
    # for n_idx, n_id in enumerate(sel_mv_n_id):
    #     for p_id in placedb.node2pin_map[n_id]:
    #         p_idx = pin_id2idx_map[p_id]
    #         constrs.append(xn_m[n_idx] + xpo[p_id] == xp_m[p_idx])
    #         constrs.append(yn_m[n_idx] + ypo[p_id] == yp_m[p_idx])
    for n_idx, n_id in enumerate(sel_mv_n_id):
        constrs.extend(
            [
                xn_m[n_idx] + xpo[p_id] == xp_m[pin_id2idx_map[p_id]]
                for p_id in placedb.node2pin_map[n_id]
            ]
            + [
                yn_m[n_idx] + ypo[p_id] == yp_m[pin_id2idx_map[p_id]]
                for p_id in placedb.node2pin_map[n_id]
            ]
        )
    for n_idx, n_id in enumerate(sel_mv_n_id):
        constrs.extend(
            [
                xn_m[n_idx] >= 0,
                xn_m[n_idx] + n_wth[n_id] <= block_wth,
                yn_m[n_idx] >= 0,
                yn_m[n_idx] + n_hgt[n_id] <= block_hgt,
            ]
        )

    # Objective
    obj_constant = xp_f.T @ L_ff @ xp_f + yp_f.T @ L_ff @ yp_f
    prob = MyProb(
        objective=cp.Minimize(
            cp.quad_form(xp_m, L_mm, assume_PSD=True)
            + 2 * xp_f @ L_fm @ xp_m
            + cp.quad_form(yp_m, L_mm, assume_PSD=True)
            + 2 * yp_f @ L_fm @ yp_m
            + obj_constant
        ),
        constraints=constrs,
    )
    print("Objective definition end")

    prob.mv_n_id = sel_mv_n_id
    prob.x, prob.y = xn_m, yn_m
    return prob


def return_sol(prob: MyProb) -> Tuple[Dict[int, float], Dict[int, float]]:
    prob.solve(verbose=True)
    if prob.status == "infeasible":
        return {}, {}
    # x_dict = {}
    # for idx, n_id in enumerate(prob.mv_n_id):
    #     x_dict[n_id] = x.value[idx]
    x_dict = {n_id: prob.x.value[idx] for idx, n_id in enumerate(prob.mv_n_id)}
    y_dict = {n_id: prob.y.value[idx] for idx, n_id in enumerate(prob.mv_n_id)}
    for n_id in x_dict:
        print(f"Node {n_id} x={x_dict[n_id]:.3f} y={y_dict[n_id]:.3f}")
    raise UserWarning
    return x_dict, y_dict


def main():
    s_dt = datetime.datetime.now()
    mdl = make_model()
    elapsed_d = datetime.datetime.now() - s_dt
    logging.info(f"Math model building took {elapsed_d}"[:-3])
    s_dt = datetime.datetime.now()
    mdl.hideOutput()  # silent mode
    mdl.optimize()
    elapsed_d = datetime.datetime.now() - s_dt
    logging.info(f"Math model solving took {elapsed_d}"[:-3])


if __name__ == "__main__":
    START_DT = datetime.datetime.now()
    main()
    end_dt = datetime.datetime.now()
    elapsed_d = end_dt - START_DT
    logging.info(
        f"{__name__} program end @ {end_dt}"[:-3] + f"; took total {elapsed_d}"[:-3]
    )
