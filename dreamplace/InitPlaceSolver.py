import datetime
import logging
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

from dreamplace.LaplacianCalc_JAX import calc_laplacian
from dreamplace.PlaceDB import PlaceDB


class MyProb(cp.Problem):
    mv_n_id: np.array
    x: cp.Variable
    y: cp.Variable


def do_initial_place(placedb: PlaceDB) -> Tuple[Dict[int, float], Dict[int, float]]:
    # TODO: define these values outside the code
    large_movable_node_area_criterion = 10000
    large_fixed_node_area_criterion = 10000

    # initialize (x,y) of large movable nodes utilizing math model
    s_dt = datetime.datetime.now()
    sel_mv_n_id = np.array(
        [
            n_id
            for n_id in np.array([n_id for n_id in range(placedb.num_movable_nodes)])
            if placedb.node_size_x[n_id] * placedb.node_size_y[n_id]
            >= large_movable_node_area_criterion
        ]
    )
    if sel_mv_n_id.size > 0:
        logging.info(
            "  Among %d movable nodes, those with area larger than %d are selected"
            % (placedb.num_movable_nodes, large_movable_node_area_criterion)
        )
        sel_fx_n_id = np.array(
            [
                n_id
                for n_id in np.array(
                    [
                        n_id
                        for n_id in range(
                            placedb.num_movable_nodes,
                            placedb.num_movable_nodes + placedb.num_terminals,
                        )
                    ]
                )
                if placedb.node_size_x[n_id] * placedb.node_size_y[n_id]
                >= large_fixed_node_area_criterion
            ]
        )
        logging.info(
            "  Among %d fixed nodes, those with area larger than %d are selected"
            % (placedb.num_terminals, large_fixed_node_area_criterion)
        )
        mdl = make_lmn_model(placedb, sel_mv_n_id, sel_fx_n_id)
        logging.info(
            f"Initial-placing large nodes: math model building takes {datetime.datetime.now() - s_dt} sec."
        )

        s_dt = datetime.datetime.now()
        x_dict, y_dict = return_sol(mdl)
        logging.info(
            f"Initial-placing large nodes: math model solving takes {datetime.datetime.now() - s_dt} sec."
        )
    else:
        logging.info(
            f"Initial-placing large nodes: no movable nodes; checking takes {datetime.datetime.now() - s_dt} sec."
        )
        x_dict, y_dict = {}, {}

    # initialize (x,y) of small movable nodes utilizing math model

    return x_dict, y_dict


def make_lmn_model(placedb: PlaceDB, sel_mv_n_id, sel_fx_n_id) -> MyProb:
    s_dt = datetime.datetime.now()

    # Index definition

    # selected subset of movable nodes
    m_node_count = sel_mv_n_id.size
    # node width & height
    n_wth, n_hgt = placedb.node_size_x, placedb.node_size_y

    # selected subset of movable and fixed pins
    sel_mv_p_id = np.concatenate(
        [placedb.node2pin_map[n_id] for n_id in sel_mv_n_id], axis=None
    )
    m_pin_count = sel_mv_p_id.size
    sel_fx_p_id = np.concatenate(
        [placedb.node2pin_map[n_id] for n_id in sel_fx_n_id], axis=None
    )
    f_pin_count = sel_fx_p_id.size
    logging.info(f"Index definition takes {datetime.datetime.now()-s_dt}"[:-3])
    logging.info(
        "  Selected %d large fixed nodes have %d pins" % (sel_fx_n_id.size, f_pin_count)
    )
    logging.info(
        "  Selected %d large movable nodes have %d pins" % (m_node_count, m_pin_count)
    )
    logging.info(
        "  Total %d pins in selected large nodes" % (f_pin_count + m_pin_count)
    )
    s_dt = datetime.datetime.now()

    # Parameter definition
    # list of pin offset
    xpo, ypo = placedb.pin_offset_x, placedb.pin_offset_y
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
    # block area
    block_wth, block_hgt = placedb.xh - placedb.xl, placedb.yh - placedb.yl
    logging.info(
        f"Parameter definition before Laplacian takes {datetime.datetime.now()-s_dt}"[
            :-3
        ]
    )
    s_dt = datetime.datetime.now()

    # Create Laplacian matrix with pins
    L_matrix = calc_laplacian(p_prime_set, placedb.net2pin_map)
    L_mm = L_matrix[0:m_pin_count, 0:m_pin_count]
    L_fm = L_matrix[m_pin_count:, 0:m_pin_count]
    L_ff = L_matrix[m_pin_count:, m_pin_count:]

    logging.info(
        f"Graph Laplacian calculation takes {datetime.datetime.now()-s_dt}"[:-3]
    )
    logging.info(f"  Size of Lagrangian matrix is {L_matrix.shape}")
    s_dt = datetime.datetime.now()
    # Parameters end

    # Variables
    xn_m = cp.Variable(m_node_count)
    xp_m = cp.Variable(m_pin_count)
    yn_m = cp.Variable(m_node_count)
    yp_m = cp.Variable(m_pin_count)
    logging.info(f"Variable definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

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
    logging.info(f"Constraint definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

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
    logging.info(f"Objective definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

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
    # for n_id in x_dict:
    #     print(f"Node {n_id} x={x_dict[n_id]:.3f} y={y_dict[n_id]:.3f}")
    return x_dict, y_dict
