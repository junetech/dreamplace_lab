import datetime
import logging
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

from dreamplace.LaplacianCalc import calc_laplacian
from dreamplace.PlaceDB import PlaceDB


class MyProb(cp.Problem):
    mv_n_id: np.array
    x: cp.Variable
    y: cp.Variable


def make_model(placedb: PlaceDB) -> MyProb:
    s_dt = datetime.datetime.now()
    # TODO: define these values outside the code
    large_movable_node_area_criterion = 10000
    large_fixed_node_area_criterion = 10000

    # Index definition
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
    # if no large movable node, return empty problem
    if m_node_count == 0:
        my_prob = MyProb()
        my_prob.mv_n_id = []
        return my_prob

    # selected subset of fixed nodes
    sel_fx_n_id = np.array(
        [
            n_id
            for n_id in fx_n_id
            if n_wth[n_id] * n_hgt[n_id] >= large_fixed_node_area_criterion
        ]
    )

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
        "  Among %d fixed nodes, those with area larger than %d are selected"
        % (placedb.num_terminals, large_fixed_node_area_criterion)
    )
    logging.info(
        "  Selected %d large fixed nodes have %d pins" % (sel_fx_n_id.size, f_pin_count)
    )
    logging.info(
        "  Among %d movable nodes, those with area larger than %d are selected"
        % (placedb.num_movable_nodes, large_movable_node_area_criterion)
    )
    logging.info(
        "  Selected %d large movable nodes have %d pins" % (m_node_count, m_pin_count)
    )
    logging.info(
        "  Size of Lagrangian matrix is (%d,%d)"
        % (f_pin_count + m_pin_count, f_pin_count + m_pin_count)
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
