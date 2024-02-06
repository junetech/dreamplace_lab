import datetime
import logging
from itertools import combinations
from typing import Dict, Tuple, Union

import numpy as np
from pyscipopt import Model, quicksum
from pyscipopt.scip import Variable

from dreamplace.PlaceDB import PlaceDB


def make_model(placedb: PlaceDB) -> Model:
    large_movable_node_area_criterion = 10000
    large_fixed_node_area_criterion = 10000
    mdl = Model("Diet")

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
    # node width & height
    n_wth, n_hgt = placedb.node_size_x, placedb.node_size_y
    # Position of fixed cells
    x: Dict[int, Union[Variable, int]] = {
        v: placedb.node_x[v]
        for v in range(
            placedb.num_movable_nodes, placedb.num_movable_nodes + placedb.num_terminals
        )
    }
    y: Dict[int, Union[Variable, int]] = {
        v: placedb.node_y[v]
        for v in range(
            placedb.num_movable_nodes, placedb.num_movable_nodes + placedb.num_terminals
        )
    }

    # pin id -> node id
    pin_to_node_map = placedb.pin2node_map
    # list of pin offset
    xpo, ypo = placedb.pin_offset_x, placedb.pin_offset_y

    # block area
    block_wth, block_hgt = placedb.xh - placedb.xl, placedb.yh - placedb.yl

    # selected subset of movable nodes
    sel_mv_n_id = np.array(
        [
            n_id
            for n_id in mv_n_id
            if n_wth[n_id] * n_hgt[n_id] >= large_movable_node_area_criterion
        ]
    )
    print(len(sel_mv_n_id), "large movable nodes selected")
    # selected subset of fixed nodes
    sel_fx_n_id = np.array(
        [
            n_id
            for n_id in fx_n_id
            if n_wth[n_id] * n_hgt[n_id] >= large_fixed_node_area_criterion
        ]
    )
    print(len(sel_fx_n_id), "large fixed nodes selected")
    v_prime_set = np.concatenate((sel_mv_n_id, sel_fx_n_id), axis=None)
    p_prime_set = np.concatenate(
        [placedb.node2pin_map[n_id] for n_id in v_prime_set], axis=None
    )
    e_prime_set = {}
    for net_id, pin_in_a_net in enumerate(placedb.net2pin_map):
        bool_list = np.isin(p_prime_set, pin_in_a_net, assume_unique=True)
        if bool_list.sum() <= 1:
            continue
        e_prime_set[net_id] = np.intersect1d(placedb.net2pin_map[net_id], p_prime_set)

    print("Parameter definition end")
    # Parameters end

    # Variables
    x_var: Dict[int, Variable] = {}
    y_var: Dict[int, Variable] = {}
    for n_id in sel_mv_n_id:
        x_var[n_id] = mdl.addVar(name="x(%s)" % n_id)
        x[n_id] = x_var[n_id]
        y_var[n_id] = mdl.addVar(name="y(%s)" % n_id)
        y[n_id] = y_var[n_id]
    print(f"{len(x_var)} x variables, {len(y_var)} y variables declared")
    print("Variables definition end")

    # Objective
    obj_term: Dict[int, Dict[int, Variable]] = {}
    for pins in e_prime_set.values():
        for p, q in combinations(pins, 2):
            if p not in obj_term:
                obj_term[p] = {}
            obj_term[p][q] = mdl.addVar(name="O(%s,%s)" % (p, q))
            u, v = pin_to_node_map[p], pin_to_node_map[q]
            mdl.addCons(
                obj_term[p][q]
                == x[u] ** 2
                + xpo[p] ** 2
                + 2 * x[u] * xpo[p]
                + x[v] ** 2
                + xpo[q] ** 2
                + 2 * x[v] * xpo[q]
                - 2 * x[u] * x[v]
                - 2 * xpo[p] * xpo[q]
                - 2 * x[u] * xpo[q]
                - 2 * x[v] * xpo[p]
                + y[u] ** 2
                + ypo[p] ** 2
                + 2 * y[u] * ypo[p]
                + y[v] ** 2
                + ypo[q] ** 2
                + 2 * y[v] * ypo[q]
                - 2 * y[u] * y[v]
                - 2 * ypo[p] * ypo[q]
                - 2 * y[u] * ypo[q]
                - 2 * y[v] * ypo[p]
            )
    print(f"{sum(len(val) for val in obj_term.values())} O variables declared")
    print("Pre-objective definition end")
    mdl.setObjective(
        quicksum(quicksum(obj_term[p].values()) for p in obj_term), "minimize"
    )
    print("Objective definition end")
    mdl.data = x_var, y_var
    mdl.setParam("limits/gap", 0.1)  # stop when optimality gap >= 10%

    return mdl


def return_sol(mdl: Model) -> Tuple[Dict[int, float], Dict[int, float]]:
    mdl.optimize()
    if mdl.getStatus() == "infeasible":
        return {}, {}
    x, y = mdl.data
    x_dict = {v: mdl.getVal(x[v]) for v in x}
    y_dict = {v: mdl.getVal(y[v]) for v in y}
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
