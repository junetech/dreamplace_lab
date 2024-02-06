import datetime
import logging
from itertools import combinations
from typing import Dict, Union

import numpy as np
from pyscipopt import Model, quicksum
from pyscipopt.scip import Variable

from dreamplace.PlaceDB import PlaceDB


def make_model(placedb: PlaceDB) -> Model:
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

    # pin id -> node id
    pin_to_node_map = placedb.pin2node_map
    # list of pin offset
    xpo, ypo = placedb.pin_offset_x, placedb.pin_offset_y

    # block area
    block_wth, block_hgt = placedb.xh - placedb.xl, placedb.yh - placedb.yl

    # selected subset of movable nodes
    sel_mv_n_id = np.array(
        [n_id for n_id in mv_n_id if n_wth[n_id] * n_hgt[n_id] >= 10000]
    )
    print(len(sel_mv_n_id), "large movable macros selected")

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
    # Parameters end
    # Variables
    for n_id in mv_n_id:
        x[n_id] = mdl.addVar(name="x(%s)" % n_id)
        y[n_id] = mdl.addVar(name="y(%s)" % n_id)

    # Constraints start
    # selected cells should not be placed outside placement area
    for n_id in sel_mv_n_id:
        mdl.addCons(x[n_id] <= block_wth, name="XUB(%s)" % n_id)
        mdl.addCons(x[n_id] + n_wth[n_id] >= 0, name="XLB(%s)" % n_id)
        mdl.addCons(y[n_id] <= block_hgt, name="YUB(%s)" % n_id)
        mdl.addCons(y[n_id] + n_hgt[n_id] >= 0, name="YLB(%s)" % n_id)

    # there is no overlap among selected movable cells and fixed cells
    big_M_x, big_M_y = block_wth, block_hgt
    delta: Dict[int, Dict[int, Dict[int, Variable]]] = {}
    for u in sel_mv_n_id:
        delta = {u: {}}
        for v in np.concatenate((sel_mv_n_id, fx_n_id), axis=None):
            delta[u] = {
                v: {
                    k: mdl.addVar(vtype="I", name="delta(%s,%s,%s)" % (u, v, k))
                    for k in range(1, 5)
                }
            }
            mdl.addCons(
                x[u] <= -n_wth[u] + x[v] + big_M_x * delta[u][v][1],
                name="Ovlp(%s,%s,%s)" % (u, v, 1),
            )
            mdl.addCons(
                x[u] >= x[v] + n_wth[v] - big_M_x * delta[u][v][2],
                name="Ovlp(%s,%s,%s)" % (u, v, 2),
            )
            mdl.addCons(
                y[u] <= -n_hgt[u] + y[v] + big_M_y * delta[u][v][3],
                name="Ovlp(%s,%s,%s)" % (u, v, 3),
            )
            mdl.addCons(
                y[u] >= y[v] + n_hgt[v] - big_M_y * delta[u][v][4],
                name="Ovlp(%s,%s,%s)" % (u, v, 4),
            )
            mdl.addCons(
                quicksum(delta[u][v][k] for k in range(1, 5)) <= 3,
                name="OvlpCnt(%s,%s)" % (u, v),
            )
    print("Add constraints end")
    # Constraints end
    # Objective
    obj_term: Dict[int, Dict[int, Variable]] = {}
    for net_id in range(len(placedb.net2pin_map)):
        pins = placedb.net2pin_map[net_id]
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
    mdl.setObjective(
        quicksum(obj_term[p].values() for p in placedb.pin2node_map), "minimize"
    )
    mdl.data = x, y, delta

    return mdl


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
