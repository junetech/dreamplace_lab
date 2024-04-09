import datetime
import logging
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np
import numpy.typing as npt
from dreamplace.NetworkCalc import create_simple_graph, calc_vpin_laplacian
from dreamplace.PlaceDB import PlaceDB


class MyProb(cp.Problem):
    mv_n_id: npt.NDArray[np.int32]
    x: cp.Variable
    y: cp.Variable


def do_initial_place(placedb: PlaceDB) -> Tuple[Dict[int, float], Dict[int, float]]:
    s_dt = datetime.datetime.now()
    # TODO: define these values outside the code
    large_node_portion = 0.01

    # largest movable nodes
    large_node_count = np.ceil(placedb.num_movable_nodes * large_node_portion).astype(
        int
    )
    logging.info(
        "  Among %d movable nodes, %d largest nodes are selected"
        % (placedb.num_movable_nodes, large_node_count)
    )
    # print(
    #     placedb.node_size_x[: placedb.num_movable_nodes]
    #     * placedb.node_size_y[: placedb.num_movable_nodes]
    # )
    large_mv_n_id_set = set(
        np.argpartition(
            placedb.node_size_x[: placedb.num_movable_nodes]
            * placedb.node_size_y[: placedb.num_movable_nodes],
            -large_node_count,
        )[-large_node_count:]
    )
    # print(type(large_mv_n_id_set), large_mv_n_id_set)

    # largest fixed nodes
    large_node_count = np.ceil(placedb.num_terminals * large_node_portion).astype(int)
    logging.info(
        "  Among %d fixed nodes, %d largest nodes are selected"
        % (placedb.num_terminals, large_node_count)
    )
    # print(
    #     placedb.node_size_x[
    #         placedb.num_movable_nodes : placedb.num_movable_nodes
    #         + placedb.num_terminals
    #     ]
    #     * placedb.node_size_y[
    #         placedb.num_movable_nodes : placedb.num_movable_nodes
    #         + placedb.num_terminals
    #     ]
    # )
    large_fx_n_id_set = set(
        np.argpartition(
            placedb.node_size_x[
                placedb.num_movable_nodes : placedb.num_movable_nodes
                + placedb.num_terminals
            ]
            * placedb.node_size_y[
                placedb.num_movable_nodes : placedb.num_movable_nodes
                + placedb.num_terminals
            ],
            -large_node_count,
        )[-large_node_count:]
        + placedb.num_movable_nodes
    )
    # print(type(large_fx_n_id_set), large_fx_n_id_set)

    mdl = make_qp_model(placedb, large_mv_n_id_set, large_fx_n_id_set)
    logging.info(
        f"Initial-placing large nodes: math model building takes {datetime.datetime.now() - s_dt} sec."
    )

    s_dt = datetime.datetime.now()
    x_dict, y_dict = return_sol(mdl)
    logging.info(
        f"Initial-placing large nodes: math model solving takes {datetime.datetime.now() - s_dt} sec."
    )

    return x_dict, y_dict


class VPinStat:
    def __init__(self):
        # case 1 count
        self.small_node_count = 0
        self.small_node_original_pin_count = 0
        # case 2 count
        self.few_pins_count = 0
        self.few_pins_original_pin_count = 0
        # case 3 count
        self.large_node_many_pins_count = 0
        self.large_node_many_original_pin_count = 0
        self.large_node_many_vpin_count = 0

    def create_log(self):
        logging.info(
            "  %d small nodes have one vpin at the center" % self.small_node_count
        )
        logging.info(
            "    %d pins -> %d vpins"
            % (self.small_node_original_pin_count, self.small_node_count)
        )
        logging.info(
            "  %d nodes with few pins have original pin offset" % self.few_pins_count
        )
        logging.info(
            "    %d pins -> %d vpins"
            % (self.few_pins_original_pin_count, self.few_pins_original_pin_count)
        )
        logging.info(
            "  %d nodes with many pins have at most 4 vpins"
            % self.large_node_many_pins_count
        )
        logging.info(
            "    %d pins -> %d vpins"
            % (self.large_node_many_original_pin_count, self.large_node_many_vpin_count)
        )


def make_qp_model(
    placedb: PlaceDB,
    large_mv_n_id_set: set[int],
    large_fx_n_id_set: set[int],
) -> MyProb:
    s_dt = datetime.datetime.now()

    # Index definition
    # node count
    m_node_count = placedb.num_movable_nodes
    f_node_count = placedb.num_terminals
    # node index
    mv_n_id_array = np.arange(start=0, stop=m_node_count)
    fx_n_id_array = np.arange(start=m_node_count, stop=m_node_count + f_node_count)
    # node width & height
    n_wth, n_hgt = placedb.node_size_x, placedb.node_size_y
    # original node to pin (array of arrays)
    node2pin_map = placedb.node2pin_map
    # original pin offset
    pin_offset_x, pin_offset_y = placedb.pin_offset_x, placedb.pin_offset_y

    # Virtual pin definition
    mv_vp_count, fx_vp_count = 0, 0  # count of all virtual pins created

    # \cal{P}^0(n): node ID -> list of virtual pin ID
    vp_id_dict: dict[int, list[int]] = {}
    # N <- \cal{P}^0
    _vpin2node_map = []
    # f(p): original pin ID -> virtual pin ID
    pin2vpin_map = np.zeros(placedb.num_pins, dtype=np.int32)
    # list of virtual pin offset
    _vpin_offset_x: list[float] = []
    _vpin_offset_y: list[float] = []

    def pin_into_vpin(
        n_id_array: npt.NDArray[np.int32], large_n_id_set: set[int], vp_start: int
    ) -> int:
        vp_count = vp_start
        for n_id in n_id_array:
            original_pin_list = node2pin_map[n_id]
            original_pin_count = len(original_pin_list)
            half_wth, half_hgt = n_wth[n_id] / 2, n_hgt[n_id] / 2

            if n_id not in large_n_id_set:
                # case 1: small nodes
                vp_id_dict[n_id] = [vp_count]
                _vpin2node_map.append(n_id)
                _vpin_offset_x.append(half_wth)
                _vpin_offset_y.append(half_hgt)
                for o_p_id in original_pin_list:
                    pin2vpin_map[o_p_id] = vp_count
                vp_count += 1
                vpin_stat.small_node_count += 1
                vpin_stat.small_node_original_pin_count += original_pin_count
            else:
                if original_pin_count <= 4:
                    # case 2: large nodes with few pins
                    vp_id_dict[n_id] = [vp_count + i for i in range(original_pin_count)]
                    _vpin2node_map.extend([n_id] * original_pin_count)
                    _vpin_offset_x.extend(pin_offset_x[original_pin_list])
                    _vpin_offset_y.extend(pin_offset_y[original_pin_list])
                    for i, o_p_id in enumerate(original_pin_list):
                        pin2vpin_map[o_p_id] = vp_count + i
                    vp_count += original_pin_count
                    vpin_stat.few_pins_count += 1
                    vpin_stat.few_pins_original_pin_count += original_pin_count
                else:
                    # case 3: large nodes with lots of pins
                    vp_id_list: list[int] = []
                    set00: set[int] = set()  # southwest
                    set01: set[int] = set()  # southeast
                    set10: set[int] = set()  # northwest
                    set11: set[int] = set()  # northeast
                    for o_p_id in original_pin_list:
                        # print(o_p_id, pin_offset_x[o_p_id], pin_offset_y[o_p_id])
                        # offset starts from the southwest corner, not center
                        if pin_offset_x[o_p_id] < half_wth:
                            if pin_offset_y[o_p_id] < half_hgt:
                                set00.add(o_p_id)
                                # print("SW")
                            else:
                                set10.add(o_p_id)
                                # print("NW")
                        else:
                            if pin_offset_y[o_p_id] < half_hgt:
                                set01.add(o_p_id)
                                # print("SE")
                            else:
                                set11.add(o_p_id)
                                # print("NE")
                    if len(set00) > 0:
                        vp_id_list.append(vp_count)
                        _vpin_offset_x.append(0)
                        _vpin_offset_y.append(0)
                        for o_p_id in set00:
                            pin2vpin_map[o_p_id] = vp_count
                        vp_count += 1
                    if len(set01) > 0:
                        vp_id_list.append(vp_count)
                        _vpin_offset_x.append(n_wth[n_id])
                        _vpin_offset_y.append(0)
                        for o_p_id in set01:
                            pin2vpin_map[o_p_id] = vp_count
                        vp_count += 1
                    if len(set10) > 0:
                        vp_id_list.append(vp_count)
                        _vpin_offset_x.append(0)
                        _vpin_offset_y.append(n_hgt[n_id])
                        for o_p_id in set10:
                            pin2vpin_map[o_p_id] = vp_count
                        vp_count += 1
                    if len(set11) > 0:
                        vp_id_list.append(vp_count)
                        _vpin_offset_x.append(n_wth[n_id])
                        _vpin_offset_y.append(n_hgt[n_id])
                        for o_p_id in set11:
                            pin2vpin_map[o_p_id] = vp_count
                        vp_count += 1
                    vp_id_dict[n_id] = vp_id_list
                    _vpin2node_map.extend([n_id] * len(vp_id_list))
                    vpin_stat.large_node_many_pins_count += 1
                    vpin_stat.large_node_many_original_pin_count += original_pin_count
                    vpin_stat.large_node_many_vpin_count += len(vp_id_list)
        return vp_count

    vpin_stat = VPinStat()
    mv_vp_count = pin_into_vpin(mv_n_id_array, large_mv_n_id_set, 0)
    total_vp_count = pin_into_vpin(fx_n_id_array, large_fx_n_id_set, mv_vp_count)
    fx_vp_count = total_vp_count - mv_vp_count

    vpin2node_map = np.array(_vpin2node_map, dtype=np.int32)
    vpin_offset_x = np.array(_vpin_offset_x, dtype=np.float32)
    vpin_offset_y = np.array(_vpin_offset_y, dtype=np.float32)

    logging.info(f"Virtual pin definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()
    logging.info("  Total %d virtual pins defined" % (total_vp_count))
    vpin_stat.create_log()

    # Net reduction
    # \cal{P} <- \cal{E}
    simple_graph, npe = create_simple_graph(
        mv_vp_count, fx_vp_count, placedb.net2pin_map, pin2vpin_map
    )
    # \cal{P}
    m_pin_count = len(npe.mv_vp_id_set)
    p_id_array = np.array(
        sorted(npe.mv_vp_id_set.union(npe.fx_vp_id_set)), dtype=np.int32
    )
    pin_id2idx_map = {p_id: idx for idx, p_id in enumerate(p_id_array)}
    # Node reduction
    # \cal{N}^m, \cal{N}^f, \cal{P}(n)
    npe.select_nodes_with_vpins(vp_id_dict, vpin2node_map)
    # \cal{N}^m
    mv_n_id_array = np.array(sorted(npe.mv_n_id_set), dtype=np.int32)
    m_node_count = len(mv_n_id_array)
    # \cal{N}^f
    fx_n_id_array = np.array(sorted(npe.fx_n_id_set), dtype=np.int32)
    # \cal{P}(n)
    vp_id_array_dict: dict[int, npt.NDArray[np.int32]] = {
        n_id: np.array(sorted(npe.vp_id_dict[n_id]), dtype=np.int32)
        for n_id in npe.vp_id_dict
    }

    logging.info(f"Virtual net definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

    # Parameter definition start
    # Block area
    block_wth, block_hgt = placedb.xh - placedb.xl, placedb.yh - placedb.yl

    # Position of fixed pins
    _xp_f, _yp_f = [], []
    for n_id in fx_n_id_array:
        _xp_f.extend(
            [
                placedb.node_x[n_id] + vpin_offset_x[p_id]
                for p_id in vp_id_array_dict[n_id]
            ]
        )
        _yp_f.extend(
            [
                placedb.node_y[n_id] + vpin_offset_y[p_id]
                for p_id in vp_id_array_dict[n_id]
            ]
        )
    xp_f = np.array(_xp_f, dtype=np.float32)
    yp_f = np.array(_yp_f, dtype=np.float32)

    logging.info(
        f"Parameter definition before Laplacian takes {datetime.datetime.now()-s_dt}"[
            :-3
        ]
    )
    s_dt = datetime.datetime.now()

    # Create Laplacian matrix with pins
    L_matrix = calc_vpin_laplacian(simple_graph, p_id_array)
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
    for n_idx, n_id in enumerate(mv_n_id_array):
        constrs.extend(
            [
                xn_m[n_idx] + vpin_offset_x[p_id] == xp_m[pin_id2idx_map[p_id]]
                for p_id in vp_id_array_dict[n_id]
            ]
            + [
                yn_m[n_idx] + vpin_offset_y[p_id] == yp_m[pin_id2idx_map[p_id]]
                for p_id in vp_id_array_dict[n_id]
            ]
        )
    for n_idx, n_id in enumerate(mv_n_id_array):
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

    prob.mv_n_id = mv_n_id_array
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
