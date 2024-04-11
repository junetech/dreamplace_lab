import datetime
import logging
from typing import Dict, List, Tuple

import cvxpy as cp
import networkx as nx
import numpy as np
import numpy.typing as npt
from dreamplace.NetworkCalc import (
    calc_laplacian,
    create_simple_graph,
    create_virtual_graph,
)
from dreamplace.Partitioner import partition
from dreamplace.PlaceDB import PlaceDB
from dreamplace.VirtualElem import StarVDB, VPinDB, VPinStat, PartitionDB


class MyProb(cp.Problem):
    mv_n_id: npt.NDArray[np.int32]
    x: cp.Variable
    y: cp.Variable


def do_initial_place(placedb: PlaceDB) -> Tuple[Dict[int, float], Dict[int, float]]:
    s_dt = datetime.datetime.now()
    # TODO: define these values outside the code
    large_mv_node_portion = 0.00001
    large_fx_node_portion = 0.0001

    # alias
    mv_node_count = placedb.num_movable_nodes
    fx_node_count = placedb.num_terminals
    all_node_count = mv_node_count + fx_node_count

    # largest movable nodes
    large_node_count = np.ceil(mv_node_count * large_mv_node_portion).astype(int)
    logging.info(
        "  Among %d movable nodes, %d largest nodes are selected"
        % (mv_node_count, large_node_count)
    )
    # print(
    #     placedb.node_size_x[: mv_node_count]
    #     * placedb.node_size_y[: mv_node_count]
    # )
    large_mv_n_id_set = set(
        np.argpartition(
            placedb.node_size_x[:mv_node_count] * placedb.node_size_y[:mv_node_count],
            -large_node_count,
        )[-large_node_count:]
    )
    # print(type(large_mv_n_id_set), large_mv_n_id_set)

    # largest fixed nodes
    large_node_count = np.ceil(fx_node_count * large_fx_node_portion).astype(int)
    logging.info(
        "  Among %d fixed nodes, %d largest nodes are selected"
        % (fx_node_count, large_node_count)
    )
    large_fx_n_id_set = set(
        np.argpartition(
            placedb.node_size_x[mv_node_count:all_node_count]
            * placedb.node_size_y[mv_node_count:all_node_count],
            -large_node_count,
        )[-large_node_count:]
        + mv_node_count
    )
    # print(type(large_fx_n_id_set), large_fx_n_id_set)

    # make virtual pins
    vpin_db = create_vpin_db(placedb, large_mv_n_id_set, large_fx_n_id_set)
    logging.info(f"Virtual pin definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

    # create simple graph
    simple_graph, star_vertices = create_simple_graph(
        vpin_db, placedb.net2pin_map, mv_node_count, all_node_count
    )
    vpin_db.preset_vp_id_list(star_vertices.is_selected)
    logging.info(
        f"Simple graph creation definition takes {datetime.datetime.now()-s_dt}"[:-3]
    )
    s_dt = datetime.datetime.now()
    # partition movable vpins
    ms_mt_partition = partition(simple_graph, star_vertices, vpin_db)
    # ms_mt_partition[3] += 1
    logging.info(f"Partitioning simple graph takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

    partition_db = PartitionDB()
    partition_db.ms_mt_partition_dict = ms_mt_partition

    # create virtual graph
    vgraph, other_partition = create_virtual_graph(simple_graph, ms_mt_partition)
    partition_db.other_partition_dict = other_partition
    logging.info(
        f"Virtual graph creation definition takes {datetime.datetime.now()-s_dt}"[:-3]
    )
    partition_db.report()
    s_dt = datetime.datetime.now()

    # create QP model
    mdl = make_qp_model(
        placedb,
        vpin_db,
        star_vertices,
        vgraph,
        partition_db,
    )

    logging.info(
        f"Initial-placing large nodes: math model building takes {datetime.datetime.now() - s_dt} sec."
    )

    s_dt = datetime.datetime.now()
    x_dict, y_dict = return_sol(mdl)
    logging.info(
        f"Initial-placing large nodes: math model solving takes {datetime.datetime.now() - s_dt} sec."
    )

    return x_dict, y_dict


def create_vpin_db(
    placedb: PlaceDB,
    large_mv_n_id_set: set[int],
    large_fx_n_id_set: set[int],
) -> VPinDB:
    # node count
    mv_node_count = placedb.num_movable_nodes
    fx_node_count = placedb.num_terminals
    all_node_count = mv_node_count + fx_node_count
    # node index
    mv_n_id_array = np.arange(start=0, stop=mv_node_count)
    fx_n_id_array = np.arange(start=mv_node_count, stop=all_node_count)
    # node width & height
    n_wth, n_hgt = placedb.node_size_x, placedb.node_size_y
    # original node to pin (array of arrays)
    node2pin_map = placedb.node2pin_map
    # original pin offset
    pin_offset_x, pin_offset_y = placedb.pin_offset_x, placedb.pin_offset_y

    # \cal{P}^0(n): node ID -> list of virtual pin ID
    vp_id_dict: dict[int, list[int]] = {}
    # N <- \cal{P}^0
    _vpin2node_map = []
    # f(p): original pin ID -> virtual pin ID
    pin2vpin_map = np.zeros(placedb.num_pins, dtype=np.int32)
    # list of virtual pin offset
    _vpin_offset_x: list[float] = []
    _vpin_offset_y: list[float] = []

    _is_movable_node = [True] * mv_node_count + [False] * fx_node_count
    _is_small_node = [True] * all_node_count

    def pin_into_vpin(
        n_id_array: npt.NDArray[np.int32], large_n_id_set: set[int], vp_start: int
    ) -> int:
        vp_count = vp_start

        for n_id in n_id_array:
            original_pin_list = node2pin_map[n_id]
            original_pin_count = len(original_pin_list)
            half_wth, half_hgt = n_wth[n_id] / 2, n_hgt[n_id] / 2

            if n_id not in large_n_id_set:
                # case 1: small node
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
                # is a large node
                _is_small_node[n_id] = False
                if original_pin_count <= 4:
                    # case 2: large node with few pins
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
                    # case 3: large node with lots of pins
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

    vpin_db = VPinDB()
    vpin_db.mv_vp_count = mv_vp_count
    vpin_db.fx_vp_count = fx_vp_count
    vpin_db.vpin_offset_x = np.array(_vpin_offset_x, dtype=np.float32)
    vpin_db.vpin_offset_y = np.array(_vpin_offset_y, dtype=np.float32)
    vpin_db.is_movable_node = np.array(_is_movable_node, dtype=bool)
    vpin_db.is_small_node = np.array(_is_small_node, dtype=bool)
    vpin_db.node2vpin_array_dict = {
        n_id: np.array(sorted(vp_id_dict[n_id]), dtype=np.int32) for n_id in vp_id_dict
    }
    vpin_db.pin2vpin_map = pin2vpin_map
    vpin_db.vpin2node_map = np.array(_vpin2node_map, dtype=np.int32)

    logging.info("  Total %d virtual pins defined" % (total_vp_count))
    vpin_stat.create_log()
    return vpin_db


def make_qp_model(
    placedb: PlaceDB,
    vpin_db: VPinDB,
    star_v_db: StarVDB,
    vgraph: nx.Graph,
    partition_db: PartitionDB,
) -> MyProb:
    s_dt = datetime.datetime.now()

    # Index definition
    # \cal{N}^m: set of movable nodes
    mv_n_id_array = np.array(
        [
            n_id
            for n_id in range(placedb.num_movable_nodes)
            if star_v_db.is_selected[n_id]
        ],
        dtype=np.int32,
    )
    mv_node_count = len(mv_n_id_array)
    # \cal{N}^f: set of fixed nodes
    fx_n_id_array = np.array(
        [
            n_id
            for n_id in range(
                placedb.num_movable_nodes,
                placedb.num_movable_nodes + placedb.num_terminals,
            )
            if star_v_db.is_selected[n_id]
        ],
        dtype=np.int32,
    )
    # \cal{P}(n): node -> set of pins
    node2vpin = vpin_db.node2vpin_array_dict

    # \cal{P}: set of pins
    m_pin_count = sum(len(node2vpin[n_id]) for n_id in mv_n_id_array)

    # V: set of vertices
    # partition for movable pins \union partition for a star vertex
    ms_vertices: set[int] = set()
    ml_vertices: set[int] = set()
    mt_vertices: set[int] = set()
    ft_vertices: set[int] = set()
    # fs_vertices: set[int] = set()
    # fl_vertices: set[int] = set()
    fs_fl_vertices: set[int] = set()

    for n_id in mv_n_id_array:
        if vpin_db.is_small_node[n_id]:
            p_id = node2vpin[n_id][0]  # has only single pin
            ms_vertices.add(partition_db.ms_mt_partition_dict[p_id])
        else:
            for p_id in node2vpin[n_id]:
                ml_vertices.add(partition_db.other_partition_dict[p_id])
    for mv_star_v in star_v_db.mv_star_v_id_list:
        mt_vertices.add(partition_db.ms_mt_partition_dict[mv_star_v])
    for fx_star_v in star_v_db.fx_star_v_id_list:
        ft_vertices.add(partition_db.other_partition_dict[fx_star_v])
    for n_id in fx_n_id_array:
        # if vpin_db.is_small_node[n_id]:
        #     p_id = node2vpin[n_id][0]  # has only single pin
        #     fs_vertices.add(partition_db.other_partition_dict[p_id])
        # else:
        #     for p_id in node2vpin[n_id]:
        #         fl_vertices.add(partition_db.other_partition_dict[p_id])
        for p_id in node2vpin[n_id]:
            fs_fl_vertices.add(partition_db.other_partition_dict[p_id])

    concat_list: list[list[int]] = []
    for v_set in [ms_vertices, ml_vertices, mt_vertices, ft_vertices]:
        if v_set:
            concat_list.append(sorted(v_set))
    vertices_1 = np.concatenate(concat_list, dtype=np.int32)
    # vertices_2 = np.concatenate(
    #     (
    #         sorted(fs_vertices),
    #         sorted(fl_vertices),
    #     ),
    #     dtype=np.int32,
    # )
    vertices_2 = np.array(sorted(fs_fl_vertices), dtype=np.int32)

    all_vertices = np.concatenate((vertices_1, vertices_2), dtype=np.int32)
    logging.info(f"Index definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

    # Parameter definition start
    # Block area
    block_wth, block_hgt = placedb.xh - placedb.xl, placedb.yh - placedb.yl
    # node width & height
    n_wth, n_hgt = placedb.node_size_x, placedb.node_size_y
    # Pin offset
    vpin_offset_x = vpin_db.vpin_offset_x
    vpin_offset_y = vpin_db.vpin_offset_y

    # Position of fixed pins
    # fixed node ID -> pin ID -> position
    _x_pf, _y_pf = {}, {}
    num_v1 = len(vertices_1)
    _xp_vf = [0] * len(vertices_2)
    _yp_vf = [0] * len(vertices_2)
    for n_id in fx_n_id_array:
        _x_pf = {
            p_id: placedb.node_x[n_id] + vpin_offset_x[p_id] for p_id in node2vpin[n_id]
        }
        _y_pf = {
            p_id: placedb.node_y[n_id] + vpin_offset_y[p_id] for p_id in node2vpin[n_id]
        }
        # if vpin_db.is_small_node[n_id]:
        #     p_id = node2vpin[n_id][0]  # has only single pin
        #     _xp_vf[partition_db.other_partition_dict[p_id] - num_v1] = _x_pf[p_id]
        #     _yp_vf[partition_db.other_partition_dict[p_id] - num_v1] = _y_pf[p_id]
        # else:
        for p_id in node2vpin[n_id]:
            _xp_vf[partition_db.other_partition_dict[p_id] - num_v1] = _x_pf[p_id]
            _yp_vf[partition_db.other_partition_dict[p_id] - num_v1] = _y_pf[p_id]

    xp_vf = np.array(_xp_vf, dtype=np.float32)
    yp_vf = np.array(_yp_vf, dtype=np.float32)

    vpin_id_list = []
    for n_id in mv_n_id_array:
        vpin_id_list.extend(node2vpin[n_id])
    vpin_id2idx_map = {vpin_id: idx for idx, vpin_id in enumerate(vpin_id_list)}

    logging.info(
        f"Parameter definition before Laplacian takes {datetime.datetime.now()-s_dt}"[
            :-3
        ]
    )
    s_dt = datetime.datetime.now()
    # Create Laplacian matrix with pins
    L_matrix = calc_laplacian(vgraph, all_vertices)
    L_mm = L_matrix[0:num_v1, 0:num_v1]
    L_fm = L_matrix[num_v1:, 0:num_v1]
    L_ff = L_matrix[num_v1:, num_v1:]

    logging.info(f"Graph Laplacian calc. takes {datetime.datetime.now()-s_dt}"[:-3])
    logging.info(f"  Size of Lagrangian matrix is {L_matrix.shape}")
    s_dt = datetime.datetime.now()
    # Parameters end

    # Variables
    x_m = cp.Variable(mv_node_count)
    x_pm = cp.Variable(m_pin_count)
    xp_vm = cp.Variable(num_v1)
    y_m = cp.Variable(mv_node_count)
    y_pm = cp.Variable(m_pin_count)
    yp_vm = cp.Variable(num_v1)
    logging.info(f"Variable definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

    # Constraints
    constrs: List[cp.Constraint] = []
    for n_idx, n_id in enumerate(mv_n_id_array):
        # Movable cells should not be placed outside placement area
        constrs.extend(
            [
                x_m[n_idx] >= 0,
                x_m[n_idx] + n_wth[n_id] <= block_wth,
                y_m[n_idx] >= 0,
                y_m[n_idx] + n_hgt[n_id] <= block_hgt,
            ]
        )
        # Coordinate of pin p of movable node m
        constrs.extend(
            [
                x_pm[vpin_id2idx_map[p_id]] == x_m[n_idx] + vpin_offset_x[p_id]
                for p_id in node2vpin[n_id]
            ]
            + [
                y_pm[vpin_id2idx_map[p_id]] == y_m[n_idx] + vpin_offset_y[p_id]
                for p_id in node2vpin[n_id]
            ]
        )
        # Partition of pin p of movable node m
        x_constr, y_constr = [], []
        if vpin_db.is_small_node[n_id]:
            # is a small node community index
            p_id = node2vpin[n_id][0]  # has only single pin
            p_vm_idx = partition_db.ms_mt_partition_dict[p_id]
            x_constr.append(xp_vm[p_vm_idx] == x_pm[vpin_id2idx_map[p_id]])
            y_constr.append(yp_vm[p_vm_idx] == y_pm[vpin_id2idx_map[p_id]])
        else:
            for p_id in node2vpin[n_id]:
                # is a pin in large node
                p_vm_idx = partition_db.other_partition_dict[p_id]
                x_constr.append(xp_vm[p_vm_idx] == x_pm[vpin_id2idx_map[p_id]])
                y_constr.append(yp_vm[p_vm_idx] == y_pm[vpin_id2idx_map[p_id]])

        constrs.extend(x_constr + y_constr)

    logging.info(f"Constraint definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

    # Objective
    obj_constant = xp_vf.T @ L_ff @ xp_vf + yp_vf.T @ L_ff @ yp_vf
    prob = MyProb(
        objective=cp.Minimize(
            cp.quad_form(xp_vm, L_mm, assume_PSD=True)
            + 2 * xp_vf @ L_fm @ xp_vm
            + cp.quad_form(yp_vm, L_mm, assume_PSD=True)
            + 2 * yp_vf @ L_fm @ yp_vm
            + obj_constant
        ),
        constraints=constrs,
    )
    logging.info(f"Objective definition takes {datetime.datetime.now()-s_dt}"[:-3])
    s_dt = datetime.datetime.now()

    prob.mv_n_id = mv_n_id_array
    prob.x, prob.y = x_m, y_m
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
