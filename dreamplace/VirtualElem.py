import logging

import numpy as np
import numpy.typing as npt


class VPinDB:
    # number of virtual pins created
    mv_vp_count: int
    fx_vp_count: int

    @property
    def vp_count(self):
        return self.mv_vp_count + self.fx_vp_count

    pin2vpin_map: npt.NDArray[np.int32]
    vpin2node_map: npt.NDArray[np.int32]
    vpin_offset_x: npt.NDArray[np.float32]
    vpin_offset_y: npt.NDArray[np.float32]


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


class StarVDB:
    star_v_count: int

    def __init__(self):
        # cal{P}^m: set of pins in movable nodes selected
        self.mv_vp_id_set: set[int] = set()
        # cal{P}^f: set of pins in fixed nodes selected
        self.fx_vp_id_set: set[int] = set()
        # set of star vertices for nets with fixed pins only
        self.fx_star_v_id_list: list[int] = []
        # set of star vertices for nets with movable pins
        self.mv_star_v_id_list: list[int] = []

    def select_nodes_with_vpins(
        self, vp_id_dict: dict[int, list[int]], vpin2node_map: npt.NDArray[np.int32]
    ):
        # \cal{N}^m: set of movable nodes selected
        self.mv_n_id_set: set[int] = set(vpin2node_map[list(self.mv_vp_id_set)])

        # \cal{N}^f: set of fixed nodes selected
        self.fx_n_id_set: set[int] = set(vpin2node_map[list(self.fx_vp_id_set)])

        # \cal{P}(n): node -> set of vpins
        self.vp_id_dict: dict[int, set[int]] = {}
        n_id_set = self.mv_n_id_set.union(self.fx_n_id_set)
        vp_id_set = self.mv_vp_id_set.union(self.fx_vp_id_set)
        for n_id in n_id_set:
            self.vp_id_dict[n_id] = vp_id_set.intersection(vp_id_dict[n_id])
