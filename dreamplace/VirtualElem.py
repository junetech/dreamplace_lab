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

    # virtual pin offset from southwest corner
    vpin_offset_x: npt.NDArray[np.float32]
    vpin_offset_y: npt.NDArray[np.float32]

    # node classification
    is_movable_node: npt.NDArray[bool]
    is_small_node: npt.NDArray[bool]

    # pin mapping
    node2vpin_array_dict: dict[int, npt.NDArray[np.int32]]
    pin2vpin_map: npt.NDArray[np.int32]
    vpin2node_map: npt.NDArray[np.int32]

    def preset_vp_id_list(self, is_used_array):
        sm_vp_id_list: list[int] = []
        lm_vp_id_list: list[int] = []
        sf_vp_id_list: list[int] = []
        lf_vp_id_list: list[int] = []

        for n_id, is_used in enumerate(is_used_array):
            if not is_used:
                print("Node", n_id, "is not used")
                continue
            if self.is_movable_node[n_id]:
                if self.is_small_node[n_id]:
                    sm_vp_id_list.extend(self.node2vpin_array_dict[n_id])
                else:
                    lm_vp_id_list.extend(self.node2vpin_array_dict[n_id])
            else:
                if self.is_small_node[n_id]:
                    sf_vp_id_list.extend(self.node2vpin_array_dict[n_id])
                else:
                    lf_vp_id_list.extend(self.node2vpin_array_dict[n_id])

        self.small_movable_vp_id_list = np.array(sm_vp_id_list, dtype=np.int32)
        self.large_movable_vp_id_list = np.array(lm_vp_id_list, dtype=np.int32)
        self.small_fixed_vp_id_list = np.array(sf_vp_id_list, dtype=np.int32)
        self.large_fixed_vp_id_list = np.array(lf_vp_id_list, dtype=np.int32)


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
    # set of star vertices for nets with movable pins
    mv_star_v_id_list: npt.NDArray[np.int32]
    # set of star vertices for nets with fixed pins only
    fx_star_v_id_list: npt.NDArray[np.int32]
    # node classification
    is_selected: npt.NDArray[bool]

    def __init__(self):
        self.star_v_count = 0


class PartitionDB:
    # partitions for movable small pins & star vertices
    ms_mt_partition_dict: dict[int, int]
    # partitions for remaining elements
    other_partition_dict: dict[int, int]

    def report(self):
        ms_mt_part_count = len(set(self.ms_mt_partition_dict.values()))
        logging.info(
            "%d movable small nodes & movable star nodes have %d partitions"
            % (len(self.ms_mt_partition_dict), ms_mt_part_count)
        )
        other_part_count = len(set(self.other_partition_dict.values()))
        logging.info(
            "%d other elements have %d partitions"
            % (len(self.other_partition_dict), other_part_count)
        )
