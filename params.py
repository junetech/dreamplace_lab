import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class ParamsParent:
    aux_input: str
    lef_input: str
    def_input: str
    verilog_input: str
    gpu: bool  # is int in JSON
    num_bins_x: int
    num_bins_y: int
    global_place_stages: Dict[str, Any]
    target_density: float
    density_weight: float
    random_seed: int
    result_dir: str
    scale_factor: float
    shift_factor: List[float]
    ignore_net_degree: int
    gp_noise_ratio: float
    enable_fillers: bool  # is int in JSON
    global_place_flag: bool  # is int in JSON
    legalize_flag: bool  # is int in JSON
    detailed_place_flag: bool  # is int in JSON
    stop_overflow: float
    dtype: str
    detailed_place_engine: str
    detailed_place_command: str
    plot_flag: bool  # is int in JSON
    RePlAce_ref_hpwl: int
    RePlAce_LOWER_PCOF: float
    RePlAce_UPPER_PCOF: float
    gamma: float
    RePlAce_skip_energy_flag: bool  # is int in JSON
    random_center_init_flag: bool  # is int in JSON
    sort_nets_by_degree: bool  # is int in JSON
    num_threads: int
    dump_global_place_solution_flag: bool  # is int in JSON
    dump_legalize_solution_flag: bool  # is int in JSON
    routability_opt_flag: bool  # is int in JSON
    route_num_bins_x: int
    route_num_bins_y: int
    node_area_adjust_overflow: float
    max_num_area_adjust: int
    adjust_nctugr_area_flag: bool  # is int in JSON
    adjust_rudy_area_flag: bool  # is int in JSON
    adjust_pin_area_flag: bool  # is int in JSON
    area_adjust_stop_ratio: float
    route_area_adjust_stop_ratio: float
    pin_area_adjust_stop_ratio: float
    unit_horizontal_capacity: float
    unit_vertical_capacity: float
    unit_pin_capacity: float
    max_route_opt_adjust_rate: float
    route_opt_adjust_exponent: float
    pin_stretch_ratio: float
    max_pin_opt_adjust_rate: float
    deterministic_flag: float  # is int in JSON
    timing_opt_flag: float  # is int in JSON
    early_lib_input: str
    late_lib_input: str
    lib_input: str
    sdc_input: str
    wire_resistance_per_micron: int
    wire_capacitance_per_micron: int
    net_weighting_scheme: str
    momentum_decay_factor: float
    enable_net_weighting: int
    max_net_weight: Union[int, str]  # can be a string "inf"

    log_filename: str


class Params(ParamsParent):
    def __init__(self, kwargs_dict: Dict[str, Any], input_dict: Dict[str, Any]):
        super().__init__(**kwargs_dict)
        self.params_dict = input_dict

    def printWelcome(self):
        """
        @brief print welcome message
        """
        content = """\
========================================================
                       DREAMPlace
            Yibo Lin (http://yibolin.com)
   David Z. Pan (http://users.ece.utexas.edu/~dpan)
========================================================"""
        print(content)

    def printHelp(self):
        """
        @brief print help message for JSON parameters
        """
        content = self.toMarkdownTable()
        print(content)

    def toMarkdownTable(self):
        """
        @brief convert to markdown table
        """
        key_length = len("JSON Parameter")
        key_length_map = []
        default_length = len("Default")
        default_length_map = []
        description_length = len("Description")
        description_length_map = []

        def getDefaultColumn(key, value):
            if sys.version_info.major < 3:  # python 2
                flag = isinstance(value["default"], unicode)
            else:  # python 3
                flag = isinstance(value["default"], str)
            if flag and not value["default"] and "required" in value:
                return value["required"]
            else:
                return value["default"]

        for key, value in self.params_dict.items():
            key_length_map.append(len(key))
            default_length_map.append(len(str(getDefaultColumn(key, value))))
            description_length_map.append(len(value["description"]))
            key_length = max(key_length, key_length_map[-1])
            default_length = max(default_length, default_length_map[-1])
            description_length = max(description_length, description_length_map[-1])

        content = "| %s %s| %s %s| %s %s|\n" % (
            "JSON Parameter",
            " " * (key_length - len("JSON Parameter") + 1),
            "Default",
            " " * (default_length - len("Default") + 1),
            "Description",
            " " * (description_length - len("Description") + 1),
        )
        content += "| %s | %s | %s |\n" % (
            "-" * (key_length + 1),
            "-" * (default_length + 1),
            "-" * (description_length + 1),
        )
        count = 0
        for key, value in self.params_dict.items():
            content += "| %s %s| %s %s| %s %s|\n" % (
                key,
                " " * (key_length - key_length_map[count] + 1),
                str(getDefaultColumn(key, value)),
                " " * (default_length - default_length_map[count] + 1),
                value["description"],
                " " * (description_length - description_length_map[count] + 1),
            )
            count += 1
        return content

    def toJson(self):
        """
        @brief convert to json
        """
        data = {}
        for key, value in self.__dict__.items():
            if key != "params_dict":
                data[key] = value
        return data

    def fromJson(self, data):
        """
        @brief load form json
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(key, value)

    def dump(self, filename):
        """
        @brief dump to json file
        """
        with open(filename, "w") as f:
            json.dump(self.toJson(), f)

    def load(self, filename):
        """
        @brief load from json file
        """
        with open(filename, "r") as f:
            self.fromJson(json.load(f))


def init_params() -> Params:
    json_path = os.path.join(os.path.dirname(__file__), "dreamplace", "params.json")
    input_dict: Dict[str, Any]
    with open(json_path, "r") as f:
        input_dict = json.load(f, object_pairs_hook=OrderedDict)
    kwargs_dict: Dict[str, Any] = {}
    for key, value in input_dict.items():
        if "default" in value:
            kwargs_dict[key] = value["default"]
        else:
            kwargs_dict[key] = None
    return Params(kwargs_dict, input_dict)
