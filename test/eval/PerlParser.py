import ast
import datetime
import logging
import re
import subprocess
import sys
from pathlib import PurePath
from typing import Any, Dict, List, Tuple

from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

INS_PARENT_DIR = "/home/junetech/data/"
INS_DIR_DICT = {
    "ispd_benches/ispd2005": [
        "adaptec1",
        "adaptec2",
        "adaptec3",
        "adaptec4",
        "bigblue1",
        "bigblue2",
        "bigblue3",
        "bigblue4",
    ],
    "ispd_benches/ispd2006": [
        "adaptec5",
        "newblue1",
        "newblue2",
        "newblue3",
        "newblue4",
        "newblue5",
        "newblue6",
        "newblue7",
    ],
}
NODES_SUFFIX, INIT_PL_SUFFIX, NETS_SUFFIX, SCL_SUFFIX = ".nodes", ".pl", ".nets", ".scl"
RESULT_PARENT_DIR = "/home/junetech/data/dp20240312_2/"
RESULT_FOLDER_PREFIX = "result_rs"
RAND_SEED_LIST = [1001 + i for i in range(30)]
RESULT_PL_SUFFIX = ".gp.pl"

OUTPUT_FILENAME = "dp20240312_2_perl.xlsx"

HPWL_PATTERN = r"^TotalHPWL:(\d+)"


def get_hpwl(split_stdout: list[bytes]):
    hpwl_regex = re.compile(HPWL_PATTERN)
    for line_b in split_stdout:
        line_str = line_b.decode("utf-8").replace(" ", "")
        mo = hpwl_regex.search(line_str)
        if mo:
            hpwl_val = ast.literal_eval(mo.group(1))
            return hpwl_val
    raise ValueError("No HPWL rows were found")


def get_perl_result() -> (
    Tuple[List[Tuple[str, str, int]], Dict[str, Dict[str, Dict[str, Any]]]]
):
    """
    Returns
    - list of benchmark instance IDs
    - dictionary: benchmark instance ID -> process name -> key -> value
    """
    run_list: List[Tuple[str, str, int]] = []  # list of (ins_name, result_dir, idx)
    #
    run_proc_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for comp_folder, ins_names in INS_DIR_DICT.items():
        for ins_name in ins_names:
            input_path = PurePath(INS_PARENT_DIR, comp_folder, ins_name)
            nodes_path = PurePath(input_path, ins_name + NODES_SUFFIX)
            init_pl_path = PurePath(input_path, ins_name + INIT_PL_SUFFIX)
            nets_path = PurePath(input_path, ins_name + NETS_SUFFIX)
            scl_path = PurePath(input_path, ins_name + SCL_SUFFIX)

            for rand_seed in RAND_SEED_LIST:
                logging.info(
                    f"Processing instance {ins_name} with rand_seed {rand_seed}"
                )
                result_folder = RESULT_FOLDER_PREFIX + str(rand_seed)
                result_path = PurePath(RESULT_PARENT_DIR, result_folder, ins_name)
                result_pl_path = PurePath(result_path, ins_name + RESULT_PL_SUFFIX)
                run_list.append((ins_name, result_folder, 1))
                run_id = result_folder + ins_name + "1"
                run_proc_dict[run_id] = {}

                # HPWL
                split_stdout = subprocess.run(
                    [
                        "perl",
                        "hpwl.pl",
                        nodes_path,
                        init_pl_path,
                        result_pl_path,
                        nets_path,
                    ],
                    stdout=subprocess.PIPE,
                ).stdout.splitlines()
                run_proc_dict[run_id]["Detailedplacement"] = {
                    "w_hpwl": get_hpwl(split_stdout)
                }
                # Overflow

    return run_list, run_proc_dict


def write_to_xlsx(
    run_list: List[Tuple[str, str, int]],
    run_proc_dict: Dict[str, Dict[str, Dict[str, Any]]],
    output_filename: str,
):
    logging.info(f"Writing to {output_filename}")
    proc_header_seq = ["DP"]
    log_proc_dict = {
        "Input": "Input",
        "IP": "Initialplacement",
        "GP": "Globalplacement",
        "LG": "Legalization",
        "DP": "Detailedplacement",
        "Output": "Output",
    }
    wb = Workbook()

    # wHPWL sheet
    w_hpwl_ws: Worksheet = wb.active
    w_hpwl_ws.title = "wHPWL"
    # # overflow sheet
    # overflow_ws: Worksheet = wb.create_sheet("overflow")
    col_header = ["Instance", "ResultDir", "#"] + proc_header_seq
    w_hpwl_ws.append(col_header)
    # overflow_ws.append(col_header)

    for ins_name, result_dir, idx in run_list:
        run_id = result_dir + ins_name + str(idx)
        wHPWL_dict: Dict[str, Any] = {}
        # overflow_dict: Dict[str, Any] = {}
        for proc in proc_header_seq:
            log_proc = log_proc_dict[proc]
            val_dict = run_proc_dict[run_id][log_proc]
            wHPWL_dict[proc] = val_dict["w_hpwl"]
            # overflow_dict[proc] = val_dict["overflow"]
        wHPWL_row = [ins_name, result_dir, idx] + [
            wHPWL_dict[proc_header] for proc_header in proc_header_seq
        ]
        w_hpwl_ws.append(wHPWL_row)

    wb.save(filename=output_filename)


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Logging to a .log file
    handler = logging.FileHandler("log_parser.log", encoding="utf-8")
    root_logger.addHandler(handler)
    # show log messages on terminal as well
    root_logger.addHandler(logging.StreamHandler(sys.stdout))

    global START_DT
    logging.info(f"{__name__} program start @ {START_DT}"[:-3])

    run_list, run_proc_dict = get_perl_result()
    write_to_xlsx(run_list, run_proc_dict, OUTPUT_FILENAME)


if __name__ == "__main__":
    START_DT = datetime.datetime.now()
    main()
    END_DT = datetime.datetime.now()
    elapsed_d = END_DT - START_DT
    logging.info(
        f"{__name__} program end @ {END_DT}"[:-3] + f"; took total {elapsed_d}"[:-3]
    )
