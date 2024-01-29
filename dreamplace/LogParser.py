import ast
import datetime
import logging
import re
import sys
from typing import Any, Dict, List

START_DT: datetime.datetime

PLACER_LOG_FILENAME = "dreamplace_lab.log"
PARAMS_PATTERN = r"^\[INFO\]root-parameters=(\{.*\})"
TIME_PATTERN = r"^\[INFO\]root-Process:(.*)takes(\d+\.\d+)sec$"
OBJ_PATTERN = r"^\[INFO\]root-Process:(.*)haswHPWLof(\d+\.\d+)&overflowof(\d+\.\d+)"


def parse_a_line(a_line: str) -> Dict[str, Dict[str, Any]]:
    parsed_dict = {}
    key_str = ""

    _a_line = a_line.replace(" ", "")
    params_regex = re.compile(PARAMS_PATTERN)
    mo = params_regex.search(_a_line)
    if mo:
        key_str = "Input"
        params_dict = ast.literal_eval(mo.group(1))
        if params_dict["def_input"]:
            input_path = params_dict["def_input"]
        elif params_dict["aux_input"]:
            input_path = params_dict["aux_input"]
        parsed_dict[key_str] = {"path": input_path}
    else:
        time_regex = re.compile(TIME_PATTERN)
        mo = time_regex.search(_a_line)
        if mo:
            key_str, runtime = mo.groups()
            parsed_dict[key_str] = {"runtime": runtime}
        else:
            obj_regex = re.compile(OBJ_PATTERN)
            mo = obj_regex.search(_a_line)
            if mo:
                key_str, w_hpwl, overflow = mo.groups()
                parsed_dict[key_str] = {"w_hpwl": w_hpwl, "overflow": overflow}

    return parsed_dict


def parse():
    ins_count: Dict[str, int] = {}
    ins_list: List[str] = []
    ins_proc_dict: Dict[str, dict[str, float]] = {}

    with open(PLACER_LOG_FILENAME) as p_log:
        ins_key: str = ""
        for line in p_log:
            _dict = parse_a_line(line)
            if _dict:
                if "Input" in _dict:
                    if "path" in _dict["Input"]:
                        ins_path = _dict["Input"]["path"]
                        if ins_path not in ins_count:
                            ins_count[ins_path] = 1
                        else:
                            ins_count[ins_path] += 1
                        ins_key = ins_path + str(ins_count[ins_path])
                        ins_list.append(ins_key)
                if ins_key not in ins_proc_dict:
                    ins_proc_dict[ins_key] = {}
                for val_key, value in _dict.items():
                    if val_key not in ins_proc_dict[ins_key]:
                        ins_proc_dict[ins_key][val_key] = {}
                    ins_proc_dict[ins_key][val_key].update(value)

    from pprint import pprint

    pprint(ins_proc_dict)


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

    parse()


if __name__ == "__main__":
    START_DT = datetime.datetime.now()
    main()
    END_DT = datetime.datetime.now()
    elapsed_d = END_DT - START_DT
    logging.info(
        f"{__name__} program end @ {END_DT}"[:-3] + f"; took total {elapsed_d}"[:-3]
    )
