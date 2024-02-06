import datetime
import logging
import os
import sys

from Params import init_params
from placer import place

START_DT: datetime.datetime


def main():
    params = init_params()

    # Log config
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_format = logging.Formatter("[%(levelname)-7s] %(name)s - %(message)s")
    # Logging to a .log file
    f_handler = logging.FileHandler(params.log_filename, encoding="utf-8")
    f_handler.setFormatter(log_format)
    root_logger.addHandler(f_handler)
    # Show log messages on terminal as well
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setFormatter(log_format)
    root_logger.addHandler(s_handler)

    global START_DT
    logging.info(f"{__name__} program start @ {START_DT}"[:-3])

    params.printWelcome()
    if len(sys.argv) == 1 or "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        params.printHelp()
        exit()
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format is required")
        params.printHelp()
        exit()

    # load input metadata
    params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

    # run placement
    place(params)


if __name__ == "__main__":
    START_DT = datetime.datetime.now()
    main()
    end_dt = datetime.datetime.now()
    elapsed_d = end_dt - START_DT
    logging.info(
        f"{__name__} program end @ {end_dt}"[:-3] + f"; took total {elapsed_d}"[:-3]
    )
