import logging
import time

import numpy as np

import dreamplace.configure as configure
from params import Params
from place_db import PlaceDB


def place(params: Params):
    assert (not params.gpu) or configure.compile_configurations[
        "CUDA_FOUND"
    ] == "TRUE", "CANNOT enable GPU without CUDA compiled"

    np.random.seed(params.random_seed)
    # read database
    tt = time.time()
    placedb = PlaceDB()
    placedb(params)
    proc_time = time.time() - tt
    logging.info(f"Process: Input takes {proc_time:.3f} sec")
