##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#

import datetime
import logging
import os
import sys
import time
import traceback

import matplotlib
import numpy as np

matplotlib.use("Agg")
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import dreamplace.configure as configure
import NonLinearPlace
import Params
import PlaceDB
import Timer


def place(params):
    """
    @brief Top API to run the entire placement flow.
    @param params parameters
    """

    assert (not params.gpu) or configure.compile_configurations[
        "CUDA_FOUND"
    ] == "TRUE", "CANNOT enable GPU without CUDA compiled"

    np.random.seed(params.random_seed)
    # read database
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    proc_time = time.time() - tt
    logging.info("reading database takes %.2f seconds" % (proc_time))
    logging.info(f"Process: Input takes {proc_time:.3f} sec")

    # Read timing constraints provided in the benchmarks into out timing analysis
    # engine and then pass the timer into the placement core.
    timer = None
    if params.timing_opt_flag:
        tt = time.time()
        timer = Timer.Timer()
        timer(params, placedb)
        # This must be done to explicitly execute the parser builders.
        # The parsers in OpenTimer are all in lazy mode.
        timer.update_timing()
        proc_time = time.time() - tt
        logging.info("reading timer takes %.2f seconds" % proc_time)

        # Dump example here. Some dump functions are defined.
        # Check instance methods defined in Timer.py for debugging.
        # timer.dump_pin_cap("pin_caps.txt")
        # timer.dump_graph("timing_graph.txt")

    # solve placement
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    # my_place
    proc_time = time.time() - tt
    logging.info("non-linear placement initialization takes %.2f seconds" % proc_time)
    logging.info(f"Process: Initial placement takes {proc_time:.3f} sec")

    tt = time.time()
    metrics = placer(params, placedb)
    proc_time = time.time() - tt
    logging.info("non-linear placement optimization takes %.2f seconds" % proc_time)

    # write placement solution
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    gp_out_file = os.path.join(
        path, "%s.gp.%s" % (params.design_name(), params.solution_file_suffix())
    )
    placedb.write(params, gp_out_file)

    # call external detailed placement
    # TODO: support more external placers, currently only support
    # 1. NTUplace3/NTUplace4h with Bookshelf format
    # 2. NTUplace_4dr with LEF/DEF format
    if params.detailed_place_engine and os.path.exists(params.detailed_place_engine):
        logging.info(
            "Use external detailed placement engine %s" % (params.detailed_place_engine)
        )
        if params.solution_file_suffix() == "pl" and any(
            dp_engine in params.detailed_place_engine
            for dp_engine in ["ntuplace3", "ntuplace4h"]
        ):
            dp_out_file = gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if params.target_density < 1.0 and not params.routability_opt_flag:
                target_density_cmd = " -util %f" % (params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                params.detailed_place_engine,
                params.aux_input,
                gp_out_file,
                target_density_cmd,
                dp_out_file,
                params.detailed_place_command,
            )
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            # my_place
            proc_time = time.time() - tt
            logging.info("External detailed placement takes %.2f seconds" % proc_time)

            if params.plot_flag:
                # read solution and evaluate
                placedb.read_pl(params, dp_out_file + ".ntup.pl")
                iteration = len(metrics)
                pos = placer.init_pos
                pos[0 : placedb.num_physical_nodes] = placedb.node_x
                pos[
                    placedb.num_nodes : placedb.num_nodes + placedb.num_physical_nodes
                ] = placedb.node_y
                hpwl, density_overflow, max_density = placer.validate(
                    placedb, pos, iteration
                )
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density)
                )
                placer.plot(params, placedb, iteration, pos)
        elif "ntuplace_4dr" in params.detailed_place_engine:
            dp_out_file = gp_out_file.replace(".gp.def", "")
            cmd = "%s" % (params.detailed_place_engine)
            for lef in params.lef_input:
                if "tech.lef" in lef:
                    cmd += " -tech_lef %s" % (lef)
                else:
                    cmd += " -cell_lef %s" % (lef)
                benchmark_dir = os.path.dirname(lef)
            cmd += " -floorplan_def %s" % (gp_out_file)
            if params.verilog_input:
                cmd += " -verilog %s" % (params.verilog_input)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (
                # os.path.dirname(params.verilog_input))
                benchmark_dir
            )
            cmd += " -noglobal %s ; " % (params.detailed_place_command)
            # cmd += " %s ; " % (params.detailed_place_command) ## test whole flow
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                dp_out_file
            )
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            # my_place
            proc_time = time.time() - tt
            logging.info("External detailed placement takes %.2f seconds" % proc_time)
        else:
            logging.warning(
                "External detailed placement only supports NTUplace3/NTUplace4dr API"
            )
    elif params.detailed_place_engine:
        logging.warning(
            "External detailed placement engine %s or aux file NOT found"
            % (params.detailed_place_engine)
        )

    return metrics


if __name__ == "__main__":
    """
    @brief main function to invoke the entire placement flow.
    """
    try:
        start_dt = datetime.datetime.now()
        params = Params.Params()

        # my_place
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

        params.printWelcome()
        if len(sys.argv) == 1 or "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
            params.printHelp()
            exit()
        elif len(sys.argv) != 2:
            logging.error("One input parameters in json format is required")
            params.printHelp()
            exit()

        logging.info(f"{__name__} program start @ {start_dt}"[:-3])

        # load parameters
        params.load(sys.argv[1])
        logging.info("parameters = %s" % (params))
        # control numpy multithreading
        os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)

        # run placement
        tt = time.time()
        place(params)
        proc_time = time.time() - tt
        logging.info("placement takes %.3f seconds" % proc_time)

        end_dt = datetime.datetime.now()
        elapsed_d = end_dt - start_dt
        logging.info(
            f"{__name__} program end @ {end_dt}"[:-3] + f"; took total {elapsed_d}"[:-3]
        )
    except:
        logging.error(traceback.format_exc())
