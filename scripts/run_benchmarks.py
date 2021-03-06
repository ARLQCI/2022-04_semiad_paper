#!/usr/bin/env python
from pathlib import Path
from multiprocessing import Pool
from subprocess import STDOUT, run, SubprocessError

from datetime import datetime
from functools import partial
import time
import sys
import os
import psutil
import numpy as np
import h5py


COLLECT_ONLY = os.environ.get("COLLECT_ONLY", "") == "1"
# Run with env var "COLLECT_ONLY=1" in order to only create new CSV files from
# existing benchmark data.


BENCHMARKS = {
    "PE_benchmark_levels": {
        "method": "grape",
        "args": ["J_T_PE", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "PE_benchmark_times": {
        "method": "grape",
        "args": ["J_T_PE", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "PE_U_benchmark_levels_semi_ad": {
        "method": "grape",
        "args": ["J_T_PE_U", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "PE_U_benchmark_times_semi_ad": {
        "method": "grape",
        "args": ["J_T_PE_U", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "PE_benchmark_levels_full_ad": {
        "method": "full_ad",
        "args": ["J_T_PE", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8],
    },
    "PE_benchmark_times_full_ad": {
        "method": "full_ad",
        "args": ["J_T_PE", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "PE_benchmark_levels_full_ad_cheby": {
        "method": "full_ad_cheby",
        "args": ["J_T_PE", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "PE_benchmark_times_full_ad_cheby": {
        "method": "full_ad_cheby",
        "args": ["J_T_PE", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "C_benchmark_levels_semi_ad": {
        "method": "grape",
        "args": ["J_T_C", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "C_benchmark_times_semi_ad": {
        "method": "grape",
        "args": ["J_T_C", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "C_U_benchmark_levels_semi_ad": {
        "method": "grape",
        "args": ["J_T_C_U", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "C_U_benchmark_times_semi_ad": {
        "method": "grape",
        "args": ["J_T_C_U", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "C_benchmark_levels_full_ad": {
        "method": "full_ad",
        "args": ["J_T_C", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6],
    },
    "C_benchmark_times_full_ad": {
        "method": "full_ad",
        "args": ["J_T_C", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "C_benchmark_levels_full_ad_cheby": {
        "method": "full_ad_cheby",
        "args": ["J_T_C", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "C_benchmark_times_full_ad_cheby": {
        "method": "full_ad_cheby",
        "args": ["J_T_C", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "SM_benchmark_levels": {
        "method": "grape",
        "args": ["J_T_sm", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "SM_benchmark_times": {
        "method": "grape",
        "args": ["J_T_sm", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "SM_SemiAD_benchmark_levels": {
        "method": "grape",
        "args": ["J_T_sm_AD", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "SM_SemiAD_benchmark_times": {
        "method": "grape",
        "args": ["J_T_sm_AD", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 600, 700, 800],
    },
    "SM_FullAD_benchmark_levels": {
        "method": "full_ad",
        "args": ["J_T_sm", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8],
    },
    "SM_FullAD_benchmark_times": {
        "method": "full_ad",
        "args": ["J_T_sm", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500],
    },
    "SM_FullADcheby_benchmark_levels": {
        "method": "full_ad_cheby",
        "args": ["J_T_sm", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    },
    "SM_FullADcheby_benchmark_times": {
        "method": "full_ad_cheby",
        "args": ["J_T_sm", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500],
    },
}


CMD_TIMES_BASE = [
    "julia",
    "--sysimage=semiad_sysimage.so",
    "--project=.",
    "scripts/benchmark_optimization.jl",
]
CMD_MEM_BASE = [
    "julia",
    "--sysimage=semiad_sysimage.so",
    "--project=.",
    "scripts/run_optimization.jl",
]


def get_option(args, option, default):
    option_prefix = option + "="
    for option in args:
        if option.startswith(option_prefix):
            key, value = option.split("=", maxsplit=1)
            return value
    return default


def benchmark_times_alloc(cmd):
    functional = cmd[-3]
    levels = int(cmd[-2])
    T = int(cmd[-1])
    method = get_option(cmd, "--method", "grape")
    filename = "%s_%s_levels=%d_T=%d.jld2" % (method, functional, levels, T)
    file = Path("data", "benchmarks", filename)
    log_file = file.with_suffix(".log")
    fg = 0
    # nanosec_per_fg = np.NaN
    nanosec_per_fg_min = np.NaN
    nanosec_per_fg_max = np.NaN
    nanosec_per_fg_median = np.NaN
    alloc_memory_MB = np.NaN
    status_ok = True
    if not (file.is_file() or COLLECT_ONLY):
        try:
            print("RUN:", " ".join(cmd), ">", log_file, file=sys.stderr)
            with open(log_file, "wb", buffering=0) as proc_log:
                run(cmd, stderr=STDOUT, stdout=proc_log)
            print("DONE", " ".join(cmd), file=sys.stderr)
        except SubprocessError as exc_info:
            print("ERROR for %s: %s" % (" ".join(cmd), exc_info), file=sys.stderr)
            status_ok = False
    if log_file.is_file():
        if "Instability detected. Aborting" in log_file.read_text():
            status_ok = False
    if status_ok:
        try:
            data = h5py.File(file, "r")
            # nanosec_per_fg = data["nanosec_per_fg"][()]
            fg = int(data["fg"][()])
            alloc_memory_MB = data["alloc_memory_MB"][()]
            timesref = data["benchmark_result"][()][1]
            times = data[timesref][()]
            nanosec_per_fg_min = np.min(times) / fg
            nanosec_per_fg_max = np.max(times) / fg
            nanosec_per_fg_median = np.median(times) / fg
        except OSError:
            pass
    return (
        fg,
        nanosec_per_fg_min,
        nanosec_per_fg_max,
        nanosec_per_fg_median,
        alloc_memory_MB,
    )


def benchmark_mem(cmd, baseline_mb=0):
    functional = cmd[-3]
    levels = int(cmd[-2])
    T = int(cmd[-1])
    method = get_option(cmd, "--method", "grape")
    filename = "%s_%s_levels=%d_T=%d_mem.dat" % (method, functional, levels, T)
    return _benchmark_mem(cmd, filename, baseline_mb=baseline_mb)


def _benchmark_mem(cmd, filename, baseline_mb=0):
    file = Path("data", "benchmarks", filename)
    log_file = file.with_suffix(".log")
    mb_min = np.NaN
    mb_max = np.NaN
    mb_median = np.NaN
    status_ok = True
    if not (file.is_file() or COLLECT_ONLY):
        try:
            measurements = [float(baseline_mb)]
            t_start = time.time()
            for i in range(20):
                print("RUN:", " ".join(cmd), ">", log_file, file=sys.stderr)
                with open(log_file, "wb", buffering=0) as proc_log:
                    process = psutil.Popen(cmd, stderr=STDOUT, stdout=proc_log)
                peak_mem = 0
                while process.is_running():
                    if process.status() == psutil.STATUS_ZOMBIE:
                        break
                    mem = 1e-6 * process.memory_info().rss  # in MB
                    mem = mem - baseline_mb
                    if mem > peak_mem:
                        peak_mem = mem
                print("DONE", " ".join(cmd), file=sys.stderr)
                measurements.append(peak_mem)
                if (time.time() - t_start) > (24 * 3600):
                    break
            np.savetxt(file, measurements)
        except psutil.Error as exc_info:
            print(
                "ERROR for %s: %s" % (" ".join(cmd), exc_info), file=sys.stderr
            )
            status_ok = False
    if log_file.is_file():
        if "Instability detected. Aborting" in log_file.read_text():
            status_ok = False
    if status_ok:
        try:
            measurements = np.loadtxt(file)
            baseline_mb = measurements[0]
            mb_min = np.min(measurements[1:])
            mb_max = np.max(measurements[1:])
            mb_median = np.median(measurements[1:])
        except OSError:
            pass
    return baseline_mb, mb_max, mb_min, mb_median


def assemble_cmds(base, spec):
    cmds = []
    for v in spec["vals"]:
        args = [arg.format(**{spec["var"]: v}) for arg in spec["args"]]
        options = []
        for key in ["method", "iters"]:
            if key in spec:
                options.append("--{key}={val}".format(key=key, val=spec[key]))
        cmds.append(base + options + args)
    return cmds


def main():
    num_cpus = len(psutil.Process().cpu_affinity())
    if "NUM_THREADS" in os.environ:
        num_cpus = int(os.environ["NUM_THREADS"])
    tasks_benchmark_times = []
    tasks_benchmark_mem = []
    logfile = Path("data", "benchmarks", "run_benchmarks.log")
    print("See output in", logfile, file=sys.stderr)
    with open(logfile, "a") as log_fh:
        print(
            "Starting at",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S %z"),
            file=log_fh,
        )
        log_fh.flush()
        baseline_mb = _benchmark_mem(
            [
                "julia",
                "--sysimage=semiad_sysimage.so",
                "--project=.",
                "scripts/hello_world.jl",
            ],
            filename="hello_world_mem.dat",
        )[1]
        if not COLLECT_ONLY:
            assert baseline_mb > 0.0
        print("Baseline RSS memory (MB)", baseline_mb, "MB", file=sys.stderr)
        baseline_file = Path("data", "benchmarks", "hello_world_mem.dat")
        baseline_file.unlink(missing_ok=True)
        for (name, spec) in BENCHMARKS.items():
            for cmd in assemble_cmds(CMD_TIMES_BASE, spec):
                print("Schedule: " + " ".join(cmd), file=log_fh)
                tasks_benchmark_times.append(cmd)
            for cmd in assemble_cmds(CMD_MEM_BASE, spec):
                print("Schedule: " + " ".join(cmd), file=log_fh)
                tasks_benchmark_mem.append(cmd)
            log_fh.flush()
        with Pool(num_cpus) as worker:
            data_times = worker.map(
                benchmark_times_alloc, tasks_benchmark_times
            )
        print("data_times = %s" % repr(data_times), file=log_fh)
        benchmark_mem_rel = partial(benchmark_mem, baseline_mb=baseline_mb)
        with Pool(num_cpus // 2) as worker:
            # each memory benchmark takes two cores (process and watcher)
            data_mem = worker.map(benchmark_mem_rel, tasks_benchmark_mem)
        print("data_mem = %s" % repr(data_mem), file=log_fh)
        i = 0  # index in data_times, data_mem
        for (name, spec) in BENCHMARKS.items():
            outfile = Path("data", "benchmarks", "%s.csv" % name)
            print("Writing ", outfile, file=log_fh)
            log_fh.flush()
            with open(outfile, "w") as out_fh:
                print(
                    spec["var"],
                    "fg",
                    "nanosec_per_fg_min",
                    "nanosec_per_fg_max",
                    "nanosec_per_fg_median",
                    "alloc_memory_MB",
                    "rss_memory_MB_min",
                    "rss_memory_MB_max",
                    "rss_memory_MB_median",
                    "rss_baseline_MB",
                    sep=",",
                    file=out_fh,
                )
                for val in spec["vals"]:
                    fg = data_times[i][0]
                    nanosec_per_fg_min = data_times[i][1]
                    nanosec_per_fg_max = data_times[i][2]
                    nanosec_per_fg_median = data_times[i][3]
                    alloc_memory_MB = data_times[i][4]
                    baseline_mb = data_mem[i][0]
                    rss_memory_MB_max = data_mem[i][1]
                    rss_memory_MB_min = data_mem[i][2]
                    rss_memory_MB_median = data_mem[i][3]
                    print(
                        val,
                        fg,
                        nanosec_per_fg_min,
                        nanosec_per_fg_max,
                        nanosec_per_fg_median,
                        alloc_memory_MB,
                        rss_memory_MB_min,
                        rss_memory_MB_max,
                        rss_memory_MB_median,
                        baseline_mb,
                        sep=",",
                        file=out_fh,
                    )
                    i += 1
        assert len(data_times) == len(data_mem) == i
        print(
            "Done at",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S %z"),
            file=log_fh,
        )
        log_fh.flush()


if __name__ == "__main__":
    main()
