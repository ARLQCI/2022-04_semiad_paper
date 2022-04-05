#!/usr/bin/env python
from pathlib import Path
from multiprocessing import Pool
from subprocess import STDOUT, run, SubprocessError

from datetime import datetime
from functools import partial
import sys
import psutil
import numpy as np
import h5py


BENCHMARKS = {
    "PE_benchmark_levels": {
        "method": "grape",
        "args": ["J_T_PE", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 15],
    },
    "PE_benchmark_times": {
        "method": "grape",
        "args": ["J_T_PE", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 800],
    },
    "PE_benchmark_levels_full_ad": {
        "method": "full_ad",
        "args": ["J_T_PE", "{levels}", "100"],
        "var": "levels",
        "vals": [3, 4, 5, 6, 7, 8, 9, 10, 15],
    },
    "PE_benchmark_times_full_ad": {
        "method": "full_ad",
        "args": ["J_T_PE", "5", "{T}"],
        "var": "T",
        "vals": [20, 50, 100, 200, 300, 400, 500, 800],
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
    if not file.is_file():
        try:
            print("RUN:", " ".join(cmd), file=sys.stderr)
            log_file = file.with_suffix(".log")
            with open(log_file, "wb", buffering=0) as proc_log:
                run(cmd, stderr=STDOUT, stdout=proc_log)
            print("DONE", " ".join(cmd), file=sys.stderr)
        except SubprocessError as exc_info:
            print(
                "ERROR for %s: %s" % (" ".join(cmd), exc_info), file=sys.stderr
            )
            return np.NaN, np.NaN
    try:
        data = h5py.File(file, "r")
        nanosec_per_fg = data["nanosec_per_fg"][()]
        alloc_memory_MB = data["alloc_memory_MB"][()]
        return nanosec_per_fg, alloc_memory_MB
    except OSError:
        return np.NaN, np.NaN


def benchmark_mem(cmd, baseline_mb=0):
    functional = cmd[-3]
    levels = int(cmd[-2])
    T = int(cmd[-1])
    method = get_option(cmd, "--method", "grape")
    filename = "%s_%s_levels=%d_T=%d_mem.dat" % (method, functional, levels, T)
    return _benchmark_mem(cmd, filename, baseline_mb=baseline_mb)


def _benchmark_mem(cmd, filename, baseline_mb=0):
    file = Path("data", "benchmarks", filename)
    if not file.is_file():
        try:
            measurements = []
            for i in range(3):
                print("RUN:", " ".join(cmd), file=sys.stderr)
                log_file = file.with_suffix(".log")
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
            np.savetxt(file, measurements)
        except psutil.Error as exc_info:
            print(
                "ERROR for %s: %s" % (" ".join(cmd), exc_info), file=sys.stderr
            )
            return np.NaN
    try:
        measurements = np.loadtxt(file)
        return np.max(measurements)
    except OSError:
        return np.NaN


def assemble_cmds(base, spec):
    cmds = []
    for v in spec["vals"]:
        args = [arg.format(**{spec["var"]: v}) for arg in spec["args"]]
        options = []
        for key in ["method"]:
            if key in spec:
                options.append("--{key}={val}".format(key=key, val=spec[key]))
        cmds.append(base + options + args)
    return cmds


def main():
    num_cpus = len(psutil.Process().cpu_affinity())
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
        )
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
        benchmark_mem_rel = partial(benchmark_mem, baseline_mb=baseline_mb)
        with Pool(num_cpus // 2) as worker:
            # each memory benchmark takes two cores (process and watcher)
            data_mem = worker.map(benchmark_mem_rel, tasks_benchmark_mem)
        for (name, spec) in BENCHMARKS.items():
            outfile = Path("data", "benchmarks", "%s.csv" % name)
            print("Writing ", outfile, file=log_fh)
            log_fh.flush()
            with open(outfile, "w") as out_fh:
                print(
                    spec["var"],
                    "nanosec_per_fg",
                    "alloc_memory_MB",
                    "rss_memory_MB",
                    "rss_baseline_MB",
                    sep=",",
                    file=out_fh,
                )
                for i in range(len(spec["vals"])):
                    val = spec["vals"][i]
                    nanosec_per_fg = data_times[i][0]
                    alloc_memory_MB = data_times[i][1]
                    rss_memory_MB = data_mem[i]
                    print(
                        val,
                        nanosec_per_fg,
                        alloc_memory_MB,
                        rss_memory_MB,
                        baseline_mb,
                        sep=",",
                        file=out_fh,
                    )
        print(
            "Done at",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S %z"),
            file=log_fh,
        )
        log_fh.flush()


if __name__ == "__main__":
    main()
