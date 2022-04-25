import pandas as pd
import numpy as np
from pathlib import Path


def projectdir(*args, relpath=True):
    root = Path(".")
    while not (root / "Project.toml").is_file():
        if root.resolve() == (root / "..").resolve():
            raise IOError("Cannot find projectdir")
        root = root / ".."
    if relpath:
        return root.joinpath(*args)
    else:
        return root.joinpath(*args).resolve()


BENCHMARKS = projectdir("data", "benchmarks")


def apply_fun(f, fname):
    try:
        data = np.loadtxt(BENCHMARKS/fname)
        return f(data[1:])
    except:
        return np.nan


def get_csv_levels(filename, method, func_name, levels=[3, 4, 5, 6, 7, 8, 9, 10, 15]):
    df = pd.read_csv(BENCHMARKS / filename, header=0)
    try:
        a = df['rss_memory_MB'].to_numpy()
    except:
        print(filename, "is already up to date", "\n")
        return

    a = df['rss_memory_MB'].to_numpy()
    df = df.drop(columns=['rss_memory_MB'])

    name1 = method+"_"+func_name+"_levels="
    name2 = "_T=100_mem.dat"

    vals_max = [apply_fun(np.amax, name1+str(l)+name2) for l in levels]
    vals_min = [apply_fun(np.amin, name1+str(l)+name2) for l in levels]
    vals_median = [apply_fun(np.median, name1+str(l)+name2) for l in levels]

    df.insert(3, 'rss_memory_MB_median', vals_median, True)
    df.insert(3, 'rss_memory_MB_max', vals_max, True)
    df.insert(3, 'rss_memory_MB_min', vals_min, True)

    b = df['rss_memory_MB_max'].to_numpy()
    if np.sum(a-b) > 1:
        print("Something was wrong with", filename)
        print("Original rss_memory_MB_max")
        print(a)
        print("New rss_memory_MB_max")
        print(b)

    df.to_csv(BENCHMARKS / filename, index=False, na_rep='nan')


def get_csv_times(filename, method, func_name, *, level=5, Ts=[20, 50, 100, 200, 300, 400, 500, 800]):
    df = pd.read_csv(BENCHMARKS / filename, header=0)
    try:
        a = df['rss_memory_MB'].to_numpy()
    except:
        print(filename, "is already up to date", "\n")
        return

    a = df['rss_memory_MB'].to_numpy()
    df = df.drop(columns=['rss_memory_MB'])

    name1 = method+"_"+func_name+"_levels="+str(level)+"_T="
    name2 = "_mem.dat"

    vals_max = [apply_fun(np.amax, name1+str(T)+name2) for T in Ts]
    vals_min = [apply_fun(np.amin, name1+str(T)+name2) for T in Ts]
    vals_median = [apply_fun(np.median, name1+str(T)+name2) for T in Ts]

    df.insert(3, 'rss_memory_MB_median', vals_median, True)
    df.insert(3, 'rss_memory_MB_max', vals_max, True)
    df.insert(3, 'rss_memory_MB_min', vals_min, True)

    b = df['rss_memory_MB_max'].to_numpy()
    if np.sum(a-b) > 1:
        print("Something was wrong with", filename)
        print("Original rss_memory_MB_max")
        print(a)
        print("New rss_memory_MB_max")
        print(b)

    df.to_csv(BENCHMARKS / filename, index=False, na_rep='nan')


get_csv_levels('C_benchmark_levels_full_ad.csv', "full_ad",
               "J_T_C", levels=[3, 4, 5, 6, 7, 8])
get_csv_levels('C_benchmark_levels_full_ad_cheby.csv',
               "full_ad_cheby", "J_T_C")
get_csv_levels('C_benchmark_levels_semi_ad.csv', "grape", "J_T_C")
get_csv_levels('C_U_benchmark_levels_semi_ad.csv', "grape", "J_T_C_U")

get_csv_levels('PE_benchmark_levels.csv', "grape", "J_T_PE")
get_csv_levels('PE_benchmark_levels_full_ad.csv', "full_ad",
               "J_T_PE", levels=[3, 4, 5, 6, 7, 8])
get_csv_levels('PE_benchmark_levels_full_ad_cheby.csv',
               "full_ad_cheby", "J_T_PE")
get_csv_levels('PE_U_benchmark_levels_semi_ad.csv', "grape", "J_T_PE_U")

get_csv_levels('SM_benchmark_levels.csv', "grape", "J_T_sm")
get_csv_levels('SM_FullAD_benchmark_levels.csv', "full_ad",
               "J_T_sm", levels=[3, 4, 5, 6, 7, 8])
get_csv_levels('SM_FullADcheby_benchmark_levels.csv',
               "full_ad_cheby", "J_T_sm", levels=[3, 4, 5, 6, 7])
get_csv_levels('SM_SemiAD_benchmark_levels.csv', "grape", "J_T_sm_AD")
get_csv_levels('SM_benchmark_levels.csv', "grape", "J_T_sm")


get_csv_times('C_benchmark_times_full_ad.csv', "full_ad", "J_T_C")
get_csv_times('C_benchmark_times_full_ad_cheby.csv', "full_ad_cheby", "J_T_C")
get_csv_times('C_benchmark_times_semi_ad.csv', "grape", "J_T_C")
get_csv_times('C_U_benchmark_times_semi_ad.csv', "grape", "J_T_C_U")

get_csv_times('PE_benchmark_times.csv', "grape", "J_T_PE")
get_csv_times('PE_benchmark_times_full_ad.csv', "full_ad", "J_T_PE")
get_csv_times('PE_benchmark_times_full_ad_cheby.csv',
              "full_ad_cheby", "J_T_PE")
get_csv_times('PE_U_benchmark_times_semi_ad.csv', "grape", "J_T_PE_U")

get_csv_times('SM_benchmark_times.csv', "grape", "J_T_sm")
get_csv_times('SM_FullAD_benchmark_times.csv', "full_ad",
              "J_T_sm", Ts=[20, 50, 100, 200, 300, 400, 500])
get_csv_times('SM_FullADcheby_benchmark_times.csv', "full_ad_cheby",
              "J_T_sm", Ts=[20, 50, 100, 200, 300, 400, 500])
get_csv_times('SM_SemiAD_benchmark_times.csv', "grape", "J_T_sm_AD")
get_csv_times('SM_benchmark_times.csv', "grape", "J_T_sm")
