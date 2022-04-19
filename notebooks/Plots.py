# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: SemiAD Paper
#     language: python
#     name: semi-ad-paper
# ---

# # Plots

from pathlib import Path
import rsmf
import pandas as pd


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


plt = rsmf.setup(projectdir("data", "plots", "quantum-template.tex"))
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7EC4F9", "#FFB97A", "#6EEF6E"]

OUTDIR = projectdir("data", "plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = projectdir("data", "benchmarks")


def plot_comparison(
    file_names1,
    file_names2,
    methods,
    functionals,
    index=1,
    *,
    xlabel="x",
    ylabel="y",
    factor=1,
    logscale=False,
    outfile=None,
    inset=False,
    inset_pos=[0.55, 0.55, 0.4, 0.4],
    inset_index=[0, 1]
):

    fig = plt.figure(aspect_ratio=0.8)
    ax = fig.add_subplot()

    for i in range(len(file_names1)):
        print("loading", BENCHMARKS / file_names1[i])
        df = pd.read_csv(BENCHMARKS / file_names1[i], index_col=0)
        label = methods[i] + " (" + functionals[0] + ")"

        if logscale:
            ax.loglog(
                df.iloc[:, index - 1].dropna() * factor,
                "o-",
                label=label,
                color=COLORS[i],
            )
        else:
            ax.plot(
                df.iloc[:, index - 1].dropna() * factor,
                "o-",
                label=label,
                color=COLORS[i],
            )

    for i in range(len(file_names2)):
        print("loading", BENCHMARKS / file_names2[i])
        df = pd.read_csv(BENCHMARKS / file_names2[i], index_col=0)
        label = methods[i] + " (" + functionals[1] + ")"

        if logscale:
            ax.loglog(
                df.iloc[:, index - 1].dropna() * factor,
                "o-",
                label=label,
                color=COLORS[len(methods) + i],
            )
        else:
            ax.plot(
                df.iloc[:, index - 1].dropna() * factor,
                "o-",
                label=label,
                color=COLORS[len(methods) + i],
            )

    ax.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=2,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if inset:
        axins = ax.inset_axes(inset_pos)
        for i in inset_index:

            df = pd.read_csv(BENCHMARKS / file_names1[i], index_col=0)
            axins.plot(
                df.iloc[:, index - 1].dropna() * factor,
                "o-",
                label=methods[i],
                color=COLORS[i],
            )

            df = pd.read_csv(BENCHMARKS / file_names2[i], index_col=0)
            axins.plot(
                df.iloc[:, index - 1].dropna() * factor,
                "v-",
                label=methods[i],
                color=COLORS[len(methods) + i],
            )

    fig.tight_layout(pad=0.0)

    if outfile != None:
        fig.savefig(OUTDIR / outfile)
        print("Written %s" % (OUTDIR / outfile))

    return fig


level_files_PE = [
    "PE_benchmark_levels.csv",
    "PE_benchmark_levels_full_ad_cheby.csv",
    "PE_benchmark_levels_full_ad.csv",
]
time_files_PE = [
    "PE_benchmark_times.csv",
    "PE_benchmark_times_full_ad_cheby.csv",
    "PE_benchmark_times_full_ad.csv",
]
level_files_SM = [
    "SM_benchmark_levels.csv",
    "SM_FullADcheby_benchmark_levels.csv",
    "SM_FullAD_benchmark_levels.csv",
]
time_files_SM = [
    "SM_benchmark_times.csv",
    "SM_FullADcheby_benchmark_times.csv",
    "SM_FullAD_benchmark_times.csv",
]
level_files_C = [
    "C_benchmark_levels_semi_ad.csv",
    "C_benchmark_levels_full_ad_cheby.csv",
    "C_benchmark_levels_full_ad.csv",
]
time_files_C = [
    "C_benchmark_times_semi_ad.csv",
    "C_benchmark_times_full_ad_cheby.csv",
    "C_benchmark_times_full_ad.csv",
]
methods = ["Semi-AD", "Chebyshev", "ODE"]

# ## Runtime

plot_comparison(
    level_files_PE,
    level_files_C,
    methods,
    ["PE", "C"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="number of transmon levels",
    factor=1e-9,
    outfile="PE_runtimes_levels.pdf",
    inset=True,
)

plot_comparison(
    time_files_PE,
    time_files_C,
    methods,
    ["PE", "C"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="gate duration",
    factor=1e-9,
    outfile="PE_runtimes_times.pdf",
    inset=True,
    inset_pos=[0.1, 0.55, 0.35, 0.35],
)

plot_comparison(
    ["C_benchmark_levels_semi_ad.csv"],
    ["C_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="number of transmon levels",
    outfile="C_U_runtime_levels.pdf",
)

plot_comparison(
    ["C_benchmark_times_semi_ad.csv"],
    ["C_U_benchmark_times_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="gate duration",
    outfile="C_U_runtime_times.pdf",
)

plot_comparison(
    ["PE_benchmark_levels.csv"],
    ["PE_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="number of transmon levels",
    outfile="PE_U_runtime_levels.pdf",
)

plot_comparison(
    ["PE_benchmark_times.csv"],
    ["PE_U_benchmark_times_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="gate duration",
    outfile="PE_U_runtime_times.pdf.pdf",
)

# ## RSS

plot_comparison(
    level_files_PE,
    level_files_C,
    methods,
    ["PE", "C"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="number of transmon levels",
    outfile="PE_memory_levels.pdf",
    inset=True,
    inset_pos=[0.12, 0.65, 0.35, 0.3],
    inset_index=[0],
)

plot_comparison(
    time_files_PE,
    time_files_C,
    methods,
    ["PE", "C"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="gate duration",
    outfile="PE_memory_times.pdf",
    inset=True,
    inset_index=[0],
    inset_pos=[0.12, 0.63, 0.35, 0.3],
)

plot_comparison(
    ["C_benchmark_levels_semi_ad.csv"],
    ["C_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="number of transmon levels",
    outfile="C_U_memory_levels.pdf",
)

plot_comparison(
    ["C_benchmark_times_semi_ad.csv"],
    ["C_U_benchmark_times_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="gate duration",
    outfile="C_U_memory_times.pdf",
)

plot_comparison(
    ["PE_benchmark_levels.csv"],
    ["PE_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="number of transmon levels",
    outfile="PE_U_memory_levels.pdf",
)

plot_comparison(
    ["PE_benchmark_times.csv"],
    ["PE_U_benchmark_times_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="gate duration",
    outfile="PE_U_memory_times.pdf",
)

# ## Allocations

plot_comparison(
    level_files_PE,
    level_files_C,
    methods,
    ["PE", "C"],
    index=2,
    logscale=False,
    ylabel="allocated memory (TB)",
    xlabel="number of transmon levels",
    outfile="PE_allocated_levels.pdf",
    factor=1e-6,
    inset=True,
    inset_pos=[0.5, 0.65, 0.35, 0.3],
    inset_index=[0],
)

plot_comparison(
    time_files_PE,
    time_files_C,
    methods,
    ["PE", "C"],
    index=2,
    logscale=False,
    ylabel="allocated memory (TB)",
    factor=1e-6,
    xlabel="gate duration",
    outfile="PE_allocated_times.pdf",
    inset=True,
    inset_index=[0],
    inset_pos=[0.18, 0.63, 0.35, 0.3],
)

plot_comparison(
    ["C_benchmark_levels_semi_ad.csv"],
    ["C_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=2,
    logscale=False,
    ylabel="allocated memory (MB)",
    xlabel="number of transmon levels",
    outfile="C_U_allocated_levels.pdf",
)

plot_comparison(
    ["C_benchmark_times_semi_ad.csv"],
    ["C_U_benchmark_times_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=2,
    logscale=False,
    ylabel="allocated memory (MB)",
    xlabel="gate duration",
    outfile="C_U_allocated_times.pdf",
)

plot_comparison(
    ["PE_benchmark_levels.csv"],
    ["PE_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=2,
    logscale=False,
    ylabel="allocated memory (MB)",
    xlabel="number of transmon levels",
    outfile="PE_U_allocated_levels.pdf",
)

plot_comparison(
    ["PE_benchmark_times.csv"],
    ["PE_U_benchmark_times_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=2,
    logscale=False,
    ylabel="allocated memory (MB)",
    xlabel="gate duration",
    outfile="PE_U_allocated_times.pdf",
)
